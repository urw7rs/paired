import glob
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import gdown
import librosa as lr
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..pytorch3d.transforms import (
    RotateAxisAngle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
from .preprocess import Normalizer, vectorize_many
from .quaternion import ax_to_6v
from .vis import SMPLSkeleton


class AISTPP(Dataset):
    def __init__(
        self,
        root: str,
        backup_path: str,
        split: str,
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
        stride:float=0.5,
        length:int=5,
    ):
        super().__init__()

        self.build(root, split, stride, length)

        self.data_path = root
        self.raw_fps = 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.split = split

        self.normalizer = normalizer
        self.data_len = data_len

        pickle_name = f"processed_{split}_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        # save normalizer
        if not self.split != "train":
            with open(os.path.join(backup_path, "normalizer.pkl"), "wb") as f:
                pickle.dump(normalizer, f)

        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            data = self.load_aistpp(split)  # Call this last
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print(
            "Loaded {self.split} Dataset With Dimensions: "
            + f"Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )

        # process data, convert to 6dof etc
        pose_input = self.process_dataset(data["pos"], data["q"])
        self.data = {
            "pose": pose_input,
            "filenames": data["filenames"],
            "wavs": data["wavs"],
        }
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        return self.data["pose"][idx], filename_, self.data["wavs"][idx]

    def load_aistpp(self, split):
        """Load AIST++ dataset into a dictionary

        Structure:
        root
          |- train
          |    |- motion_sliced
          |    |- wav_sliced
          |    |- motions
          |    |- wavs
        """

        # open data path
        split_data_path = os.path.join(self.data_path, split)

        motion_path = os.path.join(split_data_path, "motions_sliced")
        wav_path = os.path.join(split_data_path, "wavs_sliced")
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the motions and features together
        all_pos = []
        all_q = []
        all_wavs = []
        all_names = []
        assert len(motions) > 0
        for motion, wav in zip(motions, wavs):
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert m_name == w_name, str((motion, wav))

            # load motion
            with open(motion, "rb") as f:
                data = pickle.load(f)

            pos = data["pos"]
            q = data["q"]
            all_pos.append(pos)
            all_q.append(q)
            all_wavs.append(wav)
            all_names.append(m_name)

        all_pos = np.array(all_pos)  # N x seq x 3
        all_q = np.array(all_q)  # N x seq x (joint * 3)
        # downsample the motions to the data fps
        print(all_pos.shape)
        all_pos = all_pos[:, :: self.data_stride, :]
        all_q = all_q[:, :: self.data_stride, :]
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs}
        return data

    def process_dataset(self, root_pos, local_q):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))

        # AISTPP dataset comes y-up - rotate to z-up
        # to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        # do FK
        positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

        # to 6d
        local_q = ax_to_6v(local_q)

        # now, flatten everything into: batch x sequence x [...]
        global_pose_vec_input = (
            vectorize_many([contacts, root_pos, local_q]).float().detach()
        )

        # normalize the data. Both train and test need the same normalizer.
        if self.split == "train":
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        global_pose_vec_input = global_pose_vec_input

        print(
            f"{self.split} Dataset Motion Features Dim: {global_pose_vec_input.shape}"
        )

        return global_pose_vec_input

    @staticmethod
    def download(path: str = ".", verbose: bool = True, resume: bool = False):
        """Download raw AIST++ dataset

        Arguments:
            path (str): path to download files to
            verbose (bool): print progress bar to conosle
            resume (bool): resume downloads
        """

        url = "https://drive.google.com/u/0/uc?id=16qYnN3qpmHMk2mOvOsOYNLy75xUmbyif"
        md5 = "569a60311ecebb5001c8a7321ba787f3"

        gdown.cached_download(
            url=url,
            path=str(Path(path) / "aistpp.zip"),
            md5=md5,
            quiet=not verbose,
            postprocess=gdown.extractall,
            resume=resume,
        )

    def build(self, root:str, split: str, stride: float = 0.5, length: int = 5):

        def fileToList(f: Path):
            lines = f.read_text().split("\n")
            out = [x.strip() for x in lines]
            out = [x for x in out if len(x)]
            return out

        root = Path(root)
        filter_list = set(fileToList(root / "ignore_list.txt"))
        files = set(fileToList(root / f"splits/crossmodal_{split}.txt"))

        def copy_splits(split, names):
            motion_dir = root / split / "motions"
            wav_dir = root / split / "wavs"

            motion_dir.mkdir(parents=True, exist_ok=True)
            wav_dir.mkdir(parents=True, exist_ok=True)

            for name in tqdm(names):
                if name in filter_list:
                    continue

                motion_path = root / f"motions/{name}.pkl"
                wav_path = root / f"wavs/{name}.wav"

                assert motion_path.is_file()
                assert wav_path.is_file()

                with open(motion_path, "rb") as f:
                    motion_data = pickle.load(f)

                trans = motion_data["smpl_trans"]
                pose = motion_data["smpl_poses"]
                scale = motion_data["smpl_scaling"]
                out_data = {"pos": trans, "q": pose, "scale": scale}

                with open(motion_dir / f"{name}.pkl", "wb") as f:
                    pickle.dump(out_data, f)

                shutil.copyfile(wav_path, wav_dir / f"{name}.wav")

        # split the data according to the splits files
        copy_splits(split, files)

        # slice motions/music into sliding windows to create dataset
        slice_aistpp(
            motion_dir=root / f"{split}/motions",
            wav_dir=root / f"{split}/wavs",
            stride=stride,
            length=length,
        )


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    with open(motion_file, "rb") as f:
        motion = pickle.load(f)

    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    pos /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window and slice_count < num_slices:
        pos_slice, q_slice = (
            pos[start_idx : start_idx + window],
            q[start_idx : start_idx + window],
        )
        out = {"pos": pos_slice, "q": q_slice}

        with open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb") as f:
            pickle.dump(out, f)

        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
    motion_dir = str(motion_dir)
    wav_dir = str(wav_dir)

    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))

    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"

    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)

    assert len(wavs) == len(motions)

    for wav, motion in tqdm(list(zip(wavs, motions))):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )
