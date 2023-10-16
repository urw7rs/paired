import pickle
from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset

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
    """Load AIST++ dataset into a dictionary

    Structure:
    root
      |- train
      |    |- motions
      |    |- wavs
    """

    def __init__(self, root: str, split: str, transforms=None):
        super().__init__()

        self.root = Path(root)
        self.split = split

        # load splits
        names = (self.root / f"splits/crossmodal_{split}.txt").read_text().split("\n")

        # filter names in ignore_list.txt
        lines = (self.root / "ignore_list.txt").read_text().split("\n")
        ignore_names = set(line.strip() for line in lines)

        valid_names = []
        for name in names:
            if name not in ignore_names:
                valid_names.append(name)

        motion_paths = []
        wav_paths = []
        for name in valid_names:
            motion_paths.append(self.root / f"motions/{name}.pkl")
            wav_paths.append(self.root / f"wavs/{name}.wav")

        # sort motions and sounds
        self.motion_paths = sorted(motion_paths)
        self.wav_paths = sorted(wav_paths)

        self.transforms = transforms

    def __getitem__(self, index):
        with open(self.motion_paths[index], "rb") as f:
            dance = pickle.load(f)

        y, sr = librosa.load(self.wav_paths[index])

        data = {"dance": dance, "music": {"wav": y, "sample_rate": sr}}

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.motion_paths)


class ProcessedAISTPP(Dataset):
    """Load AIST++ dataset into a dictionary

    Structure:
    root
      |- train
      |    |- motions
      |    |- wavs
    """

    def __init__(self, root: str, split: str, include_contacts: bool = True):
        super().__init__()

        self.root = Path(root)
        self.split = split

        # load raw data

        # open data path
        split_data_path = Path(root) / split
        motion_path = split_data_path / "motions"
        wav_path = split_data_path / "wavs"

        # sort motions and sounds
        motions = sorted(motion_path.glob("*.pkl"))
        wavs = sorted(wav_path.glob("*.wav"))

        self.pairs = tuple(zip(motions, wavs))

    def load_splits(self, split):
        def file_to_list(f: Path):
            lines = f.read_text().split("\n")
            out = [x.strip() for x in lines]
            out = [x for x in out if len(x)]
            return out

        filter_list = set(file_to_list(self.root / "ignore_list.txt"))
        all_files = file_to_list(self.root / f"splits/crossmodal_{split}.txt")

        files = set(self.filter(all_files, filter_list))

        return files

    def filter(self, files, filter_list):
        for file in files:
            if file not in filter_list:
                yield file

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        motion_path, wav_path = self.pairs[idx]

        with open(motion_path, "rb") as f:
            dance = pickle.load(f)

        y, sr = librosa.load(wav_path)

        return {"dance": dance, "music": {"wav": y, "sample_rate": sr}}

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
