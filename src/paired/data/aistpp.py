import copy
import pickle
from pathlib import Path

import einops
import joblib
import librosa
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..features.kinetic import extract_kinetic_features
from ..features.manual import extract_manual_features
from ..pytorch3d.transforms import (
    RotateAxisAngle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
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

        data = {"dance": dance, "music": y, "sample_rate": sr}

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.motion_paths)


def get_max_motion(dataset):
    dance = []
    for data in dataset:
        dance.append(data["dance"])

    return torch.cat(dance, dim=0).max(dim=0).values


def get_min_motion(dataset):
    dance = []
    for data in dataset:
        dance.append(data["dance"])

    return torch.cat(dance, dim=0).min(dim=0).values


def min_max_normalize(dataset, min_val, max_val):
    normed = []
    for data in dataset:
        new_data = copy.deepcopy(data)

        x = new_data["dance"]
        new_data["dance"] = x - min_val / (max_val - min_val)

        normed.append(new_data)

    return normed


def split_fn(data, stride: float = 0.5, length: int = 5, fps: int = 30):
    new_data = copy.deepcopy(data)

    dance = new_data["dance"]
    positions = new_data["dance_xyz"]
    wav = new_data["music"]
    sr = new_data["sample_rate"]

    # slice audio
    wav_slices = []

    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(wav) - window:
        wav_slice = wav[start_idx : start_idx + window]
        wav_slices.append(wav_slice)

        start_idx += stride_step
        idx += 1

    num_slices = idx

    # slice motion
    dance_slices = []
    position_slices = []

    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(dance) - window and slice_count < num_slices:
        dance_slice = dance[start_idx : start_idx + window]
        dance_slices.append(dance_slice)

        position_slice = positions[start_idx : start_idx + window]
        position_slices.append(position_slice)

        start_idx += stride_step
        slice_count += 1

    data_slices = []
    for pose, position, audio in zip(dance_slices, position_slices, wav_slices):
        data_slice = copy.deepcopy(new_data)
        data_slice["dance"] = pose
        data_slice["dance_xyz"] = position
        data_slice["music"] = audio

        data_slices.append(data_slice)

    return data_slices


@torch.no_grad()
def preprocess_fn(data, fps:int=30):
    new_data = copy.deepcopy(data)

    dance = new_data["dance"]

    # convert 60fps data to 30fps
    pose = dance["smpl_poses"][::60 // fps]
    trans = dance["smpl_trans"][::60 // fps]

    # normalize translations
    trans = trans / dance["smpl_scaling"]

    # to Tensor
    trans = torch.Tensor(trans).unsqueeze(0)
    pose = torch.Tensor(pose).unsqueeze(0)
    # to ax
    bs, sq, c = pose.shape
    pose = pose.reshape((bs, sq, -1, 3))

    # AISTPP dataset comes y-up - rotate to z-up
    # to standardize against the pretrain dataset
    root_q = pose[:, :, :1, :]  # sequence x 1 x 3
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.Tensor([0.7071068, 0.7071068, 0, 0])  # 90 degrees about the x axis
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat)
    pose[:, :, :1, :] = root_q

    # don't forget to rotate the root position too 😩
    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    trans = pos_rotation.transform_points(
        trans
    )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

    # do FK
    positions = SMPLSkeleton().forward(pose, trans)  # batch x sequence x 24 x 3
    feet = positions[:, :, (7, 8, 10, 11)]
    feetv = torch.zeros(feet.shape[:3])
    feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
    (feetv < 0.01).to(pose)  # cast to right dtype

    # to 6d
    pose = ax_to_6v(pose)

    # now, flatten everything into: batch x sequence x [...]
    pose = einops.rearrange(pose, "b t j c-> b t (j c)")
    global_pose_vec_input = torch.cat([trans, pose], dim=-1).float()

    new_data["dance"] = global_pose_vec_input[0]
    new_data["dance_xyz"] = positions[0]

    return new_data


@torch.no_grad()
def extract_features_fn(data):
    new_data = copy.deepcopy(data)

    positions = new_data["dance_xyz"]

    kinetic_features = extract_kinetic_features(positions.cpu().numpy())
    manual_features = extract_manual_features(positions.cpu().numpy())

    kinetic_features = torch.from_numpy(kinetic_features).float()
    manual_features = torch.from_numpy(manual_features).float()

    new_data["features"] = {
        "kinetic": kinetic_features,
        "geometric": manual_features,
    }

    S = librosa.feature.melspectrogram(
        y=new_data["music"], sr=new_data["sample_rate"], 
        n_fft=1024, hop_length=256
    )
    log_S = librosa.power_to_db(S, top_db=80) / 80

    new_data["mel"] = log_S

    return new_data


def load_aistpp(root, return_all: bool = False, stride: float = 0.5, length: int = 5):
    def load_split(split):
        dataset = AISTPP(root, split=split)

        def parallel(fn, dataset, return_as="list"):
            output = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(fn)(data)
                for data in tqdm(dataset, dynamic_ncols=True)
            )
            return tuple(output)


        dataset = parallel(preprocess_fn, dataset)
        dataset = parallel(split_fn, dataset)

        flattened = []
        for data_split in dataset:
            flattened.extend(data_split)
        dataset = flattened

        dataset = parallel(extract_features_fn, dataset)

        return dataset

    train_set = load_split("train")

    max_motion = get_max_motion(train_set)
    min_motion = get_min_motion(train_set)

    val_set = load_split("val")
    test_set = load_split("test")

    train_set = min_max_normalize(train_set, min_motion, max_motion)
    val_set = min_max_normalize(val_set, min_motion, max_motion)
    test_set = min_max_normalize(test_set, min_motion, max_motion)

    dataset = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    metadata = {
        "max": max_motion,
        "min": min_motion,
    }

    return dataset, metadata
