import copy

import torch
from joblib import Memory
from tqdm.auto import tqdm

from .aistpp import AISTPP
from .features.kinetic import extract_kinetic_features
from .features.manual import extract_manual_features
from .preprocess import vectorize_many
from .pytorch3d.transforms import (
    RotateAxisAngle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
from .quaternion import ax_to_6v
from .vis import SMPLSkeleton


cachedir = "./cache"
memory = Memory(cachedir, verbose=0)


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


@memory.cache
def load_aistpp(root):
    def load_split(split):
        dataset = AISTPP(root, split=split)

        dataset = preprocess_aistpp(dataset)
        dataset = extract_features(dataset)

        return dataset

    train_set = load_split("train")
    val_set = load_split("val")
    test_set = load_split("test")

    max_motion = get_max_motion(train_set)
    min_motion = get_min_motion(train_set)

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


def split_dataset(dataset, stride: float = 0.5, length: int = 5, fps: int = 30):
    new_dataset = []

    for data in tqdm(dataset, dynamic_ncols=True):
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

        for pose, position, audio in zip(dance_slices, position_slices, wav_slices):
            new_dataset.append({"dance": pose, "dance_xyz": position, "music": audio})

    return new_dataset


@memory.cache
def load_split_aistpp(root, stride: float = 0.5, length: int = 5):
    def load_split(split):
        dataset = AISTPP(root, split=split)

        dataset = preprocess_aistpp(dataset)
        dataset = split_dataset(dataset, stride, length)
        dataset = extract_features(dataset)

        return dataset

    train_set = load_split("train")
    val_set = load_split("val")
    test_set = load_split("test")

    max_motion = get_max_motion(train_set)
    min_motion = get_min_motion(train_set)

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


@torch.no_grad()
def preprocess_aistpp(dataset):
    # FK skeleton
    smpl = SMPLSkeleton()

    new_dataset = []
    for data in tqdm(dataset, dynamic_ncols=True):
        new_data = copy.deepcopy(data)

        dance = new_data["dance"]

        # convert 60fps data to 30fps
        pose = dance["smpl_poses"][::2]
        trans = dance["smpl_trans"][::2]

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
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        pose[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        trans = pos_rotation.transform_points(
            trans
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        # do FK
        positions = smpl.forward(pose, trans)  # batch x sequence x 24 x 3
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        (feetv < 0.01).to(pose)  # cast to right dtype

        # to 6d
        pose = ax_to_6v(pose)

        # now, flatten everything into: batch x sequence x [...]
        global_pose_vec_input = vectorize_many([trans, pose]).float()

        new_data["dance"] = global_pose_vec_input[0]
        new_data["dance_xyz"] = positions[0]

        new_dataset.append(new_data)

    return new_dataset


@torch.no_grad()
def extract_features(dataset):
    new_dataset = []

    for data in tqdm(dataset, dynamic_ncols=True):
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

        new_dataset.append(new_data)

    return new_dataset
