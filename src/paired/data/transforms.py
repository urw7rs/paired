import copy

import einops
import librosa
import torch

from ..features.kinetic import extract_kinetic_features
from ..features.manual import extract_manual_features
from ..pytorch3d.transforms import (
    RotateAxisAngle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
from .quaternion import ax_to_6v
from .skeleton import SMPLSkeleton


class PreProcessing:
    def __init__(self, fps: int = 30):
        self.fps = fps

    @torch.no_grad()
    def __call__(self, data):
        dance = data.pop("dance")

        # convert 60fps data to 30fps
        pose = dance["smpl_poses"][:: 60 // self.fps]
        trans = dance["smpl_trans"][:: 60 // self.fps]

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
        positions = SMPLSkeleton().forward(pose, trans)  # batch x sequence x 24 x 3

        data["positions"] = positions[0]

        # to 6d
        pose = ax_to_6v(pose)

        # now, flatten everything into: batch x sequence x [...]
        pose = einops.rearrange(pose, "b t j c-> b t (j c)")
        global_pose_vec_input = torch.cat([trans, pose], dim=-1).float()

        data["poses"] = global_pose_vec_input[0]

        return data


class MinMaxNormalize:
    def __init__(self, min_val, max_val, key: str = "poses"):
        self.min = min_val
        self.max = max_val

        self.key = key

    def __call__(self, data):
        x = data[self.key]
        data[self.key] = (x - self.min) / (self.max - self.min)
        return data


class PostProcessing:
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        self.n_fft = n_fft
        self.hop_length = hop_length

    @torch.no_grad()
    def __call__(self, data):
        positions = data["positions"]
        positions -= positions[:, :1]

        kinetic_features = extract_kinetic_features(positions.cpu().numpy())
        manual_features = extract_manual_features(positions.cpu().numpy())

        kinetic_features = torch.from_numpy(kinetic_features).float()
        manual_features = torch.from_numpy(manual_features).float()

        data["features"] = {
            "kinetic": kinetic_features,
            "geometric": manual_features,
        }

        S = librosa.feature.melspectrogram(
            y=data["music"],
            sr=data["sample_rate"],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        log_S = librosa.power_to_db(S, top_db=80) / 80

        data["mel"] = log_S

        return data


class SliceClips:
    def __init__(self, stride: float = 0.5, length: int = 5, fps: int = 30):
        self.stride = stride
        self.length = length
        self.fps = fps

    def __call__(self, data):
        dance = data["poses"]
        positions = data["positions"]
        wav = data["music"]
        sr = data["sample_rate"]

        # slice audio
        wav_slices = []

        start_idx = 0
        idx = 0
        window = int(self.length * sr)
        stride_step = int(self.stride * sr)
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
        window = int(self.length * self.fps)
        stride_step = int(self.stride * self.fps)
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
            data_slice = copy.deepcopy(data)
            data_slice["poses"] = pose
            data_slice["positions"] = position
            data_slice["music"] = audio

            data_slices.append(data_slice)

        return data_slices
