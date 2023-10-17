import torch
from smplx import SMPL
from tqdm.auto import tqdm

from .features.kinetic import extract_kinetic_features
from .features.manual import extract_manual_features
from .pytorch3d.transforms import (
    RotateAxisAngle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
from .vis import SMPLSkeleton


def extract_aistpp_features(dataset, model_path):
    smpl = SMPL(model_path=model_path, gender='MALE', batch_size=1)

    for data in tqdm(dataset):
        dance = data["dance"]
        smpl_poses = dance["smpl_poses"]
        smpl_scaling = dance["smpl_scaling"]
        smpl_trans = dance["smpl_trans"] / smpl_scaling

        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        trans = torch.Tensor(smpl_trans).unsqueeze(0)
        pose = torch.Tensor(smpl_poses).unsqueeze(0)
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

        manual_features = extract_manual_features(positions)
        kinetic_features = extract_kinetic_features(positions)
