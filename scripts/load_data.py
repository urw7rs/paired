import io
import pickle
from pathlib import Path

import librosa
import numpy as np
import webdataset as wds
from einops import rearrange
from jsonargparse import CLI
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import Compose
from tqdm.auto import tqdm


def preproc(x):
    trans = x["smpl_trans"] / x["smpl_scaling"]
    pose = x["smpl_poses"]
    pose = pose.reshape(-1, 24, 3)

    # rotate 90 degrees
    root_pose = pose[:, 0]
    r = R.from_rotvec(root_pose)
    rot_90 = R.from_euler("x", 90, degrees=True)
    pose[:, 0] = (rot_90 * r).as_rotvec()

    # rotate positions too
    trans = rot_90.apply(trans)

    return {"trans": trans, "pose": pose}


def to_6d(x):
    pose = x["pose"]
    trans = x["trans"]
    # convert tp 6d
    batch_dim = pose.shape[:-1]
    matrix = R.from_rotvec(pose.reshape(-1, 3)).as_matrix()
    matrix = matrix.reshape(batch_dim + (3, 3))

    pose = matrix[..., :2, :].reshape(batch_dim + (6,))

    return {"trans": trans, "pose": pose}


def flatten(x):
    trans = x["trans"]
    pose = x["pose"]

    pose = rearrange(pose, "t j c-> t (j c)")
    return np.concatenate([trans, pose], axis=-1)


def identity(x):
    return x


def pkl_decoder(pkl_bytes):
    f = io.BytesIO(pkl_bytes)
    dance = pickle.load(f)
    return dance


def wav_decoder(wav_bytes):
    f = io.BytesIO(wav_bytes)
    y, sr = librosa.load(f)
    return {"wav": y, "sample_rate": sr}


def slice_clips(src, stride: float = 0.5, length: float = 7.75, fps: int = 60):
    for sample in src:
        dance, music = sample

        wav = music["wav"]
        sr = music["sample_rate"]

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

        start_idx = 0
        window = int(length * fps)
        stride_step = int(stride * fps)
        slice_count = 0
        # slice until done or until matching audio slices
        while start_idx <= len(dance) - window and slice_count < num_slices:
            dance_slice = dance[start_idx : start_idx + window]
            dance_slices.append(dance_slice)

            start_idx += stride_step
            slice_count += 1

        for dance, wav in zip(dance_slices, wav_slices):
            yield (dance, wav)


def test_load(path: str):
    path = str(Path(path) / "datasets")

    dataset = (
        wds.WebDataset(
            path + "/aistpp/aistpp-train-{000000..000019}.tar", shardshuffle=True
        )
        .shuffle(1000)
        .decode(
            wds.handle_extension("dance.pkl", pkl_decoder),
            wds.handle_extension("music.wav", wav_decoder),
        )
        .to_tuple("dance.pkl", "music.wav")
        .map_tuple(Compose([preproc, to_6d, flatten]), identity)
        .compose(slice_clips)
        .shuffle(1000)
        .repeat()
    )
    dataset = dataset.batched(batchsize=8, partial=False)

    loader = (
        wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=8)
        .with_epoch(nbatches=5000)
        .with_length(5000)
    )

    for i, sample in enumerate(tqdm(loader, dynamic_ncols=True)):
        dance, music = sample
        # print(dance.shape, music.shape)
        # print(dance["pose"].shape)
        # print(dance["trans"].shape)
        # print(music.shape)


if __name__ == "__main__":
    CLI(test_load, as_positional=False)
