import functools
from pathlib import Path

import numpy as np
import webdataset as wds
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import Compose

from . import aistpp


def identity(x):
    return x


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


def slice_clips(src, stride: float = 0.5, length: float = 7.75, fps: int = 60):
    for sample in src:
        key, dance, music = sample

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

        for i, (dance, wav) in enumerate(zip(dance_slices, wav_slices)):
            yield (f"{key}_{i}", dance, wav)


def slice_wds(
    urls,
    output: str,
    stride: float = 0.5,
    length: float = 7.75,
    fps: int = 60,
    maxcount: int = 50,
):
    dataset = (
        wds.WebDataset(urls)
        .decode(
            wds.handle_extension("dance.pkl", aistpp.pkl_decoder),
            wds.handle_extension("music.wav", aistpp.wav_decoder),
        )
        .to_tuple("__key__", "dance.pkl", "music.wav")
        .map_tuple(identity, Compose([aistpp.preproc, to_6d, flatten]), identity)
        .compose(functools.partial(slice_clips, stride=stride, length=length, fps=fps))
    )

    Path(output).parent.mkdir(exist_ok=True, parents=True)

    with wds.ShardWriter(output, maxcount=maxcount) as sink:
        for key, dance, music in dataset:
            sample = {
                "__key__": key,
                "dance.pyd": dance,
                "music.pyd": music,
            }
            sink.write(sample)


def numpy_collate(batch):
    return batch


def make_train(
    pattern: str,
    steps: int,
    batch_size: int = 64,
    num_workers: int = 8,
    stride: float = 0.5,
    length: float = 7.75,
    fps: int = 60,
):
    dataset = (
        wds.WebDataset(pattern, shardshuffle=True)
        .shuffle(10_000)
        .decode()
        .to_tuple("dance.pyd", "music.pyd")
        .repeat()
    )
    dataset = dataset.batched(batchsize=batch_size, partial=True)

    loader = (
        wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=numpy_collate,
        )
        .with_epoch(nbatches=steps)
        .with_length(steps)
    )
    return loader


def make_eval(
    pattern: str,
    steps: int,
    batch_size: int = 64,
    num_workers: int = 8,
    stride: float = 0.5,
    length: float = 7.75,
    fps: int = 60,
):
    dataset = (
        wds.WebDataset(pattern)
        .decode(
            wds.handle_extension("dance.pkl", aistpp.pkl_decoder),
            wds.handle_extension("music.wav", aistpp.wav_decoder),
        )
        .to_tuple("dance.pkl", "music.wav")
        .map_tuple(Compose([aistpp.preproc, to_6d, flatten]), identity)
    )

    return dataset
