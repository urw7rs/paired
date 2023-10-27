import pickle
import zipfile
from pathlib import Path

import gdown
import joblib
import librosa
import numpy as np
import torch
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from . import transforms
from .data_list import DataList


def download(root: str, verbose: bool = True):
    url = "https://drive.google.com/u/0/uc?id=16qYnN3qpmHMk2mOvOsOYNLy75xUmbyif"
    md5 = "569a60311ecebb5001c8a7321ba787f3"

    zip_path = Path(root) / "aistpp.zip"
    try:
        gdown.cached_download(
            url=url, path=str(zip_path), md5=md5, quiet=not verbose, resume=False
        )
    except FileNotFoundError:
        exit()

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(root)


def load_dance(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        dance = pickle.load(f)

    return dance


def load_music(path: str, **kwargs) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, **kwargs)
    return y, sr


def load_names(root: Path, split: str):
    if split == "all":
        names = (root / "splits/all.txt").read_text().split("\n")
    else:
        names = (root / f"splits/crossmodal_{split}.txt").read_text().split("\n")

    return names


class Dataset(torch.utils.data.Dataset):
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

        # read split files and load names
        if split == "all":
            names = (self.root / "splits/all.txt").read_text().split("\n")
        else:
            names = (
                (self.root / f"splits/crossmodal_{split}.txt").read_text().split("\n")
            )

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
        path_pairs = zip(motion_paths, wav_paths)
        self.path_pairs = sorted(path_pairs)

        self.transforms = transforms

    def __getitem__(self, index):
        motion_path, wav_path = self.path_pairs[index]

        assert motion_path.with_suffix("").name == wav_path.with_suffix("").name

        dance = load_dance(motion_path)
        y, sr = load_music(wav_path)

        data = {"dance": dance, "music": y, "sample_rate": sr}

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.motion_paths)


def build_aistpp(root, stride: float = 0.5, length: float = 7.75, fps: int = 60):
    def process_split(split):
        dataset = AISTPP(
            root,
            split=split,
            transforms=Compose(
                [
                    transforms.PreProcessing(fps=fps),
                    transforms.SliceClips(stride=stride, length=length, fps=fps),
                ]
            ),
        )

        post_fn = transforms.PostProcessing()

        def fn(i):
            slices = dataset[i]

            new_slices = []
            for data in slices:
                data = post_fn(data)
                new_slices.append(data)

            return new_slices

        def parallel(generator, return_as="generator", total: int = None):
            if total is None:
                total = len(dataset)

            output = joblib.Parallel(n_jobs=-1, return_as=return_as)(generator)

            return tqdm(output, dynamic_ncols=True, total=total)

        data_list = DataList(Path(root) / split)

        for slices in parallel(joblib.delayed(fn)(i) for i in range(len(dataset))):
            for data in slices:
                data_list.add(data)

        return data_list

    train_set = process_split("train")
    val_set = process_split("val")
    test_set = process_split("test")

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }


def get_min_max(dataset, key: str = "poses"):
    max_vals = []
    min_vals = []
    for data in dataset:
        max_vals.append(data[key].max(dim=0).values)
        min_vals.append(data[key].min(dim=0).values)

    train_max = torch.stack(max_vals, dim=0).max(dim=0).values
    train_min = torch.stack(min_vals, dim=0).min(dim=0).values

    return train_min, train_max


memory = joblib.Memory("~/.paired", verbose=0)


def load_aistpp(root, splits, subset_size: int = -1):
    train_set = DataList(Path(root) / "train")
    train_min, train_max = memory.cache(get_min_max)(train_set, key="poses")

    dataset = {}
    for split in splits:
        full_data = DataList(
            Path(root) / split,
            transforms=transforms.MinMaxNormalize(train_min, train_max),
        )

        if subset_size != -1:
            dataset[split] = Subset(full_data, indices=list(range(0, subset_size)))
        else:
            dataset[split] = full_data

    metadata = {"max": train_max, "min": train_min}

    return dataset, metadata
