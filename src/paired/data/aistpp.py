import pickle
import zipfile
from pathlib import Path

import gdown
import joblib
import librosa
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from . import transforms
from .data_list import DataList


class AISTPP(Dataset):
    """Load AIST++ dataset into a dictionary

    Structure:
    root
      |- train
      |    |- motions
      |    |- wavs
    """

    def __init__(self, root: str, split: str, transforms=None, download: bool = False):
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

    @staticmethod
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


def build_aistpp(root, stride: float = 0.5, length: int = 5, fps: int = 30):
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


memory = joblib.Memory("~/.paired")

memory.cache


def get_min_max(root, key: str = "poses"):
    train_set = DataList(Path(root) / "train")

    max_vals = []
    min_vals = []
    for data in train_set:
        max_vals.append(data[key].max(dim=0).values)
        min_vals.append(data[key].min(dim=0).values)

    train_max = torch.stack(max_vals, dim=0).max(dim=0).values
    train_min = torch.stack(min_vals, dim=0).min(dim=0).values

    return train_min, train_max


def load_aistpp(root, splits):
    train_min, train_max = get_min_max(root)

    dataset = {}
    for split in splits:
        dataset[split] = DataList(
            Path(root) / split,
            transforms=transforms.MinMaxNormalize(train_max, train_min),
        )

    metadata = {"max": train_max, "min": train_min}

    return dataset, metadata
