import pickle
from pathlib import Path

import librosa
from torch.utils.data import Dataset


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
