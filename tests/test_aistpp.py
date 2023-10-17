from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from paired.aistpp import AISTPP


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dataset(data_root, split):
    root = Path(data_root)

    path = root / "aistpp"

    dataset = list(AISTPP(path, split=split))

    data = dataset[0]
    data["dance"]

    music = data["music"]
    music["wav"]
    music["sample_rate"]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_loader(data_root, split):
    root = Path(data_root) / "aistpp"
    dataset = AISTPP(root, split)

    cached = []
    for data in dataset:
        dance = data["dance"]
        music = data["music"]
        sr = music["sample_rate"]
        wav = music["wav"][: sr * 1]

        trans = dance["smpl_trans"][: 60 * 1]
        poses = dance["smpl_poses"][: 60 * 1]

        cached.append(
            {"dance": {"trans": trans, "pose": poses}, "music": {"wav": wav, "sr": sr}}
        )

    loader = DataLoader(cached, batch_size=128)

    for batch in loader:
        trans = batch["dance"]["trans"]
        batch["dance"]["pose"]
        wav = batch["music"]["wav"]
