from pathlib import Path

import pytest

from paired.aistpp import AISTPP
from paired.aistpp.dataloader import get_test_loader, get_train_val_loader


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


def test_train_val_loader(data_root):
    path = Path(data_root) / "aistpp"
    train_loader, val_loader = get_train_val_loader(path, batch_size=4, num_workers=0)

    pose, filename, wav = next(iter(train_loader))
    pose, filename, wav = next(iter(val_loader))


def test_get_test_loader(data_root):
    path = Path(data_root) / "aistpp"
    test_loader = get_test_loader(path, batch_size=4, num_workers=0)

    pose, filename, wav = next(iter(test_loader))
