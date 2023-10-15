import shutil
from pathlib import Path

from paired.aistpp import AISTPP
from paired.aistpp.dataloader import get_test_loader, get_train_val_loader


def test_dataset(data_root):
    root = Path(data_root)

    path = root / "aistpp"

    train_set = AISTPP(path, path / "cache", split="train")
    val_set = AISTPP(path, path / "cache", split="val", normalizer=train_set.normalizer)
    test_set = AISTPP(
        path, path / "cache", split="test", normalizer=train_set.normalizer
    )

    shutil.rmtree(path / "cache")


def test_train_val_loader(data_root):
    path = Path(data_root) / "aistpp"
    train_loader, val_loader = get_train_val_loader(path, batch_size=4, num_workers=0)

    pose, filename, wav = next(iter(train_loader))
    pose, filename, wav = next(iter(val_loader))


def test_get_test_loader(data_root):
    path = Path(data_root) / "aistpp"
    test_loader = get_test_loader(path, batch_size=4, num_workers=0)

    pose, filename, wav = next(iter(test_loader))
