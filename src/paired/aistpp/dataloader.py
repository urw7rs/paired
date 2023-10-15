from torch.utils.data import DataLoader

from .dataset import AISTPP


def get_train_val_loader(
    path: str,
    batch_size: int,
    num_workers: int = 0,
    stride: float = 0.5,
    length: int = 5,
):
    train_set = AISTPP(
        path, path / "cache", split="train", stride=stride, length=length
    )
    val_set = AISTPP(
        path,
        path / "cache",
        split="val",
        normalizer=train_set.normalizer,
        stride=stride,
        length=length,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader


def get_test_loader(
    path: str,
    batch_size: int,
    num_workers: int = 0,
    stride: float = 0.5,
    length: int = 5,
):
    train_set = AISTPP(path, path / "cache", split="train")
    test_set = AISTPP(
        path,
        path / "cache",
        split="test",
        normalizer=train_set.normalizer,
        stride=stride,
        length=length,
    )

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader
