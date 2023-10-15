import shutil
from pathlib import Path

import pytest

from paired.aistpp import AISTPP


@pytest.fixture
def aistpp_path(data_root):
    root = Path(data_root).expanduser() / "aistpp"
    yield root
    shutil.rmtree(root)


def test_download(aistpp_path):
    AISTPP.download(path=str(aistpp_path))

    def assert_exists(path):
        assert (aistpp_path / "aistpp" / path).exists()

    assert_exists("motions")
    assert_exists("wavs")
    assert_exists("splits/crossmodal_train.txt")
    assert_exists("splits/crossmodal_val.txt")
    assert_exists("splits/crossmodal_test.txt")
    assert_exists("ignore_list.txt")

def test_aistpp(aistpp_path):
    root = Path(aistpp_path)

    AISTPP.download(path=root)

    train_set = AISTPP(root / "aistpp", root / "cache", split="train")
    val_set = AISTPP(
        root / "aistpp", root / "cache", split="val", normalizer=train_set.normalizer
    )
    test_set = AISTPP(
        root / "aistpp", root / "cache", split="test", normalizer=train_set.normalizer
    )
