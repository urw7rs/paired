from pathlib import Path

import webdataset as wds
from jsonargparse import CLI

from paired.data import aistpp, loaders


def raw(root: Path, split: str):
    names = aistpp.load_names(root, split)

    output = str(root / f"{split}/shard-%06d.tar")
    Path(output).parent.mkdir(exist_ok=True)

    with wds.ShardWriter(output, encoder=False, maxcount=50) as sink:
        for basename in sorted(names):
            motion_bytes = (root / f"motions/{basename}.pkl").read_bytes()
            wav_bytes = (root / f"wavs/{basename}.wav").read_bytes()

            sample = {
                "__key__": basename,
                "dance.pkl": motion_bytes,
                "music.wav": wav_bytes,
            }

            sink.write(sample)


def slice(
    pattern: str,
    output: str,
    stride: float = 0.5,
    length: float = 7.75,
    fps: int = 60,
    maxcount: int = 50,
):
    loaders.slice_wds(
        pattern, output, stride=stride, length=length, fps=fps, maxcount=maxcount
    )


def download(root: Path):
    zip_path = aistpp.download(root)
    root = aistpp.extract(zip_path)
    return root


if __name__ == "__main__":
    CLI([download, raw, slice], as_positional=False)
