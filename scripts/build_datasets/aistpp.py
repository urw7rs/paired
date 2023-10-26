from jsonargparse import CLI

from paired.data.aistpp import AISTPP, build_aistpp


def download(root: str, verbose: bool = True):
    AISTPP.download(root, verbose=verbose)


def build(root: str, stride: float = 0.5, length: float = 7.75, fps: int = 60):
    build_aistpp(root, stride=stride, length=length, fps=fps)


if __name__ == "__main__":
    CLI([download, build], as_positional=False)
