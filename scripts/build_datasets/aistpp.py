from jsonargparse import CLI

from paired.data.aistpp import AISTPP, build_aistpp


def download(root: str, verbose: bool = True):
    AISTPP.download(root, verbose=verbose)


def build(root: str, stride: float = 0.5, length: int = 5):
    build_aistpp(root, stride=stride, length=length)


if __name__ == "__main__":
    CLI([download, build], as_positional=False)
