from jsonargparse import CLI

from paired.data.aistpp import AISTPP


def download(root: str, verbose: bool = True):
    AISTPP.download(root, verbose=verbose)


def build(root: str, verbose: bool = True):
    AISTPP(root, split=split)


if __name__ == "__main__":
    CLI([download], as_positional=False)
