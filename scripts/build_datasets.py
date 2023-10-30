from jsonargparse import CLI

from paired.data import aistpp


def download(root: str, verbose: bool = True):
    aistpp.download(root, verbose=verbose)


if __name__ == "__main__":
    CLI([download], as_positional=False)
