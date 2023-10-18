import zipfile
from pathlib import Path

import gdown
from jsonargparse import CLI


def download(root: str, verbose: bool = True):
    url = "https://drive.google.com/u/0/uc?id=16qYnN3qpmHMk2mOvOsOYNLy75xUmbyif"
    md5 = "569a60311ecebb5001c8a7321ba787f3"

    zip_path = Path(root) / "aistpp.zip"
    try:
        gdown.cached_download(
            url=url, path=str(zip_path), md5=md5, quiet=not verbose, resume=False
        )
    except FileNotFoundError:
        exit()

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(root)

if __name__ == "__main__":
    CLI([download], as_positional=False)
