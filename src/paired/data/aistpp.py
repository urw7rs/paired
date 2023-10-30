import io
import pickle
import zipfile
from pathlib import Path

import gdown
import librosa
from scipy.spatial.transform import Rotation as R


def download(root: str, verbose: bool = True):
    url = "https://drive.google.com/u/0/uc?id=16qYnN3qpmHMk2mOvOsOYNLy75xUmbyif"

    zip_path = Path(root) / "aistpp.zip"
    try:
        gdown.download(url=url, output=str(zip_path), quiet=not verbose, resume=True)

        return str(zip_path)
    except FileNotFoundError:
        exit()


def extract(src):
    print("Extracting...")
    output = Path(src).parent

    with zipfile.ZipFile(src) as zf:
        zf.extractall(output)

    return output


def load_names(root: Path, split: str):
    if split == "all":
        names = (root / "splits/all.txt").read_text().split("\n")
    else:
        names = (root / f"splits/crossmodal_{split}.txt").read_text().split("\n")

    # filter names in ignore_list.txt
    ignore = set(x.strip() for x in (root / "ignore_list.txt").read_text().split("\n"))

    names = [name for name in names if name not in ignore]

    return names


def preproc(x):
    trans = x["smpl_trans"] / x["smpl_scaling"]
    pose = x["smpl_poses"]
    pose = pose.reshape(-1, 24, 3)

    # rotate 90 degrees
    root_pose = pose[:, 0]
    r = R.from_rotvec(root_pose)
    rot_90 = R.from_euler("x", 90, degrees=True)
    pose[:, 0] = (rot_90 * r).as_rotvec()

    # rotate positions too
    trans = rot_90.apply(trans)

    return {"trans": trans, "pose": pose}


def pkl_decoder(pkl_bytes):
    f = io.BytesIO(pkl_bytes)
    dance = pickle.load(f)
    return dance


def wav_decoder(wav_bytes):
    f = io.BytesIO(wav_bytes)
    y, sr = librosa.load(f)
    return {"wav": y, "sample_rate": sr}
