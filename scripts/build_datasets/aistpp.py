import glob
import os
import pickle
import zipfile
from pathlib import Path

import gdown
import librosa as lr
import soundfile as sf
from jsonargparse import CLI
from tqdm.auto import tqdm


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


def build(root: str, stride: float = 0.5, length: int = 5):
    # slice motions/music into sliding windows to create dataset
    root = Path(root)
    slice_aistpp(
        motion_dir=root / "motions",
        wav_dir=root / "wavs",
        stride=stride,
        length=length,
    )


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    with open(motion_file, "rb") as f:
        motion = pickle.load(f)

    trans, pose = motion["smpl_trans"], motion["smpl_poses"]
    scale = motion["smpl_scaling"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    trans /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(trans) - window and slice_count < num_slices:
        sliced_trans, sliced_pose = (
            trans[start_idx : start_idx + window],
            pose[start_idx : start_idx + window],
        )
        out = {"smpl_trans": sliced_trans, "smpl_poses": sliced_pose}

        with open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb") as f:
            pickle.dump(out, f)

        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
    motion_dir = str(motion_dir)
    wav_dir = str(wav_dir)

    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))

    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"

    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)

    assert len(wavs) == len(motions)

    for wav, motion in tqdm(list(zip(wavs, motions))):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )


if __name__ == "__main__":
    CLI([download, build], as_positional=False)
