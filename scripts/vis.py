import os
from pathlib import Path

import joblib
import lightning as L
import torch
from einops import rearrange
from jsonargparse import CLI
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from paired.data.aistpp import load_aistpp
from paired.data.quaternion import ax_from_6v
from paired.data.skeleton import SMPLSkeleton
from paired.data.vis import plot_skeleton
from paired.ddpm import DDPM
from paired.model import UNet


def vis(
    root: str,
    ckpt_dir: str,
    gif_dir: str,
    gpus: int = 1,
    precision: str = "16-mixed",
    strategy: str = "auto",
):
    ckpt_dir = Path(ckpt_dir)
    h = joblib.load(ckpt_dir / "hparams.joblib")

    fabric = L.Fabric(
        accelerator="gpu",
        devices=gpus,
        precision=precision,
        strategy=strategy,
    )

    fabric.seed_everything(h.seed)
    fabric.launch()

    model = UNet(
        x_channels=147 * 2,
        y_channels=1,
        pos_dim=h.pos_dim,
        emb_dim=h.emb_dim,
        channels_per_depth=h.channels,
        attention_depths=h.attention_depths,
    )

    ckpt_path = ckpt_dir / "best.ckpt"
    state = {"model": model}
    fabric.load(ckpt_path, state)

    dm = DDPM(model, h.timesteps, h.start, h.end)
    dm = fabric.setup_module(dm)

    dataset, metadata = load_aistpp(root, splits=["train", "val"])

    train_loader = DataLoader(
        dataset["train"],
        batch_size=h.batch_size,
        num_workers=min(16, os.cpu_count()),
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        dataset["val"],
        batch_size=h.batch_size,
        num_workers=min(16, os.cpu_count()),
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    resizer = Resize(size=(32, 32), antialias=True).to(fabric.device)

    def norm(x):
        return (x - 0.5) * 2

    def process_batch(batch):
        # convert to 1d wav
        dance = batch["poses"]
        batch_size = dance.shape[0]
        motion_wav = rearrange(dance, "b t c -> (b c) t")

        spec = torch.stft(motion_wav, n_fft=30, hop_length=30 // 4, return_complex=True)

        # convert to image
        spec = rearrange(spec, "(b c) f t -> b c f t", b=batch_size)

        mag = spec.abs()
        phase = spec.angle()

        mag /= 30
        mag = norm(mag)

        phase /= torch.pi

        # batch x 147 x 16 x 51
        x = torch.cat((mag, phase), dim=1)
        x_shape = x.shape
        x = resizer(x)
        # batch x 1 x 128 x 431
        y = rearrange(batch["mel"].clone(), "b f t -> b 1 f t")
        y_shape = y.shape
        y = resizer(y)

        return x, y, x_shape, y_shape

    skeleton = SMPLSkeleton(fabric.device)

    gif_dir = Path(gif_dir)
    gif_dir.mkdir()

    for batch in train_loader:
        with torch.no_grad():
            x, y, x_shape, y_shape = process_batch(batch)

            x_hat = dm.generate(x.shape, y)
            # x_hat = x

            resizer = Resize(x_shape[-2:], antialias=True).to(fabric.device)
            x_hat = resizer(x_hat)

            mag, phase = torch.chunk(x_hat, chunks=2, dim=1)
            mag = mag / 2 + 0.5
            mag *= 30
            phase *= torch.pi

            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)

            spec = torch.complex(real, imag)

            spec = rearrange(spec, "b c f t -> (b c) f t")

            motion_wav = torch.istft(
                spec, n_fft=30, hop_length=30 // 4, length=batch["poses"].shape[1]
            )
            poses = rearrange(motion_wav, "(b c) t -> b t c", b=x.shape[0])

            max_val = fabric.to_device(metadata["max"])
            min_val = fabric.to_device(metadata["min"])

            poses = poses * (min_val - max_val) + max_val

            trans = poses[:, :, :3]
            pose = rearrange(poses[:, :, 3:], "n t (j c) -> n t j c", j=24)

            pose = ax_from_6v(pose)
            positions = skeleton.forward(pose, trans).cpu().numpy()
            positions -= positions[:, :1, :1]

            joblib.Parallel(n_jobs=-1)(
                joblib.delayed(plot_skeleton)(gif_dir / f"{i}.gif", position, fps=30)
                for i, position in enumerate(positions)
            )

        break


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    CLI([vis], as_positional=False)
