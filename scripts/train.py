from pathlib import Path

import wandb
import joblib
import torch
from einops import rearrange
from jsonargparse import CLI
from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm.auto import tqdm

from paired.data.aistpp import load_aistpp
from paired.ddpm import DDPM
from paired.model import UNet
from paired.training import HyperParams


def infinite(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def main(root: str, h: HyperParams, log_interval: int = 50, val_interval: int = 1000):
    fabric = Fabric(accelerator="gpu", devices=1, precision="16-mixed")

    model = UNet(x_channels=147 * 2, y_channels=1, channels_per_depth=(64, 128, 128, 128))
    dm = DDPM(model, h.timesteps, h.start, h.end)

    optimizer = torch.optim.Adam(model.parameters())

    dm, optimizer = fabric.setup(dm, optimizer)

    cache_path = Path(root) / "dataset.joblib"

    if cache_path.exists():
        dataset = joblib.load(cache_path)
    else:
        dataset = load_aistpp(root, return_all=False, stride=0.5, length=5)
        joblib.dump(dataset, cache_path)

    data, metadata = dataset

    train_loader = DataLoader(
        data["train"],
        batch_size=h.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        data["val"],
        batch_size=h.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    resizer = Resize(size=(64, 64), antialias=True)
    resizer = fabric.setup_module(resizer)

    wandb.init()

    def training_step(batch):
        # convert to 1d wav
        dance = batch["dance"]
        batch_size = dance.shape[0]
        motion_wav = rearrange(dance, "b t c -> (b c) t")

        spec = torch.stft(motion_wav, n_fft=30, hop_length=6, return_complex=True)

        # convert to image
        spec = rearrange(spec, "(b c) f t -> b c f t", b=batch_size)

        mag = spec.abs()
        phase = spec.angle()

        # batch x 147 x 16 x 51
        x = torch.cat((mag, phase), dim=1)
        x = resizer(x)
        # batch x 1 x 128 x 431
        y = rearrange(batch["mel"].clone(), "b f t -> b 1 f t")
        y = resizer(y)

        loss, parts = dm.training_step(x, y)

        fabric.backward(loss.mean())

        optimizer.step()

        return model, (loss, parts)

    def val_step(batch):
        with torch.no_grad():
            dance = batch["dance"]
            batch_size = dance.shape[0]
            motion_wav = rearrange(dance, "b t c -> (b c) t")

            spec = torch.stft(motion_wav, n_fft=30, hop_length=6, return_complex=True)

            # convert to image
            spec = rearrange(spec, "(b c) f t -> b c f t", b=batch_size)

            mag = spec.abs()
            phase = spec.angle()

            # batch x 147 x 16 x 51
            x = torch.cat((mag, phase), dim=1)
            x = resizer(x)
            # batch x 1 x 128 x 431
            y = rearrange(batch["mel"].clone(), "b f t -> b 1 f t")
            y = resizer(y)

            loss, parts = dm.training_step(x, y)
            return (loss, parts)

    def logging(logs):
        if logs is None:
            return

        loss, parts = logs
        wandb.log(parts)

    step = 0
    logs = None

    for batch in tqdm(infinite(train_loader), total=h.training_steps, dynamic_ncols=True, position=0):
        step += 1

        if step > h.training_steps:
            break

        if step % log_interval == 0:
            logging(logs)

        if step % val_interval == 0:
            model = model.eval()

            for batch in tqdm(val_loader, position=1, leave=False):
                logs = val_step(batch)
                state = {"dm": dm, "optimizer": optimizer, "step": step, "h":h}
                fabric.save(f"model_{step}.ckpt", state)

            model = model.train()

        model, logs = training_step(batch)


if __name__ == "__main__":
    CLI(main, as_positional=False)
