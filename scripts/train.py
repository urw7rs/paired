import math
from pathlib import Path

import joblib
import torch
import wandb
from einops import rearrange
from jsonargparse import CLI
from lightning.fabric import Fabric
from torch.optim.lr_scheduler import _LRScheduler
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


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup=0.0, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        steps = self.optimizer._step_count + 1

        if steps < self.warmup_steps:
            return [
                group["initial_lr"] * (steps / self.warmup_steps)
                for group in self.optimizer.param_groups
            ]
        else:
            return [group["initial_lr"] for group in self.optimizer.param_groups]


def main(
    root: str,
    h: HyperParams,
    gpus: int = 1,
    ckpt_dir: str = "checkpoints",
    log_interval: int = 50,
    val_interval: int = 1000,
):
    fabric = Fabric(
        accelerator="gpu", devices=gpus, precision="16-mixed", strategy="ddp"
    )
    fabric.seed_everything(h.seed)
    fabric.launch()

    model = UNet(
        x_channels=147 * 2,
        y_channels=1,
        channels_per_depth=(256, 512, 512, 512),
        attention_depths=(2, 3),
    )
    dm = DDPM(model, h.timesteps, h.start, h.end)

    optimizer = torch.optim.Adam(model.parameters(), lr=h.lr)
    scheduler = WarmupLR(optimizer, warmup=5000)

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
        drop_last=True,
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

    resizer = Resize(size=(32, 32), antialias=True).to(fabric.device)

    if fabric.local_rank == 0:
        wandb.init()

    def norm(x):
        return (x - 0.5) * 2

    def process_batch(batch):
        # convert to 1d wav
        dance = batch["dance"]
        batch_size = dance.shape[0]
        motion_wav = rearrange(dance, "b t c -> (b c) t")

        spec = torch.stft(motion_wav, n_fft=30, hop_length=30 // 4, return_complex=True)

        # convert to image
        spec = rearrange(spec, "(b c) f t -> b c f t", b=batch_size)

        mag = torch.clamp(spec.abs(), 1e-5).log()

        mag = (mag - math.log(1e-5)) / (math.log(30) - math.log(1e-5))
        mag = norm(mag)

        phase = spec.angle()
        phase /= torch.pi

        # batch x 147 x 16 x 51
        x = torch.cat((mag, phase), dim=1)
        x = resizer(x)
        # batch x 1 x 128 x 431
        y = rearrange(batch["mel"].clone(), "b f t -> b 1 f t")
        y = resizer(y)

        return x, y

    def training_step(batch):
        x, y = process_batch(batch)

        loss, parts = dm.training_step(x, y)

        fabric.backward(loss.mean())

        optimizer.step()
        scheduler.step()

        return model, (loss, parts)

    def val_step(batch):
        with torch.no_grad():
            x, y = process_batch(batch)
            loss, parts = dm.training_step(x, y)
            return (loss, parts)

    def logging(logs):
        if logs is None:
            return

        loss, parts = logs
        if fabric.local_rank == 0:
            wandb.log(
                {
                    "train_loss": loss,
                    "train_x_loss": parts["x_loss"],
                    "train_y_loss": parts["y_loss"],
                },
                step=step,
            )

    step = 0
    logs = None

    ckpt_dir = Path(ckpt_dir)
    if fabric.local_rank == 0:
        ckpt_dir.mkdir()

    fabric.barrier()

    for batch in tqdm(
        infinite(train_loader), total=h.training_steps, dynamic_ncols=True, position=0
    ):
        step += 1

        if step > h.training_steps:
            break

        if step % log_interval == 0:
            if fabric.local_rank == 0:
                logging(logs)

        if step % val_interval == 0:
            model = model.eval()

            x_loss = 0
            y_loss = 0
            total_loss = 0
            for batch in tqdm(val_loader, position=1, leave=False):
                (loss, parts) = val_step(batch)

                x_loss += parts["x_loss"]
                y_loss += parts["y_loss"]
                total_loss += loss

            n = len(val_loader)
            if fabric.local_rank == 0:
                wandb.log(
                    {
                        "val_loss": total_loss,
                        "val_x_loss": x_loss / n,
                        "val_y_loss": y_loss / n,
                    },
                    step=step,
                )

            joblib.dump(h, ckpt_dir / "hparams.joblib")
            state = {
                "dm": dm,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "step": step,
            }
            fabric.save(ckpt_dir / f"model_{step}.ckpt", state)

            model = model.train()

        model, logs = training_step(batch)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    CLI(main, as_positional=False)
