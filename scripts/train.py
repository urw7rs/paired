from pathlib import Path

import joblib
import numpy as np
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
from paired.data.quaternion import ax_from_6v
from paired.data.skeleton import SMPLSkeleton
from paired.ddpm import DDPM
from paired.features.kinetic import extract_kinetic_features
from paired.features.manual import extract_manual_features
from paired.features.metrics import calc_fid, calculate_avg_distance, normalize
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
    precision: str = "16-mixed",
    strategy: str = "auto",
    ckpt_dir: str = "checkpoints",
    log_interval: int = 100,
    val_interval: int = 10_000,
):
    fabric = Fabric(
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
        pos_dim=256,
        channels_per_depth=h.channels,
        attention_depths=(2, 3),
    )
    dm = DDPM(model, h.timesteps, h.start, h.end)

    optimizer = torch.optim.Adam(model.parameters(), lr=h.lr)
    scheduler = WarmupLR(optimizer, warmup=5000)

    dm, optimizer = fabric.setup(dm, optimizer)

    dataset, metadata = load_aistpp(root, splits=["train", "val"])

    train_loader = DataLoader(
        dataset["train"],
        batch_size=h.batch_size,
        num_workers=16,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        dataset["val"],
        batch_size=h.batch_size,
        num_workers=16,
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

    def training_step(batch):
        x, y, _, _ = process_batch(batch)

        loss, parts = dm.training_step(x, y)

        fabric.backward(loss.mean())

        optimizer.step()
        scheduler.step()

        return model, (loss, parts)

    def val_step(batch):
        with torch.no_grad():
            x, y, x_shape, y_shape = process_batch(batch)
            loss, parts = dm.training_step(x, y)

            x_hat = dm.generate(x.shape, y)
            mag, phase = torch.chunk(x_hat, chunks=2, dim=1)
            mag = mag / 2 + 0.5
            mag *= 30
            phase *= torch.pi

            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)

            spec = torch.complex(real, imag)
            spec = Resize(x_shape[-2:], antialias=True).to(fabric.device)(spec)
            spec = rearrange(spec, "b c f t -> (b c) f t")

            motion_wav = torch.istft(
                spec, n_fft=30, hop_length=30 // 4, length=batch["poses"].shape[1]
            )
            poses = rearrange(motion_wav, "(b c) t -> b t c", b=x.shape[0])

            trans = poses[:, :, :3]
            pose = rearrange(poses[:, :, 3:], "n t (j c) -> n t j c", j=24)

            pose = ax_from_6v(pose)
            positions = skeleton.forward(pose, trans)

            return loss, parts, positions

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

    skeleton = SMPLSkeleton(fabric.device)

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

            pred_features_k = []
            pred_features_m = []
            gt_features_k = []
            gt_features_m = []

            x_loss = 0
            y_loss = 0
            total_loss = 0
            for batch in tqdm(val_loader, position=1, leave=False):
                loss, parts, positions = val_step(batch)

                x_loss += parts["x_loss"]
                y_loss += parts["y_loss"]
                total_loss += loss

                for position in positions:
                    kinetic_features = extract_kinetic_features(position.cpu().numpy())
                    geometric_features = extract_manual_features(position.cpu().numpy())

                    pred_features_k.append(kinetic_features)
                    pred_features_m.append(geometric_features)

                gt_features_k.append(batch["features"]["kinetic"].cpu().numpy())
                gt_features_m.append(batch["features"]["geometric"].cpu().numpy())

            pred_features_k = np.stack(pred_features_k)  # Nx72 p40
            pred_features_m = np.stack(pred_features_m)  # Nx32

            gt_features_k = np.concatenate(gt_features_k, axis=0)  # N` x 72 N` >> N
            gt_features_m = np.concatenate(gt_features_m, axis=0)  # N` x 32 N' >> N

            pred_features_m = fabric.all_gather(pred_features_m).cpu().numpy()
            pred_features_k = fabric.all_gather(pred_features_k).cpu().numpy()

            pred_features_m = rearrange(pred_features_m, "d n c -> (d n) c")
            pred_features_k = rearrange(pred_features_k, "d n c -> (d n) c")

            gt_features_m = fabric.all_gather(gt_features_m).cpu().numpy()
            gt_features_k = fabric.all_gather(gt_features_k).cpu().numpy()

            gt_features_m = rearrange(gt_features_m, "d n c -> (d n) c")
            gt_features_k = rearrange(gt_features_k, "d n c -> (d n) c")

            gt_features_k, pred_features_k = normalize(gt_features_k, pred_features_k)
            gt_features_m, pred_features_m = normalize(gt_features_m, pred_features_m)

            fid_k = calc_fid(pred_features_k, gt_features_k)
            fid_g = calc_fid(pred_features_m, gt_features_m)

            div_k_gt = calculate_avg_distance(gt_features_k)
            div_g_gt = calculate_avg_distance(gt_features_m)
            div_k = calculate_avg_distance(pred_features_k)
            div_g = calculate_avg_distance(pred_features_m)

            metrics = {
                "fid_k": fid_k,
                "fid_g": fid_g,
                "div_k_gt": div_k_gt,
                "div_g_gt": div_g_gt,
                "div_k": div_k,
                "div_g": div_g,
            }

            valid_metrics = {}
            for k, v in metrics.items():
                if not np.iscomplexobj(v):
                    valid_metrics[k] = v

            n = len(val_loader)
            if fabric.local_rank == 0:
                wandb.log(
                    {
                        "val_loss": total_loss,
                        "val_x_loss": x_loss / n,
                        "val_y_loss": y_loss / n,
                        **valid_metrics,
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
