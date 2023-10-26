import math
import os
from pathlib import Path

import joblib
import lightning as L
import numpy as np
import torch
import wandb
from einops import rearrange
from jsonargparse import CLI
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
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
    ckpt_interval: int = 500,
    log_interval: int = 100,
    val_interval: int = 10_000,
):
    print(h)

    ckpt_dir = Path(ckpt_dir)

    if ckpt_dir.exists():
        h = joblib.load(ckpt_dir / "hparams.joblib")
    else:
        ckpt_dir.mkdir(parents=True)
        joblib.dump(h, ckpt_dir / "hparams.joblib")

    fabric = L.Fabric(
        accelerator="gpu",
        devices=gpus,
        precision=precision,
        strategy=strategy,
    )

    fabric.seed_everything(h.seed)
    fabric.launch()

    model = UNet(
        x_channels=h.x_channels,
        y_channels=h.y_channels,
        pos_dim=h.pos_dim,
        emb_dim=h.emb_dim,
        channels_per_depth=h.channels,
        attention_depths=h.attention_depths,
        dropout=h.dropout,
        num_blocks=h.num_blocks,
        num_groups=h.num_groups,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=h.lr)
    scheduler = WarmupLR(optimizer, warmup=5000)

    model, optimizer = fabric.setup(model, optimizer)

    dm = DDPM(model, h.timesteps, h.start, h.end)
    dm = fabric.setup_module(dm)

    # load from previous checkpoint if it exists
    if (ckpt_dir / "last.ckpt").exists():
        state = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        remainder = fabric.load(ckpt_dir / "last.ckpt", state)
        step = remainder["step"]
        best_metrics = remainder["best_metrics"]
    else:
        step = 0

    dataset, metadata = load_aistpp(root, splits=["train", "val"])

    print(h)

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

    skeleton = SMPLSkeleton(fabric.device)

    if fabric.local_rank == 0:
        wandb.init()

    def norm(x):
        return (x - 0.5) * 2

    def power_to_db(S, amin: float = 1e-10, top_db=None):
        log_spec = torch.log(torch.clamp(S, min=amin))
        log_spec -= math.log(amin)

        return log_spec

    def db_to_power(S_db, amin=1e-10):
        S_db += math.log(amin)
        return torch.exp(S_db)

    def process_batch(batch):
        # convert to 1d wav
        dance = batch["poses"]
        batch_size = dance.shape[0]
        motion_wav = rearrange(dance, "b t c -> (b c) t")

        spec = torch.stft(motion_wav, n_fft=60, hop_length=60 // 4, return_complex=True)

        # convert to image
        spec = rearrange(spec, "(b c) f t -> b c f t", b=batch_size)

        mag = spec.abs() ** 2
        phase = spec.angle()

        S_db = power_to_db(mag) / (math.log(3600) - math.log(1e-10))
        mag = norm(S_db)

        phase /= torch.pi

        # batch x 147 x 31 x 32
        x = torch.cat((mag, phase), dim=1)
        # batch x 294 x 32 x 32
        x = torch.nn.functional.pad(x, (0, 0, 0, 1), value=0)

        # batch x 1 x 128 x 431
        y = rearrange(batch["mel"], "b f t -> b 1 f t")
        y = TF.resize(
            y, size=(32, 32), antialias=True, interpolation=InterpolationMode.NEAREST
        )
        y_shape = y.shape

        return x, y, y_shape

    def training_step(batch):
        x, y, _ = process_batch(batch)

        loss, parts = dm.training_step(x, y)

        fabric.backward(loss.mean())

        optimizer.step()
        scheduler.step()

        return model, (loss, parts)

    def val_step(batch):
        with torch.no_grad():
            x, y, y_shape = process_batch(batch)

            # unpad
            x_hat = dm.generate(x.shape, y)
            x_hat = x_hat[:, :, :31]

            mag, phase = torch.chunk(x_hat, chunks=2, dim=1)
            mag = mag / 2 + 0.5

            mag = db_to_power(mag * (math.log(3600) - math.log(1e-10)))
            mag = mag**0.5

            phase *= torch.pi

            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)

            spec = torch.complex(real, imag)

            spec = rearrange(spec, "b c f t -> (b c) f t")

            motion_wav = torch.istft(
                spec, n_fft=60, hop_length=60 // 4, length=batch["poses"].shape[1]
            )
            poses = rearrange(motion_wav, "(b c) t -> b t c", b=x.shape[0])

            max_val = fabric.to_device(metadata["max"])
            min_val = fabric.to_device(metadata["min"])

            poses = poses * (max_val - min_val) + min_val

            trans = poses[:, :, :3]
            pose = rearrange(poses[:, :, 3:], "n t (j c) -> n t j c", j=24)

            pose = ax_from_6v(pose)
            positions = skeleton.forward(pose, trans)

            return positions

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

    best_metrics = {
        "fid_k": np.inf,
        "fid_g": np.inf,
    }

    gt_features_k = []
    gt_features_m = []
    for batch in tqdm(train_loader, dynamic_ncols=True):
        gt_features_k.append(batch["features"]["kinetic"].cpu().numpy())
        gt_features_m.append(batch["features"]["geometric"].cpu().numpy())

    gt_features_k = np.concatenate(gt_features_k, axis=0)  # N` x 72 N` >> N
    gt_features_m = np.concatenate(gt_features_m, axis=0)  # N` x 32 N' >> N

    if fabric.world_size > 1:
        gt_features_m = fabric.all_gather(gt_features_m).cpu().numpy()
        gt_features_k = fabric.all_gather(gt_features_k).cpu().numpy()

        gt_features_m = rearrange(gt_features_m, "d n c -> (d n) c")
        gt_features_k = rearrange(gt_features_k, "d n c -> (d n) c")

    for batch in tqdm(
        infinite(train_loader),
        initial=step,
        total=h.training_steps,
        dynamic_ncols=True,
        position=0,
    ):
        step += 1

        if step > h.training_steps:
            break

        if step % ckpt_interval == 0:
            state = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "step": step,
                "best_metrics": best_metrics,
            }

            fabric.save(ckpt_dir / "last.ckpt", state)

        if step % val_interval == 0:
            model = model.eval()

            pred_features_k = []
            pred_features_m = []

            for batch in tqdm(val_loader, position=1, leave=False):
                positions = val_step(batch)

                for position in tqdm(
                    positions, dynamic_ncols=True, position=1, leave=False
                ):
                    kinetic_features = extract_kinetic_features(position.cpu().numpy())
                    geometric_features = extract_manual_features(position.cpu().numpy())

                    pred_features_k.append(kinetic_features)
                    pred_features_m.append(geometric_features)

            pred_features_k = np.stack(pred_features_k)  # Nx72 p40
            pred_features_m = np.stack(pred_features_m)  # Nx32

            if fabric.world_size > 1:
                pred_features_m = fabric.all_gather(pred_features_m).cpu().numpy()
                pred_features_k = fabric.all_gather(pred_features_k).cpu().numpy()

                pred_features_m = rearrange(pred_features_m, "d n c -> (d n) c")
                pred_features_k = rearrange(pred_features_k, "d n c -> (d n) c")

            norm_gt_features_k, pred_features_k = normalize(
                gt_features_k, pred_features_k
            )
            norm_gt_features_m, pred_features_m = normalize(
                gt_features_m, pred_features_m
            )

            fid_k = calc_fid(pred_features_k, norm_gt_features_k)
            fid_g = calc_fid(pred_features_m, norm_gt_features_m)

            div_k_gt = calculate_avg_distance(norm_gt_features_k)
            div_g_gt = calculate_avg_distance(norm_gt_features_m)

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

            for k, v in metrics.items():
                if np.iscomplexobj(v):
                    metrics[k] = np.inf

            if fabric.local_rank == 0:
                wandb.log(
                    metrics,
                    step=step,
                )

            if metrics["fid_k"] < best_metrics["fid_k"]:
                best_metrics = metrics

                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "step": step,
                    "best_metrics": metrics,
                }
                fabric.save(ckpt_dir / "best.ckpt", state)

            model = model.train()

        model, logs = training_step(batch)

        if step % log_interval == 0 and fabric.local_rank == 0:
            logging(logs)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    CLI(main, as_positional=False)
