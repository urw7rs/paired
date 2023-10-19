from pathlib import Path

import joblib
import torch
from jsonargparse import CLI
from lightning.fabric import Fabric
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from paired.data.aistpp import load_aistpp
from paired.diffusion.training import HyperParams


class Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x, c):
        return x, c


def infinite(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def forward_diffuse(h: HyperParams, x_0, t):
    ...

def main(root: str, h: HyperParams, log_interval: int = 50, val_interval: int = 1000):
    fabric = Fabric(accelerator="gpu", devices=1, precision="16-mixed")

    model = Model(h)
    optimizer = torch.optim.Adam(model.parameters())

    model, optimizer = fabric.setup(model, optimizer)

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

    def training_step(batch):
        return model

        mag = batch["mag"]
        phase = batch["phase"]

        spec = torch.cat((mag, phase), dim=-1)
        mel = batch["mel"]

        t =torch.randint(low=1, high=h.timesteps, size=(h.batch_size,))

        noisy_spec = forward_diffuse(h, spec, t)
        noisy_mel =  forward_diffuse(h, mel, t)

        denoised_spec, denoised_mel = model(noisy_spec, noisy_mel)

        mel_l2 = (denoised_mel - mel) ** 2
        mel_l2 = mel_l2.mean()

        spec_l2 = (denoised_spec - spec) ** 2
        spec_l2 = spec_l2.mean()

        loss = spec_l2 + mel_l2

        fabric.backward(loss.mean())

        optimizer.step()

        return model

    def val_step(batch):
        ...

    def logging():
        ...

    step = 0

    for batch in tqdm(infinite(train_loader), total=h.training_steps, dynamic_ncols=True, position=0):
        step += 1

        if step > h.training_steps:
            break

        if step % log_interval == 0:
            logging()

        if step % val_interval == 0:
            model = model.eval()

            for batch in tqdm(val_loader, position=1, leave=False):
                val_step(batch)

            model = model.train()

        model = training_step(batch)


if __name__ == "__main__":
    CLI(main, as_positional=False)
