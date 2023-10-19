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


def main(root: str, h: HyperParams, log_interval: int = 50, val_interval: int = 1000):
    fabric = Fabric(accelerator="gpu", devices=1, precision="16-mixed")

    model = Model(h)
    optimizer = torch.optim.Adam(model.parameters())

    model, optimizer = fabric.setup(model, optimizer)

    data, metadata = load_aistpp(root)

    train_loader = DataLoader(
        data["train"],
        batch_size=h.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        data["val"],
        batch_size=h.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    def training_step(batch):
        ...

    def val_step(batch):
        ...

    def logging():
        print(step)

    step = 0

    while True:
        for batch in tqdm(train_loader, dynamic_ncols=True, position=0):
            step += 1

            if step > h.training_steps:
                break

            if step % log_interval == 0:
                logging()

            if step % val_interval == 0:
                for batch in tqdm(val_loader, position=1):
                    val_step(batch)

            model = training_step(batch)


if __name__ == "__main__":
    CLI(main, as_positional=False)
