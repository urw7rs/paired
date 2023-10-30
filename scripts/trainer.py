from jsonargparse import CLI

from paired import ddpm
from paired.data import loaders


def fit(
        hparams: ddpm.trainer.HyperParams,
        ckpt: str, train: str, val:str,
        steps: int = 300_000, batch_size=64, num_workers: int = 8
):
    train_loader = loaders.make_train(
        train, steps=steps, batch_size=batch_size, num_workers=num_workers
    )

    val_loader = loaders.make_eval(
        val, steps=steps, batch_size=batch_size, num_workers=num_workers
    )

    ddpm.trainer.fit(hparams, ckpt, train_loader, val_loader)


def eval():
    ...


if __name__ == "__main__":
    CLI([fit, eval], as_positional=False)
