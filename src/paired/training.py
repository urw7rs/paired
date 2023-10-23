from dataclasses import dataclass
from typing import Tuple


@dataclass
class HyperParams:
    seed: int = 42
    batch_size: int = 128
    training_steps: int = 300_000
    timesteps: int = 1000
    start: float = 1e-4
    end: float = 2e-2
    lr: float = 2e-4
    channels :Tuple[int, ...] = (512, 512, 512, 512)
