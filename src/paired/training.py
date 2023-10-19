from dataclasses import dataclass


@dataclass
class HyperParams:
    batch_size: int = 128
    training_steps: int = 300_000
    timesteps: int = 1000
    start: float = 1e-4
    end: float=1e-2
