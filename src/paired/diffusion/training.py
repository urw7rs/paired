from dataclasses import dataclass


@dataclass
class HyperParams:
    batch_size: int = 128
    training_steps: int = 300_000
