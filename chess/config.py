"""Hyperparameters for the Chess GPT. Mirrors the phanillm Config."""

from dataclasses import dataclass, asdict

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


@dataclass
class Config:
    # data / batching
    batch_size: int = 32
    block_size: int = 128      # how many moves of context the model sees
    max_games: int | None = 20000  # cap games loaded from PGN (None = all)

    # training schedule
    max_iters: int = 3000
    eval_interval: int = 300
    eval_iters: int = 100
    learning_rate: float = 3e-4

    # model size
    n_embd: int = 192
    n_head: int = 6
    n_layer: int = 4
    dropout: float = 0.1

    # runtime / io
    device: str = pick_device()
    checkpoint_dir: str = "checkpoints"

    def to_dict(self) -> dict:
        return asdict(self)


config = Config()
