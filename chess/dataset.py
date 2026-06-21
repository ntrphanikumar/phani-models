"""Concatenate games into one token stream and serve random blocks."""

import torch

from tokenizer import MoveTokenizer


class DataModule:
    def __init__(self, games, tokenizer: MoveTokenizer, config, train_frac: float = 0.9):
        ids: list[int] = []
        for g in games:
            ids.extend(tokenizer.encode(g))
            ids.append(tokenizer.eos_id)  # mark game boundary
        self._setup(torch.tensor(ids, dtype=torch.long), config, train_frac)

    @classmethod
    def from_tokens(cls, data: torch.Tensor, config, train_frac: float = 0.9):
        """Build directly from a prebuilt 1-D token tensor (e.g. a cached stream)."""
        obj = cls.__new__(cls)
        obj._setup(data, config, train_frac)
        return obj

    def _setup(self, data: torch.Tensor, config, train_frac: float):
        self.config = config
        n = int(train_frac * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        # Each split must hold more tokens than one context window, or we cannot
        # sample a batch. With tiny data this is the first thing that breaks.
        smallest = min(len(self.train_data), len(self.val_data))
        if smallest <= config.block_size:
            raise ValueError(
                f"Not enough data: smallest split has {smallest} tokens but "
                f"block_size is {config.block_size}. Add more games to data/ "
                f"(e.g. Lichess PGNs) or lower block_size in config.py."
            )

    def get_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data
        block_size = self.config.block_size
        ix = torch.randint(len(data) - block_size, (self.config.batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(self.config.device), y.to(self.config.device)
