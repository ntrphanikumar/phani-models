"""Move-as-word tokenizer: each SAN move is one token."""

import json
import os


class MoveTokenizer:
    EOS = "<eos>"

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.stoi = {m: i for i, m in enumerate(vocab)}
        self.itos = {i: m for i, m in enumerate(vocab)}
        self.eos_id = self.stoi[self.EOS]

    @classmethod
    def from_games(cls, games: list[list[str]]) -> "MoveTokenizer":
        moves = set()
        for g in games:
            moves.update(g)
        # EOS first (index 0), then moves sorted for determinism
        vocab = [cls.EOS] + sorted(moves)
        return cls(vocab)

    def encode(self, moves: list[str]) -> list[int]:
        return [self.stoi[m] for m in moves]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids if i != self.eos_id]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str) -> "MoveTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(meta["vocab"])
