import torch
from config import Config
from tokenizer import MoveTokenizer
from dataset import DataModule


def _many_games():
    # enough tokens to exceed block_size
    return [["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"] for _ in range(80)]


def test_batch_shapes_and_shift():
    games = _many_games()
    tok = MoveTokenizer.from_games(games)
    cfg = Config()
    cfg.batch_size = 4
    cfg.block_size = 8
    cfg.device = "cpu"
    dm = DataModule(games, tok, cfg)

    x, y = dm.get_batch("train")
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y is x shifted by one: y[:, :-1] == x[:, 1:]
    assert torch.equal(y[:, :-1], x[:, 1:])
