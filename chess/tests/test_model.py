import torch
from config import Config
from model import GPTLanguageModel


def _cfg():
    c = Config()
    c.n_embd = 32
    c.n_head = 4
    c.n_layer = 2
    c.block_size = 16
    c.dropout = 0.0
    c.device = "cpu"
    return c


def test_forward_shapes_and_loss():
    cfg = _cfg()
    vocab = 20
    model = GPTLanguageModel(vocab, cfg)
    x = torch.randint(0, vocab, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, vocab)
    assert loss.ndim == 0  # scalar
    # without targets -> no loss
    logits2, none = model(x)
    assert none is None


def test_generate_extends_sequence():
    cfg = _cfg()
    model = GPTLanguageModel(20, cfg)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 6)
