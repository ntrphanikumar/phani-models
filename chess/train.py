"""Train the Chess GPT on PGN games in data/. Run: python train.py"""

import math
import os
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from config import config
from dataset import DataModule
from model import GPTLanguageModel
from pgn_loader import load_games
from tokenizer import MoveTokenizer

# Where to read PGN games from. Override to use an external disk, e.g.:
#   CHESS_PGN_DIR=/Volumes/HPSSD/phani-chess-data/pgn python train.py
PGN_DIR = os.environ.get("CHESS_PGN_DIR", "data")
# Output location for weights/tokenizer (override to keep runs separate):
#   CHESS_CHECKPOINT_DIR=checkpoints_v2 python train.py
config.checkpoint_dir = os.environ.get("CHESS_CHECKPOINT_DIR", config.checkpoint_dir)
TOKENIZER_PATH = os.path.join(config.checkpoint_dir, "tokenizer.json")
CHECKPOINT_PATH = os.path.join(config.checkpoint_dir, "model.pt")


def amp_context():
    """Mixed precision (bf16) on CUDA for ~2x speed; no-op elsewhere."""
    if config.device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def get_lr(it):
    """Linear warmup then cosine decay to 10% of the base LR."""
    warmup = max(100, config.max_iters // 100)
    min_lr = config.learning_rate * 0.1
    if it < warmup:
        return config.learning_rate * (it + 1) / warmup
    ratio = min(1.0, (it - warmup) / max(1, config.max_iters - warmup))
    return min_lr + 0.5 * (config.learning_rate - min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def estimate_loss(model, dm):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = dm.get_batch(split)
            with amp_context():
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(1337)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # CHESS_MAX_GAMES overrides the config cap. Use "all"/"0"/"none" for no cap.
    env_cap = os.environ.get("CHESS_MAX_GAMES")
    if env_cap is not None:
        config.max_games = None if env_cap.lower() in ("all", "0", "none") else int(env_cap)

    # Optional hyperparameter overrides via env (handy for scaling up on a GPU box):
    #   CHESS_N_EMBD=384 CHESS_N_LAYER=8 CHESS_MAX_ITERS=20000 python train.py
    for _k in ("n_embd", "n_head", "n_layer", "block_size", "batch_size",
               "max_iters", "eval_interval", "eval_iters"):
        _v = os.environ.get("CHESS_" + _k.upper())
        if _v is not None:
            setattr(config, _k, int(_v))

    # Fast path: if CHESS_CACHE points at a prebuilt token cache, load it
    # instantly instead of re-parsing PGN (see build_cache.py).
    cache_dir = os.environ.get("CHESS_CACHE")
    cache_tokens = os.path.join(cache_dir, "tokens.npy") if cache_dir else None
    cache_tok = os.path.join(cache_dir, "tokenizer.json") if cache_dir else None

    if cache_dir and os.path.exists(cache_tokens) and os.path.exists(cache_tok):
        print(f"loading token cache from {cache_dir} ...")
        tokenizer = MoveTokenizer.load(cache_tok)
        ids = np.load(cache_tokens)
        print(f"cache: {len(ids):,} tokens | vocab {tokenizer.vocab_size}")
        dm = DataModule.from_tokens(torch.from_numpy(ids).long(), config)
    else:
        games = load_games(PGN_DIR, max_games=config.max_games)
        print(f"loaded {len(games)} games (cap: {config.max_games})")
        tokenizer = MoveTokenizer.from_games(games)
        print(f"vocab size: {tokenizer.vocab_size}")
        dm = DataModule(games, tokenizer, config)

    # Always save the tokenizer next to the checkpoint so play.py can use it.
    tokenizer.save(TOKENIZER_PATH)

    model = GPTLanguageModel(tokenizer.vocab_size, config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M  |  device: {config.device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def save_checkpoint():
        torch.save(
            {"model_state": model.state_dict(), "config": config.to_dict()},
            CHECKPOINT_PATH,
        )

    best_val = float("inf")
    for it in tqdm(range(config.max_iters), desc="training"):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            losses = estimate_loss(model, dm)
            tqdm.write(
                f"step {it}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, lr {lr:.2e}"
            )
            # Periodic save so a long run survives a crash; keep the best-val model.
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint()
                tqdm.write(f"  saved checkpoint (best val {best_val:.4f})")

        x, y = dm.get_batch("train")
        with amp_context():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    save_checkpoint()
    print(f"saved checkpoint -> {CHECKPOINT_PATH}")
    print(f"saved tokenizer  -> {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
