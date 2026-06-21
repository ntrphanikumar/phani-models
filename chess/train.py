"""Train the Chess GPT on PGN games in data/. Run: python train.py"""

import os

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
TOKENIZER_PATH = os.path.join(config.checkpoint_dir, "tokenizer.json")
CHECKPOINT_PATH = os.path.join(config.checkpoint_dir, "model.pt")


@torch.no_grad()
def estimate_loss(model, dm):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = dm.get_batch(split)
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

    games = load_games(PGN_DIR, max_games=config.max_games)
    print(f"loaded {len(games)} games (cap: {config.max_games})")

    tokenizer = MoveTokenizer.from_games(games)
    tokenizer.save(TOKENIZER_PATH)
    print(f"vocab size: {tokenizer.vocab_size}")

    dm = DataModule(games, tokenizer, config)

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
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            losses = estimate_loss(model, dm)
            tqdm.write(
                f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            # Periodic save so a long run survives a crash; keep the best-val model.
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint()
                tqdm.write(f"  saved checkpoint (best val {best_val:.4f})")
        x, y = dm.get_batch("train")
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
