"""Fast quality eval for a trained Chess GPT.

Metrics over real games (teacher-forced):
  - top-1 legal rate : how often the model's most-likely move is legal
  - top-1 match rate : how often it equals the move a human actually played
  - raw self-play legality: plies the model plays unaided before an illegal move

Speed: each game is scored in ONE batched forward pass (all positions at once),
and legality is checked with a single parse_san (no enumerating every legal move).

Run: python eval_model.py --games 1200 --selfplay 6
"""

import argparse
import os

import chess
import torch
from torch.nn import functional as F

from config import Config
from model import GPTLanguageModel
from pgn_loader import load_games
from tokenizer import MoveTokenizer


def load_model(ckpt_path, tok_path):
    tokenizer = MoveTokenizer.load(tok_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = Config(**ckpt["config"])
    config.device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model = GPTLanguageModel(tokenizer.vocab_size, config).to(config.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, config


@torch.no_grad()
def teacher_forced_eval(model, tokenizer, config, games, batch=64):
    """Score whole games in batched forward passes (one pass per batch)."""
    dev = config.device
    bs = config.block_size
    positions = legal = match = 0

    for i in range(0, len(games), batch):
        chunk = games[i:i + batch]
        # sequence fed to the model: [eos, move0, move1, ...] truncated to block_size
        seqs = []
        for moves in chunk:
            ids = [tokenizer.eos_id] + [tokenizer.stoi.get(m, tokenizer.eos_id) for m in moves]
            seqs.append(ids[:bs])
        maxlen = max(len(s) for s in seqs)
        padded = [s + [tokenizer.eos_id] * (maxlen - len(s)) for s in seqs]
        x = torch.tensor(padded, dtype=torch.long, device=dev)

        logits, _ = model(x)                       # (B, maxlen, vocab)
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

        # position t (input ids[:t+1]) predicts moves[t]; replay board to that point
        for gi, moves in enumerate(chunk):
            board = chess.Board()
            n = min(len(moves), bs - 1)
            for t in range(n):
                pred_san = tokenizer.itos.get(preds[gi][t], "")
                actual = moves[t]
                positions += 1
                try:
                    board.parse_san(pred_san)        # cheap legality check
                    legal += 1
                except (ValueError, AssertionError):
                    pass
                if pred_san == actual:
                    match += 1
                board.push(board.parse_san(actual))
    return positions, legal, match


@torch.no_grad()
def selfplay_legality(model, tokenizer, config, n_games, max_plies=100):
    """Model picks its own move (no legal mask); count plies until an illegal one."""
    dev = config.device
    lengths = []
    for g in range(n_games):
        board = chess.Board()
        ids = [tokenizer.eos_id]
        plies = 0
        for _ in range(max_plies):
            idx = torch.tensor([ids[-config.block_size:]], dtype=torch.long, device=dev)
            logits, _ = model(idx)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            tok = int(torch.argmax(probs)) if g == 0 else int(torch.multinomial(probs, 1))
            pred = tokenizer.itos[tok]
            try:
                mv = board.parse_san(pred)
            except ValueError:
                break
            board.push(mv)
            ids.append(tok)
            plies += 1
            if board.is_game_over():
                break
        lengths.append(plies)
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--selfplay", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pgn_dir", default=os.environ.get("CHESS_PGN_DIR", "data"))
    ap.add_argument("--checkpoint", default="checkpoints/model.pt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    args = ap.parse_args()

    model, tokenizer, config = load_model(args.checkpoint, args.tokenizer)
    print(f"device: {config.device} | block_size: {config.block_size}")

    # Load a modest sample and take the tail as the eval set (cheap to parse).
    games = load_games(args.pgn_dir, max_games=max(20000, args.games))[-args.games:]
    print(f"evaluating on {len(games)} games...")

    pos, legal, match = teacher_forced_eval(model, tokenizer, config, games, batch=args.batch)
    print(f"\npositions scored : {pos:,}")
    print(f"top-1 legal rate : {100*legal/pos:.1f}%   (model's best move is legal)")
    print(f"top-1 match rate : {100*match/pos:.1f}%   (best move = human's actual move)")

    lengths = selfplay_legality(model, tokenizer, config, args.selfplay)
    print(f"\nraw self-play (no mask): plies before illegal = {lengths}")
    print(f"  avg legal plies unaided: {sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    main()
