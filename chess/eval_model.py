"""Quick quality eval for a trained Chess GPT.

Measures, over real held-out positions (teacher-forced through actual games):
  - top-1 legal rate  : how often the model's single most-likely move is legal
                        (did it learn the rules well enough to even be legal?)
  - top-1 match rate  : how often that move equals the move a human actually played
  - raw self-play legality: how many moves it plays unaided before an illegal one

Run: python eval_model.py --games 1500 --selfplay 5
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
def teacher_forced_eval(model, tokenizer, config, games):
    """Walk real games move-by-move; check the model's top-1 prediction each ply."""
    dev = config.device
    positions = legal_top1 = match_top1 = 0
    for moves in games:
        board = chess.Board()
        ids = [tokenizer.eos_id]
        for actual in moves:
            idx = torch.tensor([ids[-config.block_size:]], dtype=torch.long, device=dev)
            logits, _ = model(idx)
            top = int(torch.argmax(logits[0, -1, :]))
            pred = tokenizer.itos[top]

            legal_sans = {board.san(m) for m in board.legal_moves}
            positions += 1
            if pred in legal_sans:
                legal_top1 += 1
            if pred == actual:
                match_top1 += 1

            ids.append(tokenizer.stoi.get(actual, tokenizer.eos_id))
            board.push(board.parse_san(actual))
    return positions, legal_top1, match_top1


@torch.no_grad()
def selfplay_legality(model, tokenizer, config, n_games, max_plies=100):
    """Let the model pick its OWN top move (no legal mask); count plies until illegal."""
    dev = config.device
    lengths = []
    for g in range(n_games):
        board = chess.Board()
        ids = [tokenizer.eos_id]
        plies = 0
        for _ in range(max_plies):
            idx = torch.tensor([ids[-config.block_size:]], dtype=torch.long, device=dev)
            logits, _ = model(idx)
            # sample from top to vary games per seed-free run index
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
    ap.add_argument("--games", type=int, default=1500, help="held-out games to score")
    ap.add_argument("--selfplay", type=int, default=5)
    ap.add_argument("--pgn_dir", default=os.environ.get("CHESS_PGN_DIR", "data"))
    ap.add_argument("--checkpoint", default="checkpoints/model.pt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    args = ap.parse_args()

    model, tokenizer, config = load_model(args.checkpoint, args.tokenizer)
    print(f"device: {config.device} | block_size: {config.block_size}")

    # Use the LAST N games in the corpus as a held-out sample (training samples
    # random blocks, so these tail games are a reasonable proxy for unseen play).
    games = load_games(args.pgn_dir, max_games=200000)[-args.games:]
    print(f"evaluating on {len(games)} games...")

    pos, legal, match = teacher_forced_eval(model, tokenizer, config, games)
    print(f"\npositions scored : {pos:,}")
    print(f"top-1 legal rate : {100*legal/pos:.1f}%   (model's best move is a legal move)")
    print(f"top-1 match rate : {100*match/pos:.1f}%   (model's best move = human's actual move)")

    lengths = selfplay_legality(model, tokenizer, config, args.selfplay)
    print(f"\nraw self-play legality (no mask): plies before illegal = {lengths}")
    print(f"  avg legal plies unaided: {sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    main()
