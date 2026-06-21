"""Generate chess games from a trained model, masking to legal moves.

Run: python play.py --max_moves 60
"""

import argparse
import os

import chess
import torch
from torch.nn import functional as F

from config import Config
from model import GPTLanguageModel
from tokenizer import MoveTokenizer

DEFAULT_CKPT = "checkpoints/model.pt"
DEFAULT_TOKENIZER = "checkpoints/tokenizer.json"


def legal_move_token_probs(probs, board, tokenizer):
    """Map each legal chess.Move to the model's probability for its SAN token."""
    mapping = {}
    for move in board.legal_moves:
        san = board.san(move)
        tid = tokenizer.stoi.get(san)
        mapping[move] = float(probs[tid]) if tid is not None else 0.0
    return mapping


def _sample_move(model, tokenizer, board, idx, config, legal_mask, generator):
    idx_cond = idx[:, -config.block_size :]
    logits, _ = model(idx_cond)
    # Sample on CPU so the (CPU) generator matches the probabilities' device.
    probs = F.softmax(logits[0, -1, :], dim=-1).cpu()

    if legal_mask:
        mapping = legal_move_token_probs(probs, board, tokenizer)
        total = sum(mapping.values())
        if total <= 0:  # model gave no mass to any legal move -> uniform legal
            legal = list(board.legal_moves)
            move = legal[int(torch.randint(len(legal), (1,), generator=generator))]
            return move
        moves = list(mapping.keys())
        weights = torch.tensor([mapping[m] for m in moves])
        pick = int(torch.multinomial(weights / weights.sum(), 1, generator=generator))
        return moves[pick]

    # raw mode: sample any token; may be illegal -> caller handles
    tid = int(torch.multinomial(probs, 1, generator=generator))
    san = tokenizer.itos[tid]
    try:
        return board.parse_san(san)
    except ValueError:
        return None  # illegal move attempted


def play_game(model, tokenizer, config, max_moves=60, legal_mask=True, seed=0):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    board = chess.Board()
    idx = torch.zeros((1, 1), dtype=torch.long, device=config.device)  # start w/ eos
    sans = []
    for _ in range(max_moves):
        if board.is_game_over():
            break
        with torch.no_grad():
            move = _sample_move(model, tokenizer, board, idx, config, legal_mask, generator)
        if move is None:  # illegal attempt in raw mode -> stop
            break
        san = board.san(move)
        sans.append(san)
        board.push(move)
        tid = tokenizer.stoi.get(san, tokenizer.eos_id)
        idx = torch.cat([idx, torch.tensor([[tid]], device=config.device)], dim=1)
    return sans


def main():
    parser = argparse.ArgumentParser(description="Generate a game from the Chess GPT.")
    parser.add_argument("--max_moves", type=int, default=60)
    parser.add_argument("--no-legal-mask", action="store_true")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint} (train first)")

    tokenizer = MoveTokenizer.load(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = Config(**ckpt["config"])
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPTLanguageModel(tokenizer.vocab_size, config).to(config.device)
    model.load_state_dict(ckpt["model_state"])

    sans = play_game(
        model, tokenizer, config,
        max_moves=args.max_moves,
        legal_mask=not args.no_legal_mask,
        seed=args.seed,
    )

    # pretty print as 1. e4 e5 2. Nf3 ...
    out = []
    for i, san in enumerate(sans):
        if i % 2 == 0:
            out.append(f"{i // 2 + 1}.")
        out.append(san)
    print(" ".join(out))


if __name__ == "__main__":
    main()
