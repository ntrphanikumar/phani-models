"""OPTIONAL: generate self-play games with the Stockfish engine.

These games are just another DATA SOURCE for imitation training (not the
model teaching itself). Requires the stockfish binary on PATH or via
--engine. Run: python gen_stockfish.py --games 50 --movetime 0.05
"""

import argparse
import os
import shutil

import chess
import chess.engine
import chess.pgn


def main():
    parser = argparse.ArgumentParser(description="Generate Stockfish self-play PGNs.")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--movetime", type=float, default=0.05, help="seconds per move")
    parser.add_argument("--engine", default=shutil.which("stockfish"))
    parser.add_argument("--out", default="data/pgn/stockfish.pgn")
    args = parser.parse_args()

    if not args.engine or not os.path.exists(args.engine):
        print(
            "Stockfish engine not found.\n"
            "Install it (macOS: `brew install stockfish`) or pass --engine /path/to/stockfish.\n"
            "This script is optional; the project trains fine on data/pgn/ PGNs from any source."
        )
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            for n in range(args.games):
                board = chess.Board()
                while not board.is_game_over():
                    result = engine.play(board, chess.engine.Limit(time=args.movetime))
                    board.push(result.move)
                game = chess.pgn.Game.from_board(board)
                game.headers["Event"] = f"Stockfish self-play {n + 1}"
                print(game, file=f, end="\n\n")
                print(f"game {n + 1}/{args.games} done ({board.result()})")
    finally:
        engine.quit()
    print(f"wrote {args.games} games -> {args.out}")


if __name__ == "__main__":
    main()
