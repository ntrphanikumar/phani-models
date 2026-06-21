"""Read PGN files into lists of SAN moves, one list per game.

Parsing (python-chess generating SAN for every move) is CPU-bound and the slow
part of startup. When several PGN files are present we parse them in parallel,
one worker process per file.
"""

import glob
import os
from multiprocessing import Pool

import chess.pgn


def _parse_file(path: str) -> list[list[str]]:
    games: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            moves: list[str] = []
            board = game.board()
            try:
                for move in game.mainline_moves():
                    moves.append(board.san(move))
                    board.push(move)
            except (ValueError, AssertionError):
                continue  # skip malformed game
            if moves:
                games.append(moves)
    return games


def load_games(pgn_dir: str, max_games: int | None = None,
               workers: int | None = None) -> list[list[str]]:
    paths = sorted(glob.glob(os.path.join(pgn_dir, "**", "*.pgn"), recursive=True))
    if not paths:
        return []

    if workers is None:
        workers = min(len(paths), os.cpu_count() or 1)

    games: list[list[str]] = []
    if workers > 1 and len(paths) > 1:
        # One process per file. imap_unordered yields each file's games as it
        # finishes, so a max_games cap stops work early and bounds memory
        # (pool.map would parse every file before we could apply the cap).
        with Pool(workers) as pool:
            for file_games in pool.imap_unordered(_parse_file, paths):
                games.extend(file_games)
                if max_games is not None and len(games) >= max_games:
                    break
    else:
        for path in paths:
            games.extend(_parse_file(path))
            if max_games is not None and len(games) >= max_games:
                break

    if max_games is not None:
        games = games[:max_games]
    return games
