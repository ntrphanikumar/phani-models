"""Parse PGN games ONCE into a token stream and cache it to disk.

Parsing millions of games with python-chess is the slow part of startup, and we
don't want to repeat it on every training run. This builds the token stream once
(parallel parse, encoding incrementally to keep memory bounded) and saves:
  <out>/tokens.npy      int32 array: all games concatenated, <eos> between them
  <out>/tokenizer.json  the move vocabulary

train.py loads these instantly when CHESS_CACHE points at <out>.

Run: python build_cache.py --pgn_dir data/pgn --out data/cache --max_games 3000000
"""

import argparse
import glob
import os
from array import array
from multiprocessing import Pool

import numpy as np

from pgn_loader import _parse_file
from tokenizer import MoveTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn_dir", default="data/pgn")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--max_games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.pgn_dir, "**", "*.pgn"), recursive=True))
    if not paths:
        raise SystemExit(f"no .pgn files under {args.pgn_dir}")
    workers = args.workers or min(len(paths), os.cpu_count() or 1)
    print(f"parsing {len(paths)} files with {workers} workers...")

    stoi = {MoveTokenizer.EOS: 0}      # build vocab incrementally
    buf = array("i")                    # int32 token stream
    total_games = 0

    # imap_unordered yields one file's games at a time -> we encode and drop it,
    # so peak memory is ~one file + the growing int stream (not all games).
    with Pool(workers) as pool:
        for file_games in pool.imap_unordered(_parse_file, paths):
            for moves in file_games:
                for m in moves:
                    j = stoi.get(m)
                    if j is None:
                        j = len(stoi)
                        stoi[m] = j
                    buf.append(j)
                buf.append(0)  # <eos> between games
                total_games += 1
                if args.max_games and total_games >= args.max_games:
                    break
            print(f"  {total_games:,} games | {len(buf):,} tokens | vocab {len(stoi):,}")
            if args.max_games and total_games >= args.max_games:
                break

    os.makedirs(args.out, exist_ok=True)
    vocab = [None] * len(stoi)
    for m, i in stoi.items():
        vocab[i] = m
    MoveTokenizer(vocab).save(os.path.join(args.out, "tokenizer.json"))
    tokens = np.frombuffer(buf, dtype=np.int32)
    np.save(os.path.join(args.out, "tokens.npy"), tokens)

    print(f"\ncached {total_games:,} games -> {len(tokens):,} tokens, "
          f"vocab {len(vocab):,}")
    print(f"  {os.path.join(args.out, 'tokens.npy')}")
    print(f"  {os.path.join(args.out, 'tokenizer.json')}")


if __name__ == "__main__":
    main()
