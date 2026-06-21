# Chess GPT — learn chess by imitation

A tiny decoder-only Transformer (same architecture as `../phanillm`) that learns
chess by predicting the **next move** from move history. It imitates games — it
does not "know" the rules or how to win. Legal play at generation time is
enforced by masking the model's output to legal moves via `python-chess`.

This is **Option A** of a ladder: imitation now; self-play reinforcement (B) and
a position-evaluation head (C) are designed-for but not built yet. See
`docs/superpowers/specs/2026-06-21-chess-gpt-design.md`.

## Install

```bash
pip install -r requirements.txt
```

## Add training games

Drop any `.pgn` files into `data/pgn/` (or keep them in `data/`). Sources:
- **Lichess** database (https://database.lichess.org/) — real human games.
- **Stockfish self-play** — run `python gen_stockfish.py` (needs the stockfish
  engine: `brew install stockfish`). These are just another data source.

A tiny `data/sample.pgn` is bundled so the pipeline runs immediately.

**Using an external disk** (e.g. to keep big downloads off a small internal SSD):
point the trainer at any folder of `.pgn` files via `CHESS_PGN_DIR`:

```bash
CHESS_PGN_DIR=/Volumes/HPSSD/phani-chess-data/pgn python train.py
```

`config.max_games` caps how many games are loaded (default 20000) so a large
monthly file won't exhaust RAM or time.

## Train

```bash
python train.py
```

Prints games loaded, vocab size, parameter count, and falling train/val loss;
saves `checkpoints/model.pt` and `checkpoints/tokenizer.json`.

## Generate a game

```bash
python play.py --max_moves 60            # legal moves only (default)
python play.py --max_moves 60 --no-legal-mask   # watch the raw model (may stop on an illegal move)
```

## What to expect

With the tiny sample data the model plays near-random but **legal** openings.
For real, human-looking play, download a few thousand Lichess games into
`data/pgn/` and raise `max_iters`. Quality is gated by data far more than model size.

## Tuning (config.py)

- Smaller/faster: lower `n_embd`, `n_layer`, `n_head`, `block_size`.
- Stronger: raise them + `max_iters`, and add **more games**.
- `n_embd` must be divisible by `n_head`.

New to this and wondering what "104M parameters" actually means or how model
size is calculated? See [docs/model-size-explained.md](docs/model-size-explained.md)
— a from-scratch, no-jargon walkthrough.

**Resuming work / project status:** see
[docs/SESSION-HANDOFF-2026-06-21.md](docs/SESSION-HANDOFF-2026-06-21.md) — full
state (trained models, infra, decisions, next steps). Trained checkpoints are on
the [Releases page](https://github.com/ntrphanikumar/phani-models/releases)
(`chess-104m-v1`, `chess-158m-v2`).

## Run tests

```bash
python -m pytest -v
```
