# Chess GPT — Design Spec

**Date:** 2026-06-21
**Status:** Approved (design phase)
**Project:** `phani-models/chess/`

## Goal

Build a decoder-only Transformer that learns chess by **imitation**: given the
moves played so far, predict the next move. This is the "next-token prediction"
game from the `phanillm` text model, with chess moves as tokens instead of
characters.

This is **Option A** of a planned ladder:

1. **A — Chess GPT (this spec):** imitate games, learn move patterns. Runs on CPU.
2. B — Self-play reinforcement (later): the model plays *itself* and improves.
   Needs GPU. Reuses A's network as its "policy."
3. C — Position evaluation head (later): score who's winning. Building block for B.

A is intentionally designed so B and C slot in as *additions*, not rewrites.

## Non-Goals (explicitly out of scope)

- Reinforcement learning / our-model self-play (that is B).
- Tree search (MCTS).
- Position-evaluation head (that is C).
- Elo / strength measurement.
- GPU-scale training.

## Key Concepts / Clarifications

- **Imitation, not understanding.** The model copies patterns from finished
  games. It has no rulebook and does not "know" how to win.
- **Stockfish self-play is a *data source*, not self-learning.** Stockfish
  playing itself produces high-quality games we imitate. This is distinct from
  our model improving by playing itself (that is B).
- **Any PGN source is welcome.** Lichess human games and Stockfish-generated
  games are both just `.pgn` files feeding the same trainer.

## Dependencies

- `torch`, `numpy`, `tqdm` — same as `phanillm`.
- `python-chess` — parse PGN, maintain board state, enumerate **legal moves**.
- (Optional) Stockfish engine binary — only for `gen_stockfish.py`.

## Move Representation (move-as-word)

- One token = one full move in SAN: `e4`, `Nf3`, `O-O`, `Qxd7`.
- A game `1.e4 e5 2.Nf3 Nc6` → `["e4","e5","Nf3","Nc6"]`.
- All games concatenated into one token stream, separated by a special `<eos>`
  token so the model learns where games begin and end.
- Vocabulary = every distinct move seen in the data + `<eos>`
  (~1,500–3,000 tokens depending on corpus size).

## Project Structure

```
phani-models/chess/
  data/
    pgn/              # drop ANY .pgn here (Lichess, Stockfish, etc.)
    sample.pgn        # tiny bundled set so the pipeline runs in minutes
  checkpoints/        # model.pt + tokenizer.json land here
  config.py           # hyperparameters (same shape as phanillm Config)
  pgn_loader.py       # read .pgn files -> list[list[str]] (games of SAN moves)
  tokenizer.py        # MoveTokenizer: move <-> int, encode/decode, save/load JSON
  dataset.py          # concat games into token stream, get_batch(split)
  model.py            # GPTLanguageModel (essentially copied from phanillm)
  train.py            # training loop -> checkpoint
  play.py             # generate a game / play vs the model, with legal masking
  gen_stockfish.py    # OPTIONAL: Stockfish self-play -> writes .pgn into data/pgn/
  requirements.txt
  README.md
```

## Data Flow

```
data/pgn/*.pgn
  -> pgn_loader        (parse to list of games, each a list of SAN moves)
  -> tokenizer         (build vocab from all moves + <eos>)
  -> dataset           (concat to one token stream, serve random blocks)
  -> model             (predict next move token; cross-entropy loss)
  -> checkpoints/model.pt + tokenizer.json

play.py
  load model + tokenizer
  -> step move-by-move, maintaining a real board via python-chess
  -> at each turn: model probs -> keep legal moves only -> renormalize -> sample
```

## Components

### config.py
Dataclass mirroring `phanillm` (`batch_size`, `block_size`, `max_iters`,
`eval_interval`, `eval_iters`, `learning_rate`, `n_embd`, `n_head`, `n_layer`,
`dropout`, `device`, `checkpoint_dir`). Defaults tuned modest for CPU. Notes:
`n_embd` must be divisible by `n_head`.

### pgn_loader.py
- `load_games(pgn_dir) -> list[list[str]]`: read every `.pgn` in the directory,
  parse each game with python-chess, return each game as a list of SAN move
  strings. Skip malformed games gracefully.

### tokenizer.py
- `MoveTokenizer`: builds vocab from a list of games; `encode(moves) -> list[int]`,
  `decode(ids) -> list[str]`, `save(path)`, `load(path)`. Reserves `<eos>`.

### dataset.py
- Concatenate all games (with `<eos>` between) into one 1-D tensor.
- 90/10 train/val split.
- `get_batch(split)` returns `x, y` of shape `(batch_size, block_size)`, `y` =
  `x` shifted by one — same contract as phanillm.

### model.py
- `GPTLanguageModel`, `Block`, `MultiHeadAttention`, `Head`, `FeedForward` —
  the architecture from phanillm, unchanged in spirit. Generic over vocab size.

### train.py
- Load games -> build + save tokenizer -> data module -> model -> AdamW + grad
  clip -> train `max_iters` -> periodic train/val loss -> save checkpoint + config.

### play.py
- Load tokenizer + checkpoint, rebuild model.
- Generate a game move-by-move. Maintain real board with python-chess.
- **Legal masking (default on):** restrict the model's distribution to legal
  moves, renormalize, sample. Flag `--no-legal-mask` to watch the raw model
  attempt illegal moves (educational).
- Print the game (SAN, and/or board).

### gen_stockfish.py (optional)
- If Stockfish engine is available, have it play itself N games and write `.pgn`
  into `data/pgn/`. If not installed, print a friendly install hint and exit.

## Verification / Success Criteria

- **tokenizer round-trip:** `decode(encode(game)) == game` for sample games.
- **pgn_loader:** parses `sample.pgn` into the expected number of games.
- **dataset:** `get_batch` returns correct shapes; `y` is `x` shifted by one.
- **smoke train:** a short run on `sample.pgn` shows loss falling from ~ln(vocab).
- **play:** with legal-masking on, `play.py` produces a full game of **legal**
  moves end to end.

## Tuning Notes

- Smaller/faster: lower `n_embd`, `n_layer`, `n_head`, `block_size`.
- Larger/stronger: raise them + `max_iters`, and (most importantly) **more games**.
- Model quality is gated by data quantity/quality far more than by size.

## Seams Left for Later (B / C)

- The move-vocabulary output layer is already a "policy over moves" — B's policy
  network.
- python-chess board management in `play.py` is the substrate for adding search.
- A second output head can be added to `model.py` for position value (C).
