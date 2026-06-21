# Chess GPT — Session Handoff (2026-06-21)

Snapshot of everything built/decided this session so work can resume cleanly.
**TL;DR:** Option A (imitation chess GPT) is done. Two models trained & released
to GitHub. Next frontier is Option B (self-play + search). GPU box is the user's
to keep/destroy.

---

## 1. Current state

**Option A (learn chess by imitating games) is COMPLETE.** The model predicts the
next move (move-as-word tokens) from move history; legality at play time is
guaranteed by masking the model's output to legal moves via python-chess.

Two trained models, both released on GitHub
(`git@github.com:ntrphanikumar/phani-models.git`, public, `chess/` subdir):

| Release | Params | Config | Data | Steps | val loss | top-1 legal | top-1 human-match |
|---|---|---|---|---|---|---|---|
| `chess-104m-v1` | 104M | n_embd 768, L12, h12, blk256 | 2M games (76M tok, vocab 7341) | 10k | 2.21 | 95.9% | 41.0% |
| `chess-158m-v2` | 158.6M | n_embd 896, L14, h14, blk256 | full 2013 = 3.4M games (230M tok, vocab 13045) | 20k | **2.05** | **97.1%** | **43.8%** |

v2 is the current best. Both checkpoints are downloadable from the Releases page
(weights are NOT in git — too large; see §6).

**Metric meanings:** val loss = cross-entropy on held-out games (perplexity =
e^loss; 2.05 → ~8 candidate moves of uncertainty). top-1 legal = how often the
model's single best move is legal (learned the rules with no rulebook). top-1
human-match = how often its best move equals the human's actual move (random
≈3%, strong chess-LMs ~50-55%). See `docs/model-size-explained.md` for params.

---

## 2. Infrastructure

**GPU box (Linode):** RTX 4000 Ada x1 Small, **`172.236.180.199`**, $0.52/hr.
- Ubuntu 24.04, 4 vCPU / 15 GB RAM / 476 GB disk, 20 GB VRAM.
- NVIDIA driver 595 installed via **DKMS** (the prebuilt-module package was broken;
  DKMS compiled it against the running kernel — see §5).
- Python venv: **`/root/chess-venv`** (torch 2.12.1+cu130, python-chess, etc.).
- Training/working dir: **`/root/chess`** (rsync'd code + data + checkpoints).
- Git clone: **`/root/phani-models`** (HTTPS origin, for `git pull`).
- **ACCESS REQUIRES THE USER'S VPN.** SSH (port 22) only works when the user's VPN
  is connected. If the VPN drops mid-run, training keeps going (detached via
  nohup); only monitoring pauses. Reconnect VPN to resume.
- **Box lifecycle is the USER'S job** — Claude does NOT provision/reboot/destroy.
  (See memory `feedback_infra_boundary`.) As of handoff the box is **up and idle**
  (billing) — user decides whether to keep it for Option B or destroy it.

**Data locations:**
- Box: `/root/chess/data/pgn/*.pgn` = full 2013 (12 months, 3.4M games).
  Token cache: `/root/chess/data/cache_2013/` (tokens.npy ~879 MB ≈ 230M int32
  tokens, tokenizer.json vocab 13045). v1 weights: `/root/chess/checkpoints/`;
  v2 weights: `/root/chess/checkpoints_v2/`.
- Mac project: `/Users/ntrphanikumar/workspace/personal/phani-models/chess`.
- Mac external SSD archive: `/Volumes/HPSSD/phani-chess-data/pgn/` (2013-01..06
  `.zst` + decompressed 2013-01). Mac uses MPS (Apple GPU) via venv
  `learn-transformers-mamba/.venv` (torch 2.10).
- Lichess source: https://database.lichess.org/ — early-2013 months are small;
  recent months are tens of GB.

**GitHub release uploads** need a Personal Access Token (fine-grained, repo
`phani-models`, **Contents: Read and write**) for the **personal** `ntrphanikumar`
account. SSH key alone is NOT enough (releases use the API). The token used this
session should be deleted by the user; a new one is needed for future releases.

---

## 3. Code map (all in `chess/`)

| File | Role |
|---|---|
| `config.py` | `Config` dataclass; `pick_device()` (cuda > mps > cpu). |
| `pgn_loader.py` | Parse PGN → list of SAN-move games. **Parallel** (one worker/file, `imap_unordered` with early cap). |
| `tokenizer.py` | `MoveTokenizer` (move-as-word, `<eos>`=0), save/load JSON. |
| `dataset.py` | `DataModule` — concat games → token stream → random blocks. `from_tokens()` for cache path. |
| `model.py` | `GPTLanguageModel` (decoder-only causal Transformer). No weight tying (skipped for checkpoint compat). |
| `build_cache.py` | Parse PGN ONCE (streaming, memory-safe) → `tokens.npy` + `tokenizer.json`. |
| `train.py` | Training loop. Loads cache if `CHESS_CACHE` set, else parses. AMP(bf16 on cuda), LR warmup+cosine, best-val + periodic checkpointing. |
| `play.py` | Generate games. Legal-masked (default, 100% legal) or `--no-legal-mask` (raw). |
| `eval_model.py` | Batched teacher-forced eval: top-1 legal/match + raw self-play legality. |
| `gen_stockfish.py` | Optional: Stockfish self-play → PGN (extra data source). |

**Env knobs (all optional):**
`CHESS_PGN_DIR`, `CHESS_MAX_GAMES` (all/N), `CHESS_CACHE` (dir with tokens.npy),
`CHESS_CHECKPOINT_DIR`, and `CHESS_{N_EMBD,N_HEAD,N_LAYER,BLOCK_SIZE,BATCH_SIZE,MAX_ITERS,EVAL_INTERVAL,EVAL_ITERS}`.

**Reproduce v2 training (on the box):**
```bash
cd /root/chess
CHESS_CACHE=data/cache_2013 CHESS_CHECKPOINT_DIR=checkpoints_v2 \
  CHESS_N_EMBD=896 CHESS_N_HEAD=14 CHESS_N_LAYER=14 \
  CHESS_BLOCK_SIZE=256 CHESS_BATCH_SIZE=32 \
  CHESS_MAX_ITERS=20000 CHESS_EVAL_INTERVAL=1000 CHESS_EVAL_ITERS=40 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/chess-venv/bin/python -u train.py
```
**Play / eval v2:**
```bash
python play.py --max_moves 60 --checkpoint checkpoints_v2/model.pt --tokenizer checkpoints_v2/tokenizer.json
CHESS_PGN_DIR=data/pgn python eval_model.py --games 1200 --checkpoint checkpoints_v2/model.pt --tokenizer checkpoints_v2/tokenizer.json
```

---

## 4. Key decisions & learnings

- **Move-as-word tokenization** (one SAN move = one token), not characters.
- **Legality = rules on top of the model.** The model only proposes; python-chess
  filters to legal moves → 100% legal play even though the raw model is ~97% legal.
  This is the simplest form of "logic guiding the model" — and the conceptual
  seed of Option B (search adds "is it *good*?" on top of "is it legal?").
- **Data is the dominant quality lever**, not model size. Model size should roughly
  match token count (158M params ≈ 230M tokens ≈ 0.7 epoch over 20k steps → no
  overfitting; train≈val throughout).
- **Parsing is the bottleneck** (python-chess SAN is CPU-bound, ~12k moves/s/core).
  ~3.4M games ≈ 40 min parse on 4 cores. → built the **token cache** (parse once,
  reuse) + **parallel parse**. Going beyond ~3.4M games needs a *streaming* parser
  (current `build_cache` holds one file's games in RAM at a time; the biggest
  recent Lichess months would still strain 15 GB).
- **Loss is NOT comparable across different vocab sizes** (v2 vocab 13045 vs v1
  7341 → higher baseline loss). Use the vocab-independent eval (legal/match) to
  compare models.
- **AMP (bf16)** gives ~1.5-2x speed but its autocast weight-cache adds memory, so
  bs64 OOMs near the 20 GB edge → use **bs32-48** with AMP.
- **VRAM ceiling:** 155M @ bs32 = 18.1 GB (safe, ~2 GB headroom) is the max useful;
  224M @ 20.2 GB is at the edge = crash risk on long runs.
- **Driver install gotcha:** Linode's plain Ubuntu image had no GPU driver; the
  apt prebuilt-module metapackage left the module in `rc` state. Fix that worked:
  `apt install nvidia-dkms-595-server-open + linux-headers-$(uname -r)` → DKMS
  builds `nvidia.ko` for the running kernel → `modprobe nvidia` → `nvidia-smi`.
- **Long runs:** launch detached (`nohup ... > log 2>&1 &`), use `python -u` for
  live logs, best-val checkpointing so a crash/VPN-drop loses nothing.

---

## 5. Next steps (decide at start of next session)

### Option B — self-play + search ("play to win") — RECOMMENDED next frontier
The real leap from *imitating* to *winning*. v2 becomes the **policy network**. Add:
1. **Value head** — a second output predicting "who's winning" from a position
   (a small change to `model.py`: a scalar head alongside the move head).
2. **Search** — MCTS-lite / shallow lookahead that uses policy (move priors) +
   value (position scores) to pick the move that *wins*, not just the likely one.
3. **Self-play RL** — model plays itself, games are scored by outcome, weights
   updated to favor winning moves (policy/value training loop).
Multi-session build; genuinely benefits from sustained GPU.
*(Historical note: this mirrors AlphaGo → AlphaZero — imitate first, then self-play.)*

### Scale Option A further (lower payoff)
Streaming parser → 10M+ games → ~200M model → val ~1.9. Prettier imitation, but
same ceiling (still doesn't try to win). Diminishing returns.

### Play-against-it (quick win, low effort, CPU-only)
A terminal board to play v2 yourself, or a Lichess-bot hookup. Satisfying; good
palate-cleanser; doesn't need the GPU.

---

## 6. Operational reminders

- **Weights never go in git** (too big). Use **GitHub Releases** (assets up to 2 GB).
  `.gitignore` excludes `**/checkpoints/*.pt`, `**/data/pgn/*`, `**/data/cache/`.
- **Infra boundary:** user provisions/destroys/reboots machines; Claude only
  trains/codes/monitors and *suggests* infra actions.
- **Delete the GitHub token** used for releases (user action); re-create for future
  releases (fine-grained, Contents: write).
- **Ladder recap:** A = imitate (done) → B = self-play + search (next) →
  C = position-eval head (folds into B as the value head).
