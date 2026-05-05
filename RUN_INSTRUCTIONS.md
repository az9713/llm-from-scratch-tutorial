# How to Train the LLM From Scratch — Step-by-Step

A consolidated guide for running the workshop in this repo end-to-end on Windows + PowerShell. Built from the repo's docs and verified against the actual files in `data/`, `docs/`, and `pyproject.toml`.

---

## What this repo actually is

This repo is a **workshop**, not a finished program. There are no `model.py`, `train.py`, or `generate.py` shipped — the six docs in `docs/` walk you through writing them yourself in a local `scratchpad/` folder.

Repo layout:

```
.
├── README.md            workshop overview
├── pyproject.toml       Python deps (torch, numpy, tqdm, tiktoken, datasets, hf-hub)
├── uv.lock              pinned dep versions
├── .python-version      → 3.12
├── data/shakespeare.txt ~1MB training corpus (already in repo, no download needed)
└── docs/                01-tokenization … 06-competition (the workshop)
```

Prereqs: Python ≥ 3.12, Windows/Mac/Linux. Training picks CUDA → MPS → CPU automatically.

---

## Step 1 — Install `uv` (one-time)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Step 2 — Install dependencies

From the repo root:

```powershell
uv sync
```

This creates `.venv\` and installs torch, numpy, tqdm, tiktoken, datasets, huggingface-hub.

## Step 3 — Make the scratchpad

```powershell
mkdir scratchpad
cd scratchpad
```

`.gitignore` excludes `**/scratchpad/`, so anything you write there stays untracked.

## Step 4 — Write the three Python files

You don't have to invent any of it — the docs give you the code, line by line. Each part tells you exactly what to paste into which file.

| Doc | File you build | What it contains |
|---|---|---|
| `docs/01-tokenization.md` | (background only) | Explains the `stoi`/`itos` char tokenizer used by `train.py` — no file to write |
| `docs/02-the-transformer.md` | `model.py` | `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` |
| `docs/03-training-loop.md` | `train.py` | `load_data`, `get_device`, `get_lr`, the `train(...)` function, `__main__` call |
| `docs/04-text-generation.md` | `generate.py` | `generate(...)` function (greedy → temperature → top-k → final) |
| `docs/05-putting-it-together.md` | — | How to run them |
| `docs/06-competition.md` | — | Optional: scale up with your own dataset |

Every code block in those docs is meant to be appended to the file the doc names at the top, in order. So `model.py` ends up as the concatenation of every Python block in `docs/02`, in order. Same pattern for `train.py` (doc 03) and `generate.py` (doc 04).

**Workflow:**

1. Open `docs/02-the-transformer.md`. Create `scratchpad/model.py`. Copy each `python` block into the file as you read.
2. Open `docs/03-training-loop.md`. Create `scratchpad/train.py`. Same drill. The doc tells you to **skip the `generate(...)` calls** in the training loop until after Part 4.
3. Open `docs/04-text-generation.md`. Create `scratchpad/generate.py`. Same drill. Then go back to `train.py` and add `from generate import generate` at the top so the in-training samples work.

You can copy-paste rather than retype — the read-along prose explains *why* each block exists, which is the actual point of the workshop.

## Step 5 — Verify the wiring before training

Inside `scratchpad/`, your `train.py` needs these imports near the top, or it won't see the other two files:

```python
from model import GPT, GPTConfig
from generate import generate
```

`train.py` also needs a `__main__` block. From doc 05:

```python
if __name__ == "__main__":
    data_path = "../data/shakespeare.txt"
    model, stoi, itos = train(data_path)
```

The path `../data/shakespeare.txt` is correct **only when running from `scratchpad/`** inside the repo.

## Step 6 — Smoke test (30 seconds, catches typos cheaply)

Before committing to a 45-minute run, do a tiny run first. Temporarily edit the `train(...)` call in `train.py` to use the Tiny config with few steps:

```python
model, stoi, itos = train(data_path,
                          n_layer=2, n_head=2, n_embd=128,
                          max_steps=50)
```

Then run:

```powershell
uv run python train.py
```

If it prints:

- `Using device: ...`
- `Dataset: 1,115,394 chars, vocab size: 65`
- a model param count
- a tqdm progress bar with decreasing loss

…the pipeline works. If it crashes, the traceback names the file and line.

## Step 7 — Real training run

Restore the defaults (or pick a config that fits your patience) and run:

```powershell
uv run python train.py
```

Pick a config based on your hardware:

| Config | Params | `max_steps` | CPU time | NVIDIA GPU |
|---|---|---|---|---|
| Tiny (`2/2/128`) | ~0.5M | 2000 | ~10 min | < 2 min |
| Small (`4/4/256`) | ~4M | 5000 | ~1–2 hr | ~5 min |
| Medium default (`6/6/384`) | ~10M | 5000 | many hours | ~10–15 min |

What you'll see:

- val loss lines and generated samples every 100 steps
- `checkpoint_<step>.pt` files written every 1000 steps
- `checkpoint_final.pt` and `loss_log.json` at the end

What loss numbers mean (character-level, vocab=65), from doc 03:

- ~4.2 — random (untrained)
- ~3.3 — learned letter frequencies
- ~2.5 — learned common bigrams
- ~1.5–2.0 — recognizable words and Shakespeare structure
- ~1.0–1.2 — good quality verse with character names
- < 1.0 — likely memorizing

Best model is usually around step 1500–2000 (lowest val loss), not step 5000. After that, val loss starts rising — overfitting. That's expected with 10M params on ~1MB of text.

## Step 8 — Generate text from a checkpoint

When training finishes (or once any `checkpoint_*.pt` exists):

```powershell
uv run python generate.py checkpoint_final.pt
```

Or any earlier checkpoint, e.g. `checkpoint_2000.pt`. The final `generate.py` from doc 04 reads the path from `sys.argv[1]`, loads the model + `stoi`/`itos` from the checkpoint, and prints text from a few prompts.

## Step 9 (optional) — Plot the loss curve

```powershell
uv pip install matplotlib
```

```python
import json, matplotlib.pyplot as plt

with open("loss_log.json") as f:
    log = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(log["steps"], log["train"], alpha=0.3, label="train")
plt.xlabel("Step"); plt.ylabel("Loss"); plt.legend()
plt.savefig("loss_curve.png"); plt.show()
```

---

## Colab fallback

If local training is too slow (CPU-only Windows on the Medium config will be painful), the README and `docs/05-putting-it-together.md` cover Colab:

1. Open a notebook at colab.research.google.com.
2. Runtime → Change runtime type → GPU (T4).
3. First cell: `!pip install -q torch numpy tqdm tiktoken`.
4. Upload `data/shakespeare.txt` to the Colab files panel.
5. Upload your `model.py`, `train.py`, `generate.py`, change the data path to `"data/shakespeare.txt"`, and run `!python train.py`.

---

## Common gotchas

- **`ModuleNotFoundError: No module named 'model'`** — `train.py`, `model.py`, `generate.py` must be in the same folder; run from that folder.
- **`FileNotFoundError: ../data/shakespeare.txt`** — you ran from somewhere other than `scratchpad/`. Either `cd scratchpad` first, or use an absolute path.
- **`torch.backends.mps`-related error on Windows** — fine, MPS is Mac-only; `get_device()` falls through to CUDA or CPU automatically.
- **Training loss stuck at ~4.2** — model is untrained / LR effectively 0. Check the warmup math and that the optimizer step is actually running.
- **Loss explodes to NaN** — gradient clip not wired in, or LR too high. The code in doc 03 includes `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
- **`from generate import generate` fails** — you forgot to add the import after writing `generate.py` in step 4.

---

## TL;DR command sequence

```powershell
# one-time
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
mkdir scratchpad

# author the three files in scratchpad/ from docs 02, 03, 04 (copy-paste each python block)

cd scratchpad

# smoke test (Tiny config, 50 steps)
uv run python train.py

# real run (after restoring defaults or choosing a config)
uv run python train.py

# generate
uv run python generate.py checkpoint_final.pt
```
