# 08 — Troubleshooting

Real failures, with the symptom you'll see and the fix that works. Organized by phase: setup → smoke test → training → generation.

If your problem isn't here, check the [`07-glossary.md`](07-glossary.md) for unfamiliar terms, then re-read the relevant Part doc.

---

## Setup

### `python: command not found`

**Symptom**

```bash
python --version
zsh: command not found: python
```

**Cause** — Some systems only ship `python3`, not `python`.

**Fix** — Use `python3` for the version check, then let `uv` create the env:

```bash
python3 --version
uv sync
uv run python -c "print('hello')"
```

`uv run` always picks the right interpreter from `.venv/`.

---

### `error: Python 3.11.5 does not satisfy the requirement Python>=3.12`

**Symptom** — `uv sync` refuses to proceed.

**Fix**

```bash
uv python install 3.12
uv sync                # uv now uses Python 3.12 from its managed cache
```

---

### `ModuleNotFoundError: No module named 'torch'`

**Symptom**

```
Traceback (most recent call last):
  File "...", line 1, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

**Cause** — You ran `python` outside the virtual environment.

**Fix** — Either activate the env or use `uv run`:

```bash
# Option A: activate
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1          # Windows PowerShell
python -c "import torch"

# Option B: don't activate, use uv run
uv run python -c "import torch"
```

If both fail, `uv sync` did not finish. Re-run it and watch for errors.

---

### `uv sync` hangs or fails downloading torch

**Symptom** — Downloads stall or you get `error: failed to fetch ... torch ... ConnectionError`.

**Fix** — Network/firewall. Try:

```bash
# pick a closer mirror
UV_INDEX_URL=https://download.pytorch.org/whl/cpu uv sync
# or use a system proxy
HTTPS_PROXY=http://your-proxy:port uv sync
```

PyTorch wheels are 600-900 MB; on slow connections expect 5-15 minutes.

---

### `torch.backends.mps.is_available()` returns `False` on a Mac

**Symptom** — You expected to use MPS but `get_device()` falls back to CPU.

**Causes**

- macOS < 13 (MPS requires macOS 13 Ventura or later).
- Intel Mac (no MPS support).
- An old `torch` (`<2.0`) that didn't support MPS yet — but `pyproject.toml` pins `>=2.8`, so this is unlikely.

**Fix** — Update macOS, or accept CPU. CPU still works, just slower. The Tiny preset trains in ~10 minutes on CPU.

---

## Smoke test

### Initial loss is wildly different from `~4.17`

**Symptom**

```
Step     0 | val loss: 6.34       ← way too high
```

or

```
Step     0 | val loss: 1.20       ← way too low
```

**Cause**

- **Too high** — wrong tokenizer (e.g., you accidentally used BPE), or the data file was corrupted/empty.
- **Too low** — you have a bug that lets `y` leak into `x`, or you're computing loss on already-shifted targets.

**Fix**

```python
import torch
from train import load_data
get_train, get_val, vocab_size, stoi, itos = load_data(
    "../data/shakespeare.txt", 256, 64, torch.device("cpu")
)
print("vocab_size:", vocab_size)                     # must be 65
x, y = get_train()
print("x shape:", x.shape, "y shape:", y.shape)      # (64, 256), (64, 256)
print("y == x shifted by 1?", (y[:, :-1] == x[:, 1:]).all().item())   # True
```

Each line must produce the expected output. If `vocab_size != 65`, your data file is wrong. If `y` is not `x` shifted, your loader has a bug.

---

### Loss not decreasing

**Symptom** — Loss stays near 4.17 across many steps.

**Causes (most common first)**

1. **Learning rate too low** — try `max_lr=1e-3` (the default).
2. **Forgot `optimizer.zero_grad()`** — gradients accumulate, eventually dominating updates.
3. **Forgot `optimizer.step()`** — gradients computed, never applied.
4. **`model.eval()` left on** — dropout etc. shouldn't matter here, but worth checking.
5. **Data not on the same device as the model** — `x.device != model.device` crashes with a clear error, but mismatched but compatible types can silently waste compute.

**Fix** — Walk through the loop carefully:

```python
optimizer.zero_grad()        # 1
loss.backward()              # 2
optimizer.step()             # 3
```

In that order, every step. Confirm with a one-step diagnostic:

```python
import torch
from model import GPT, GPTConfig
m = GPT(GPTConfig())
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randint(0, 65, (4, 32))
y = torch.randint(0, 65, (4, 32))
for step in range(20):
    _, loss = m(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(step, loss.item())
# Loss should drop sharply within 20 steps when overfitting a fixed batch.
```

If this trivial overfit-one-batch test doesn't drop loss, the bug is in `model.py` or how you wired the loop, not in the data.

---

### Loss spikes / NaNs

**Symptom** — Loss is stable, then suddenly explodes to thousands or `nan`.

**Causes**

- Learning rate too high.
- Gradient clipping not applied.
- Bad data (a pathological character or batch).

**Fix**

1. Confirm `clip_grad_norm_` is in your loop:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
2. Lower `max_lr` (try `3e-4`).
3. Increase `warmup_steps` (try 200-500).
4. If only one batch causes NaN, it's almost always `lr` — the data is fine.

---

### `RuntimeError: shape mismatch`

**Symptom** — Common shapes:

```
RuntimeError: shape '[64, 256, 6, 64]' is invalid for input of size 4915200
```

**Cause** — `n_embd` is not divisible by `n_head`, or you passed a sequence longer than `block_size`.

**Fix** — Confirm `n_embd % n_head == 0` and `idx.shape[-1] <= block_size`. The `assert` at the top of `CausalSelfAttention.__init__` catches the first; you have to add `idx_cond = idx[:, -block_size:]` yourself in generation to handle the second.

---

### `OutOfMemoryError` / `MPSBackend allocation failed`

**Symptom**

```
RuntimeError: MPS backend out of memory (MPS allocated: 7.81 GB, ...)
```

or on CUDA:

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Fix — in order of preference:**

1. Reduce `batch_size` (most direct lever):
   ```python
   train(batch_size=32)        # default 64
   train(batch_size=16)
   ```
2. Reduce `block_size` (attention is `O(T²)` in memory):
   ```python
   train(block_size=128)
   ```
3. Reduce `n_embd` and `n_layer` — train a smaller model.
4. If on CUDA, free other processes: `nvidia-smi` to see who's using VRAM.

A sample memory-vs-batch table on Apple M3 Pro (Medium config, `block_size=256`, fp32):

| `batch_size` | Peak working set |
|--------------:|-----------------:|
| 16 | ~150 MB |
| 32 | ~250 MB |
| 64 | ~400 MB |
| 128 | ~750 MB |

---

### Training is *very* slow

**Symptom** — `tqdm` reports 0.05 it/s or worse.

**Causes**

- You're on CPU. Expected if no GPU; mitigate with the Tiny preset.
- `block_size` is huge (e.g., 1024). Attention is quadratic.
- You forgot `.to(device)` somewhere — the model is on CPU even though MPS/CUDA is available. Verify with:
  ```python
  print(next(model.parameters()).device)
  ```

**Fix** — Use the smallest config that demonstrates what you want. For debugging, Tiny + 200 steps trains in seconds.

---

## Generation

### `KeyError: ' '` or similar in `generate()`

**Symptom** — You passed a prompt containing a character that wasn't in the training set. `stoi[c]` raises `KeyError`.

**Fix** — Two options:

1. Strip unknown characters (already done in our `generate()`):
   ```python
   tokens = [stoi[c] for c in prompt if c in stoi]
   ```
2. Use a prompt with only ASCII letters, spaces, and basic punctuation. The Tiny Shakespeare alphabet is 65 chars: letters (upper and lower), digits, common punctuation, and `\n`.

---

### Generated text is just the same character repeated

**Symptom**

```
To be or not eeeeeeeeeeeeeeeeeeeeeeeee
```

**Causes**

- `temperature` is too low (approaches greedy).
- The model is barely trained — at val loss > 3.0 it produces low-entropy garbage.

**Fix** — Train longer or raise temperature:

```bash
uv run python generate.py checkpoint.pt --temperature 1.0 --top_k 30 --seed 0
```

If even `temperature=1.0` produces single-character spam, the model isn't trained. Check val loss.

---

### Generated text trails off into noise

**Symptom** — First 30 chars look fine, then it descends into garbage.

**Causes**

- Temperature too high.
- Top-k is `0` (disabled) and the model occasionally samples a very unlikely token, derailing the sequence.

**Fix**

```bash
uv run python generate.py checkpoint.pt --temperature 0.7 --top_k 20
```

Lower temperature, smaller top-k. The defaults (0.8 / 40) are tuned for our 10M model at val loss ~1.6.

---

### Generation is reproducible but produces different output on different machines

**Symptom** — Same `--seed`, same checkpoint, but two laptops produce different text.

**Causes**

- `scaled_dot_product_attention` may pick different fast paths on MPS vs. CUDA vs. CPU. The math is "the same" but floating-point order differs.
- Different PyTorch versions may have non-bitwise-identical kernels.

**Fix** — This is expected. Bitwise-reproducibility across hardware is not a goal of the workshop; the same machine + seed will reproduce. For the competition, the verifying machine reproduces on its own hardware against your checkpoint.

---

### `RuntimeError: Error(s) in loading state_dict for GPT: ... unexpected key(s)`

**Symptom** — `model.load_state_dict(ckpt["model_state_dict"])` fails complaining about keys.

**Causes** — The checkpoint was trained with a different model class than you're loading into (different `n_layer`, naming, or weight tying).

**Fix** — Always reconstruct the model from the *same* config the checkpoint was saved with:

```python
ck = torch.load("checkpoint.pt", weights_only=False)
model = GPT(ck["config"])             # use the saved config
model.load_state_dict(ck["model_state_dict"])
```

Our `generate.py` already does this.

---

## Data

### `FileNotFoundError: ../data/shakespeare.txt`

**Symptom** — `train.py` cannot find the dataset.

**Cause** — You're running from somewhere other than `scratchpad/`.

**Fix** — Either `cd scratchpad` first, or pass an absolute path:

```python
train(data_path="/abs/path/to/llm-from-scratch/data/shakespeare.txt")
```

---

### Dataset has wrong char count or wrong vocab size

**Symptom**

```
Dataset: 1,000,000 chars, vocab size: 64
```

instead of the expected `1,115,394 chars, vocab size: 65`.

**Cause** — The file was edited, truncated, or has line-ending mangling (CRLF vs LF on Windows).

**Fix** — Re-clone the repo, or re-download the canonical file:

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/shakespeare.txt
wc -c data/shakespeare.txt          # should be 1115394
```

---

## Validation / overfitting

### Val loss is increasing while train loss decreases

**Symptom**

```
Step  1500 | val loss: 1.57
Step  2500 | val loss: 1.71
Step  3500 | val loss: 2.34       ← getting worse
```

This is **expected**, not a bug. It's the central lesson of [Part 3](../03-training-loop.md#overfitting-train-loss-vs-val-loss). The model has more capacity than the data. Use the checkpoint with the lowest val loss.

**If you want to delay it:**

- Reduce model size (`n_layer=4, n_head=4, n_embd=256`).
- Increase data — Shakespeare is 1 MB; TinyStories is ~470 MB.
- Add dropout (insert `nn.Dropout(0.1)` in `Block.forward`).
- Stop early — `max_steps=2000` already captures most of the signal.

---

### Val loss is *much* higher than train loss from the start

**Symptom**

```
train loss: 1.20    val loss: 3.50
```

**Cause** — The val split is unrepresentative. With Shakespeare's 90/10 contiguous split, the val set is the last 10% of the file, which has different speakers/scenes than the early text. Some divergence is normal; an extreme gap means severe overfitting.

**Fix** — Confirm by reducing model size or increasing regularization. The split itself is fine for the workshop's pedagogical purpose.

---

## Checkpointing

### Checkpoints aren't being saved

**Symptom** — `ls scratchpad/*.pt` is empty after a long run.

**Cause** — `ckpt_every` is larger than `max_steps`, or you're running from a directory you can't write to.

**Fix** — Confirm `out_dir` is writable; lower `ckpt_every` to 500 for short runs.

---

### `weights_only=False` warning on `torch.load`

**Symptom**

```
UserWarning: You are using torch.load with weights_only=False...
```

**Cause** — Newer PyTorch defaults to `weights_only=True` for security. Our checkpoints contain a `GPTConfig` dataclass, which `weights_only=True` won't deserialize.

**Fix** — Explicitly pass `weights_only=False` when loading our own checkpoints (we already do):

```python
ck = torch.load(path, weights_only=False)
```

This is safe **only because** you trust the checkpoint file. Never load `weights_only=False` from an untrusted source.

---

## Sanity checklist when something is "just weird"

When debugging, run through this in order. Each step takes seconds.

```python
# 1. Imports work
from model import GPT, GPTConfig
from train import load_data, get_device

# 2. Data loads with right shape
get_train, get_val, V, stoi, itos = load_data("../data/shakespeare.txt", 256, 64, get_device())
assert V == 65
x, y = get_train(); assert x.shape == (64, 256) and y.shape == (64, 256)

# 3. Model builds with right param count
m = GPT(GPTConfig()).to(get_device())
n = sum(p.numel() for p in m.parameters()); assert 10_500_000 < n < 11_000_000

# 4. Forward pass produces right shape
logits, loss = m(x, y)
assert logits.shape == (64, 256, 65)
assert 3.5 < loss.item() < 4.5     # near ln(65) for a fresh model

# 5. Backward pass works
loss.backward()
assert all(p.grad is not None for p in m.parameters() if p.requires_grad)

print("all checks pass")
```

If all five pass and training is still broken, the problem is in the loop wiring (zero_grad, step), not in any of the components.

## Next

- The full FAQ from the live workshop: [09-faq.md](09-faq.md)
- Back to the start: [README.md](README.md)
