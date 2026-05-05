# 03 — Setup

This walks you from a fresh checkout to a working virtual environment with all dependencies installed and a confirmed device (CPU, MPS, or CUDA). You should be able to do this in under five minutes on a decent connection.

## 1. Clone the repo

```bash
git clone <your-fork-or-the-upstream-url>.git llm-from-scratch
cd llm-from-scratch
```

After cloning you should see this layout:

```bash
ls
# data  docs  pyproject.toml  README.md  uv.lock  .python-version  .gitignore
```

There is no Python source code — that is correct. See [`01-what-is-this.md → Why the repo looks empty`](01-what-is-this.md#why-the-repo-looks-empty).

## 2. Install dependencies with `uv` (recommended)

```bash
uv sync
```

This:

1. Reads `.python-version` and ensures Python 3.12 is available (downloading it if not).
2. Reads `pyproject.toml` + `uv.lock`.
3. Creates a `.venv/` directory in the project root.
4. Installs `torch`, `numpy`, `tqdm`, `tiktoken`, `datasets`, `huggingface-hub`, and their transitive deps.

Expected output ends with something like:

```
Resolved 27 packages in 0.5s
Installed 27 packages in 12.3s
```

Total disk usage of the resulting `.venv/` is roughly 1.3-1.8 GB (PyTorch is large).

### If you do not have `uv`

```bash
python3.12 -m venv .venv
source .venv/bin/activate                 # macOS / Linux
.venv\Scripts\Activate.ps1                # Windows PowerShell
pip install "torch>=2.8.0" "numpy>=2.0.2" "tqdm>=4.67.3" "tiktoken>=0.12.0" "datasets>=4.5.0" "huggingface-hub>=1.8.0"
```

## 3. Activate the environment (only when not using `uv run`)

`uv run <cmd>` activates the `.venv` automatically. If you prefer to type bare `python` and `pip`, activate the env explicitly:

```bash
source .venv/bin/activate                 # macOS / Linux
.venv\Scripts\Activate.ps1                # Windows PowerShell
.venv/Scripts/activate                    # Windows Git Bash / MSYS2
```

After activation, `which python` (Linux/macOS) or `where python` (Windows) should point inside `.venv/`.

The rest of these docs use `uv run python ...` so they work whether or not you activated the env. Both styles are fine.

## 4. Verify PyTorch and your device

```bash
uv run python -c "import torch; print('torch', torch.__version__); print('mps', torch.backends.mps.is_available()); print('cuda', torch.cuda.is_available())"
```

Expected output:

| Hardware | Output |
|----------|--------|
| Apple Silicon Mac (M1/M2/M3/M4) | `torch 2.8.0` / `mps True` / `cuda False` |
| Linux/Windows with NVIDIA GPU | `torch 2.8.0` / `mps False` / `cuda True` |
| CPU only | `torch 2.8.0` / `mps False` / `cuda False` |

CPU-only is fine — it just means training the Medium config will take ~2-4 hours instead of ~45 minutes. You can still run the **Tiny** config (~5 min on CPU). See [`06-configuration.md → Model size presets`](06-configuration.md#model-size-presets).

## 5. Confirm the dataset is present

```bash
ls -lh data/shakespeare.txt
# -rw-r--r-- 1 you you 1.1M ... data/shakespeare.txt

uv run python -c "t = open('data/shakespeare.txt').read(); print(f'{len(t):,} chars, {len(set(t))} unique')"
# 1,115,394 chars, 65 unique
```

Both numbers must match exactly:

- **1,115,394 characters** — the full Tiny Shakespeare corpus.
- **65 unique characters** — fixes `vocab_size = 65` for the rest of the workshop.

If either differs, the file is corrupted or truncated. Re-clone the repo.

## 6. Create your scratchpad

```bash
mkdir -p scratchpad
cd scratchpad
```

This is where you will write `model.py`, `train.py`, and `generate.py` over the next four chapters. The `scratchpad/` folder is in `.gitignore` so your in-progress code is never committed.

From inside `scratchpad/`, the path to the data is `../data/shakespeare.txt`.

## 7. Sanity check: forward pass on a tiny dummy model

To confirm PyTorch can build a model, allocate it on your device, and run a forward pass — without writing any workshop code yet — paste this into a file `scratchpad/_smoke.py`:

```python
import torch, torch.nn as nn

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print("device:", device)

m = nn.Sequential(nn.Embedding(65, 32), nn.Linear(32, 65)).to(device)
x = torch.randint(0, 65, (4, 16), device=device)         # batch=4, seq=16
y = m(x)
print("input shape :", x.shape)                           # torch.Size([4, 16])
print("output shape:", y.shape)                           # torch.Size([4, 16, 65])
print("OK")
```

```bash
uv run python _smoke.py
# device: mps        (or cuda / cpu)
# input shape : torch.Size([4, 16])
# output shape: torch.Size([4, 16, 65])
# OK
```

If this prints `OK`, your environment is fully set up. Delete `_smoke.py` and move on.

## 8. Optional: enable GPU on Google Colab

If you do not have a local GPU and prefer Colab:

1. Open a new notebook at <https://colab.research.google.com/>.
2. **Runtime → Change runtime type → T4 GPU** (free tier).
3. In a cell, install the same dependencies:
   ```python
   !pip install -q "torch>=2.8.0" "numpy>=2.0.2" "tqdm>=4.67.3" "tiktoken>=0.12.0"
   ```
4. Download the Shakespeare data:
   ```python
   import os, urllib.request
   os.makedirs("data", exist_ok=True)
   urllib.request.urlretrieve(
       "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
       "data/shakespeare.txt",
   )
   ```
5. Verify GPU access:
   ```python
   import torch; torch.cuda.is_available(), torch.cuda.get_device_name(0)
   # (True, 'Tesla T4')
   ```
6. From here on, when the docs say `python train.py`, write the same code in a Colab cell or upload `train.py` and run `!python train.py`. Use `data_path="data/shakespeare.txt"` (no `..` prefix).

## Common setup pitfalls

| Symptom | Fix |
|---------|-----|
| `Python 3.11 is not >= 3.12` | Install Python 3.12: `uv python install 3.12` then re-run `uv sync`. |
| `error: failed to fetch ... torch` | Network/firewall issue. Retry, or set `UV_INDEX_URL` if behind a proxy. |
| `torch.backends.mps.is_available()` is `False` on Apple Silicon | You're on macOS < 13 or an Intel Mac. CPU still works. |
| `import torch` raises `ModuleNotFoundError` | The `.venv` was not activated. Use `uv run python ...` or `source .venv/bin/activate`. |
| Disk full mid-`uv sync` | PyTorch wheels are ~700 MB. Free at least 2 GB. |

For more, see [`08-troubleshooting.md`](08-troubleshooting.md).

## Next

[04-test-run.md →](04-test-run.md) — your first full end-to-end run.
