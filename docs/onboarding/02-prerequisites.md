# 02 — Prerequisites

Everything you need installed before you can train. Each item below has a verification command — run it and confirm the expected output before moving on.

## Hardware

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| RAM | 8 GB | 16 GB+ | Determines max `batch_size` |
| Disk | 2 GB free | 5 GB free | PyTorch wheels + checkpoints + cache |
| GPU | Optional | Apple M-series / NVIDIA / Colab T4 | CPU works but is ~10x slower |

If you have none of those, **Google Colab gives you a free NVIDIA T4 GPU** — see [`07-glossary.md → Colab`](07-glossary.md) and the Colab walkthrough in [Part 5](../05-putting-it-together.md#google-colab).

## Operating system

Tested on:

- macOS 13+ (Apple Silicon and Intel)
- Linux (Ubuntu 22.04+, Debian, Arch)
- Windows 10/11 (PowerShell, Git Bash, or WSL2)

Anything that runs Python 3.12 and PyTorch 2.8 will work.

## Software

### Python 3.12 or newer

This is a hard requirement — `pyproject.toml` pins `requires-python = ">=3.12"`. Older Pythons will fail at `uv sync`.

**Verify:**

```bash
python --version
# Expected: Python 3.12.x or higher
```

If you see anything older:

| Platform | Install command |
|----------|-----------------|
| macOS (Homebrew) | `brew install python@3.12` |
| Ubuntu / Debian | `sudo apt install python3.12 python3.12-venv` |
| Windows | Download from [python.org](https://www.python.org/downloads/) and tick *Add to PATH* |
| Anywhere | `uv python install 3.12` (after installing `uv` below) |

### uv (recommended) or pip

`uv` is a fast Python package manager from Astral. It handles Python versions, virtual envs, and dependency installation in one tool.

**Install:**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Verify:**

```bash
uv --version
# Expected: uv 0.x.x or higher
```

If you cannot install `uv`, plain `pip` works:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate              # Windows PowerShell
pip install "torch>=2.8.0" "numpy>=2.0.2" "tqdm>=4.67.3" "tiktoken>=0.12.0" "datasets>=4.5.0" "huggingface-hub>=1.8.0"
```

### Git

Needed to clone the repo.

**Verify:**

```bash
git --version
# Expected: git version 2.x.y
```

## Dependencies installed by `uv sync`

`uv sync` reads `pyproject.toml` + `uv.lock` and installs everything below into a `.venv` for you. You do not install these manually unless you are using pip without uv.

| Package | Why it's there | Pinned to |
|---------|---------------|-----------|
| **torch** | The deep-learning framework. All tensors, modules, optimizers, autograd. | `>=2.8.0` |
| **numpy** | Used for ad-hoc array operations and Colab download helpers. | `>=2.0.2` |
| **tqdm** | Progress bars in the training loop. | `>=4.67.3` |
| **tiktoken** | OpenAI's BPE tokenizer. *Optional* — only needed if you experiment with BPE in the competition. | `>=0.12.0` |
| **datasets** | HuggingFace dataset loader. Optional — useful if you swap Shakespeare for TinyStories or another corpus. | `>=4.5.0` |
| **huggingface-hub** | Pull dependency of `datasets`. | `>=1.8.0` |

The full pinned tree lives in [`uv.lock`](../../uv.lock).

## What you do NOT need

- **CUDA toolkit installed system-wide.** PyTorch wheels bundle their own CUDA runtime. If you have an NVIDIA GPU and recent drivers, it just works.
- **A HuggingFace account.** `datasets` and `huggingface-hub` are listed but the workshop never reaches for them; Shakespeare is shipped in the repo.
- **An OpenAI API key.** No external APIs are called.
- **Jupyter.** You can use one if you like (especially on Colab) but everything also runs from a shell.

## Verifying you have a working environment

Run all of these. Each should print the expected line.

```bash
python --version
# Python 3.12.x

uv --version
# uv 0.x.x

git --version
# git version 2.x.y
```

After you `uv sync` (covered in `03-setup.md`), you will run one more check:

```python
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available(), torch.cuda.is_available())"
# Example outputs:
#   2.8.0 True False    ← Apple Silicon Mac
#   2.8.0 False True    ← Linux/Windows with NVIDIA GPU
#   2.8.0 False False   ← CPU only (still fine, just slower)
```

If `torch` does not import, the `.venv` was not activated or `uv sync` failed. See [`08-troubleshooting.md`](08-troubleshooting.md).

## Next

[03-setup.md →](03-setup.md)
