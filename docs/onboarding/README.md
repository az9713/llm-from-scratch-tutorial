# Onboarding

Welcome. This folder is the front door for new contributors to **Train Your Own LLM From Scratch**, a hands-on workshop where you write every piece of a GPT training pipeline yourself — tokenizer, model, training loop, and text generation — and run the whole thing on a laptop in under an hour.

If you have never opened this repo before, read the docs in this folder in order. By the end of `04-test-run.md`, you will have a working `~10M`-parameter GPT generating Shakespeare-style text on your machine.

## Read these in order

| # | Doc | What you get |
|---|-----|--------------|
| 1 | [01-what-is-this.md](01-what-is-this.md) | What this project is, who it's for, and why the `scratchpad/` is empty by design |
| 2 | [02-prerequisites.md](02-prerequisites.md) | Exact software, hardware, and verification commands before you touch the code |
| 3 | [03-setup.md](03-setup.md) | Clone, `uv sync`, environment sanity checks, Colab fallback |
| 4 | [04-test-run.md](04-test-run.md) | End-to-end: scaffold the three Python files, run a 5-minute smoke test, train the medium model, generate text |
| 5 | [05-architecture.md](05-architecture.md) | Deep dive into the full data flow with tensor shapes, parameter counts, and gradients |
| 6 | [06-configuration.md](06-configuration.md) | Every knob you can turn — vocabulary size, block size, n_layer, n_head, n_embd, batch_size, learning rate, schedules |
| 7 | [07-glossary.md](07-glossary.md) | Every term that appears in any doc, with a one-line definition and an example |
| 8 | [08-troubleshooting.md](08-troubleshooting.md) | Failure modes you will hit, with the fix |
| 9 | [09-faq.md](09-faq.md) | Questions asked at the live workshop, with the answers |

## Once you understand the structure

Switch to the original tutorial sequence (the "Parts"), each of which walks you through writing one file:

| Part | What you write | File |
|------|----------------|------|
| [Part 1: Tokenization](../01-tokenization.md) | character-level encoder/decoder | (lives inside `train.py`) |
| [Part 2: The Transformer](../02-the-transformer.md) | `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` | `model.py` |
| [Part 3: The Training Loop](../03-training-loop.md) | data loader, LR schedule, training loop, checkpointing | `train.py` |
| [Part 4: Text Generation](../04-text-generation.md) | autoregressive decoding, temperature, top-k | `generate.py` |
| [Part 5: Putting It Together](../05-putting-it-together.md) | run, plot losses, scale | (no new file) |
| [Part 6: Competition](../06-competition.md) | submit your best generated poem | (your model) |

## How to use these two folders together

The **Parts** (1-6) teach you the *concepts and code*. The **onboarding** folder (this folder) teaches you the *project mechanics* — how to set up, what to install, what to expect, and how to recover when something breaks.

If you already know PyTorch and just want to start, skip to `04-test-run.md`. If you have never trained a neural network, start at `01-what-is-this.md`.

## Where to find the source material

| Item | Location |
|------|----------|
| Project root README | [`../../README.md`](../../README.md) |
| Conceptual chapters | [`../01-tokenization.md`](../01-tokenization.md) through [`../06-competition.md`](../06-competition.md) |
| Training data | [`../../data/shakespeare.txt`](../../data/shakespeare.txt) (~1.1 MB) |
| Python project metadata | [`../../pyproject.toml`](../../pyproject.toml) |
| Pinned dependency versions | [`../../uv.lock`](../../uv.lock) |
| The empty scratchpad you will fill | `../../scratchpad/` (you create this; it is `.gitignore`d) |
