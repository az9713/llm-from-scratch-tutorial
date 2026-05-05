# 06 — Configuration Reference

Every parameter you can pass to `GPTConfig` and `train()`, with what it does, what range is sensible, and what breaks at the extremes.

## `GPTConfig`

Defined in `model.py`. These set the model architecture and are baked into checkpoints, so you cannot change them after training without retraining.

| Field | Default | Type | Range | What it controls |
|-------|--------:|------|-------|------------------|
| `vocab_size` | `65` | `int` | `2 .. ~50_000` | Number of distinct tokens. Must match the tokenizer. For Shakespeare char-level it's 65. |
| `block_size` | `256` | `int` | `8 .. ~4096` | Maximum sequence length. The model's "memory". |
| `n_layer` | `6` | `int` | `1 .. ~24` | Number of transformer blocks. Depth of the model. |
| `n_head` | `6` | `int` | divides `n_embd` | Number of parallel attention heads. |
| `n_embd` | `384` | `int` | `64 .. ~1024` | Embedding dimension; "width" of the model. Must be divisible by `n_head`. |

### Tradeoffs

**`vocab_size`.** Shakespeare has exactly 65 unique characters, so for the workshop this is fixed. If you switch to BPE or a larger dataset, this can explode to 32k-100k. Larger vocab → embedding matrix dominates the parameter count → you need much more training data to fill it.

**`block_size`.** The attention layer's compute is `O(block_size² × n_embd)`. Doubling `block_size` quadruples attention cost. For Shakespeare-scale data, 256 is plenty (most poems and dialogue are shorter than this). Push to 512 or 1024 for a harder competition entry — but expect to lower `batch_size` to fit memory.

**`n_layer`.** Deeper models can chain more reasoning. Returns diminish on small data. For ~1 MB datasets, 4-6 is the sweet spot. Beyond ~12 layers on this much data, the only thing you gain is overfitting speed.

**`n_head`.** More heads → finer-grained attention patterns. Keep `head_dim = n_embd / n_head` between 32 and 128. For `n_embd=384`, `n_head=6` gives `head_dim=64` (matches GPT-2 small). `n_head=4` gives `head_dim=96` (also fine). `n_head=12` gives `head_dim=32` which works but is on the small side.

**`n_embd`.** The single biggest lever. Most parameters are `O(n_embd²)` per layer, so doubling `n_embd` roughly quadruples params. Doubling layers only doubles params.

### Model size presets

| Name | `n_layer` | `n_head` | `n_embd` | Params | Train time (M3 Pro, 5k steps) | When to use |
|------|----------:|---------:|---------:|-------:|-------------------------------|-------------|
| **Tiny** | 2 | 2 | 128 | ~0.5 M | ~5 min | Smoke test, debugging, CPU-only laptops |
| **Small** | 4 | 4 | 256 | ~4 M | ~20 min | Quick iteration, low-memory machines |
| **Medium** (default) | 6 | 6 | 384 | ~10 M | ~45 min | The standard workshop config |
| **Large** | 8 | 8 | 512 | ~25 M | ~3 h | Only if you switched to a bigger dataset |
| **GPT-2 Small** | 12 | 12 | 768 | ~85 M | (won't fit on most laptops) | For reference only |

Use Tiny first to confirm everything works, then move to Medium for real results.

## `train()` parameters

Defined in `train.py`. These set how training runs but are not baked into the model architecture, so you can change them between runs.

| Parameter | Default | Range | What it controls |
|-----------|---------|-------|------------------|
| `data_path` | `"../data/shakespeare.txt"` | any path | Source corpus. Must be UTF-8 text. |
| `max_steps` | `5000` | `100 .. 100_000` | Total optimizer steps. |
| `batch_size` | `64` | `1 .. ~256` | Sequences per training step. Bigger = smoother gradients, more memory. |
| `n_layer` | `6` | see `GPTConfig` | |
| `n_head` | `6` | see `GPTConfig` | |
| `n_embd` | `384` | see `GPTConfig` | |
| `block_size` | `256` | see `GPTConfig` | |
| `max_lr` | `1e-3` | `1e-5 .. 3e-3` | Peak learning rate after warmup. |
| `warmup_steps` | `100` | `0 .. ~1000` | Steps to ramp LR from ~0 to `max_lr`. |
| `sample_every` | `100` | `≥ 1` | Steps between mid-training text samples. |
| `val_every` | `100` | `≥ 1` | Steps between validation-loss evaluations. |
| `ckpt_every` | `1000` | `≥ 100` | Steps between checkpoint saves. |
| `out_dir` | `"."` | any path | Where to write checkpoints and `loss_log.json`. |

### What to tune first

**If training feels too slow** → reduce `n_embd` and `n_layer`. Architecture matters more than step count.

**If loss plateaus too early** → bump `max_lr` (try `3e-3`). If that diverges, increase `warmup_steps`.

**If loss explodes (NaN, sudden jumps)** → lower `max_lr` (try `3e-4`). Check that `clip_grad_norm_` is in your loop.

**If you OOM (out of memory)** → halve `batch_size`. If still OOM, halve `block_size`. Last resort: reduce `n_embd`.

**If your model overfits too fast** → reduce `n_embd` and `n_layer`, or get more data.

## Generation parameters

These are only used at inference time (`generate.py` CLI flags or `generate(...)` function arguments).

| Flag | Default | Sensible range | What it controls |
|------|---------|----------------|------------------|
| `--prompt` | `"To be or not"` | any string | Initial text the model continues from. |
| `--max_new_tokens` | `200` | `1 .. ~2000` | How many tokens to generate after the prompt. |
| `--temperature` | `0.8` | `0.1 .. 2.0` | Logit scaling. Lower = more deterministic; higher = more random. |
| `--top_k` | `40` | `1 .. vocab_size`, or `0` to disable | Restricts sampling to the k most probable tokens. Cuts off the long tail of unlikely tokens. |
| `--seed` | unset | any int | RNG seed for reproducibility. |
| (positional) | — | path | Checkpoint file path. |

### How temperature interacts with top-k

| `temperature` | `top_k` | Effect |
|---------------|---------|--------|
| 0.0–0.2 | any | Approaches greedy — deterministic, repetitive, often loops on common bigrams. |
| 0.5–0.7 | 30–50 | Focused, mostly safe text. Best for technical or factual generation. |
| **0.8** | **40** | **Workshop default. Coherent but varied.** |
| 0.9–1.0 | 20–40 | Creative, occasional missteps. Best for poetry/competition entries. |
| 1.5+ | small (≤10) | High creativity but high error rate. Top-k is mandatory at this temperature or output deteriorates fast. |

### Reproducibility recipe

To produce the exact same output twice:

```bash
uv run python generate.py checkpoint_2000.pt \
    --prompt "To be or not" \
    --temperature 0.8 \
    --top_k 40 \
    --seed 42
```

The seed must be set **before** any `torch.multinomial` call. The CLI handles this for you. From Python:

```python
torch.manual_seed(42)
out = generate(model, "To be or not", stoi, itos, temperature=0.8, top_k=40)
```

## Hidden constants worth knowing

These live inside the code but are rarely changed. Knowing they exist helps when reading or modifying the source.

| Where | Value | Why |
|-------|-------|-----|
| `optimizer = AdamW(...)` | `weight_decay=0.01` | Light L2 regularization. GPT-2 uses 0.1 for large-scale runs. |
| `clip_grad_norm_(..., max_norm=1.0)` | 1.0 | Caps total gradient L2 norm. Standard transformer-training value. |
| `MLP.c_fc` | `4 * n_embd` | The standard "expansion factor" of 4 in transformer FFNs. |
| `train`/`val` split | 90% / 10% | First 90% of the file becomes train; last 10% becomes val. (No shuffling — Shakespeare is contiguous text.) |
| `min_lr = max_lr * 0.1` | 0.1 × peak | Cosine decay's floor. Setting it to 0 is fine but makes restarting harder. |
| Validation loop iterations | 20 | Number of random val batches averaged for each `val_loss` measurement. |
| Mid-training sample length | 100 chars | Short enough to not slow training, long enough to see progress. |
| GELU variant | `approximate="tanh"` | The fast approximation used in GPT-2. The exact `erf` version is slightly slower but indistinguishable in quality. |

To change any of these, edit `train.py` and `model.py` directly — they are not exposed as parameters.

## Cheat sheet: copy-pasteable configs

```python
# Tiny (smoke test)
train(max_steps=200, n_layer=2, n_head=2, n_embd=128,
      batch_size=16, sample_every=100, val_every=50, ckpt_every=200)

# Small (~4M params, ~20 min on M3)
train(max_steps=3000, n_layer=4, n_head=4, n_embd=256, batch_size=32)

# Medium (default, ~10M, ~45 min on M3)
train()

# Large (~25M, requires more data than Shakespeare to be useful)
train(max_steps=8000, n_layer=8, n_head=8, n_embd=512, batch_size=32)

# Long-context Medium (block_size=512, halve batch to compensate)
train(block_size=512, batch_size=32)

# Aggressive learning rate
train(max_lr=3e-3, warmup_steps=200)
```

## Next

- Look up unfamiliar terms: [07-glossary.md](07-glossary.md)
- Errors and fixes: [08-troubleshooting.md](08-troubleshooting.md)
