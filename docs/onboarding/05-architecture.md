# 05 — Architecture Deep Dive

This is the engineering view of the system — the same pipeline you ran in [`04-test-run.md`](04-test-run.md), now with every shape, every gradient path, and every parameter accounted for.

If you have already trained a model, this doc tells you *what just happened* and *why each piece is shaped the way it is*. If you haven't trained yet, read this anyway — it's a top-down map you can keep open while writing the code.

## System overview

```
                                 INFERENCE
                          ┌─────────────────────────┐
                          │  prompt → tokens        │
                          │     ▼                   │
TRAINING                  │  GPT.forward()          │
┌─────────────────────────┤     ▼                   │
│  shakespeare.txt        │  logits at last pos     │
│      ▼                  │     ▼                   │
│  encode → tensor[T]     │  temperature + top-k    │
│      ▼ split            │     ▼                   │
│  train[0:0.9]  val[0.9:]│  multinomial sample     │
│      ▼                  │     ▼                   │
│  random batch (B,T)     │  append, repeat         │
│      ▼                  │     ▼                   │
│  GPT.forward()          │  detokenize → text      │
│      ▼                  └─────────────────────────┘
│  cross-entropy loss              ▲
│      ▼                           │
│  loss.backward()                 │
│      ▼                           │
│  clip + AdamW.step()             │
│      ▼                           │
│  every 1k steps: save checkpoint ┘
└──────────────────────────────────┘
```

There is exactly one model class (`GPT`). The training loop and the inference loop both call `GPT.forward()` — they differ only in what they do with its output.

## The data pipeline

```
data/shakespeare.txt   ──open()──>   text: str (1,115,394 chars)
                                          │
                                          ▼
                                    sorted(set(text))   ──> chars: list[65]
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                   stoi: {char→int}                itos: {int→char}
                          │
                          ▼
        torch.tensor([stoi[c] for c in text], dtype=long)
                          │
                          ▼
                   tokens: Long[1_115_394]
                          │            ┌── train_tokens = tokens[ : 1_003_854]   (90%)
                          ├──split────>│
                                       └── val_tokens   = tokens[1_003_854 : ]   (10%)
                          │
                          ▼
   for each step:
       ix = randint(len(split) - block_size - 1, (batch_size,))
       x  = stack([split[i  : i+block_size]   for i in ix])    → (B, T)
       y  = stack([split[i+1: i+block_size+1] for i in ix])    → (B, T)
```

For default `block_size=256`, `batch_size=64`:

| Tensor | Shape | Dtype | Lives on |
|--------|-------|-------|----------|
| `tokens` (train) | `(1_003_854,)` | `int64` | CPU |
| `x` (input) | `(64, 256)` | `int64` | device |
| `y` (target) | `(64, 256)` | `int64` | device |

`y` is just `x` shifted by one position. That shift IS the next-token-prediction objective.

## The model: shapes through one forward pass

Configuration constants for the Medium model:

```
B = 64   batch_size
T = 256  block_size       (max sequence length)
V = 65   vocab_size        (unique characters)
C = 384  n_embd            (embedding dim, "width" of the model)
H = 6    n_head
D = 64   head_dim = C / H
L = 6    n_layer
F = 1536 mlp_inner = 4*C
```

### Step 1 — embeddings

```
idx (B, T)                                  int64 token IDs
   │
   ├─ wte: Embedding[V, C]    ──>   tok_emb (B, T, C)        float32
   │     parameters: 65 * 384 = 24,960  (shared with lm_head, see weight tying)
   │
   ├─ pos = arange(T)          ──>  pos     (T,)              int64
   │
   └─ wpe: Embedding[T, C]    ──>   pos_emb (T, C)            float32
         parameters: 256 * 384 = 98,304

x = tok_emb + pos_emb   ──>   (B, T, C)    via broadcasting
```

### Step 2 — N transformer blocks

Each `Block` runs the same shape transformation: `(B, T, C) → (B, T, C)`. There are `L = 6` of them in series.

```
x                                            (B, T, C)
│
├──────── residual ────────┐
▼                          │
LayerNorm(C)               │      params: 2*C = 768
│                          │
▼                          │
CausalSelfAttention        │
│                          │
└────── add ───────────────┘
│
├──────── residual ────────┐
▼                          │
LayerNorm(C)               │      params: 2*C = 768
│                          │
▼                          │
MLP                        │
│                          │
└────── add ───────────────┘
│
▼
output                                       (B, T, C)
```

#### Inside `CausalSelfAttention`

```
x (B, T, C)
   │
   ▼
c_attn: Linear(C → 3C)              params: C*(3C) + 3C = 443,520
   │
   ▼
qkv (B, T, 3C)
   │
   split along last dim
   ▼
q, k, v  each (B, T, C)
   │
   reshape  (B, T, C) → (B, T, H, D) → (B, H, T, D)
   ▼
   ┌───────────────────────────────┐
   │ scaled_dot_product_attention  │
   │   - score = Q @ K^T / sqrt(D) │
   │   - mask = upper triangular   │  (causal: position i sees only 0..i)
   │   - softmax over last dim     │
   │   - out = attention @ V       │
   └───────────────────────────────┘
   │
   ▼
y (B, H, T, D) → transpose → contiguous → view → (B, T, C)
   │
   ▼
c_proj: Linear(C → C)               params: C*C + C = 147,840
   │
   ▼
output (B, T, C)
```

Total per attention layer: ~591k params. Note `is_causal=True` is what makes the model **GPT-style**: each token can only attend to tokens at or before its position. Without this mask, you would have a bidirectional encoder (BERT-style).

#### Inside `MLP`

```
x (B, T, C)
   │
   ▼
c_fc: Linear(C → 4C)                params: C*4C + 4C = 591,360
   │
   ▼
(B, T, 4C)
   │
   ▼
GELU(approximate=tanh)
   │
   ▼
c_proj: Linear(4C → C)              params: 4C*C + C = 590,208
   │
   ▼
output (B, T, C)
```

Total per MLP: ~1.18M params. The MLP holds the bulk of the per-layer parameters — most of the model's "knowledge" lives here.

### Step 3 — final layer norm and projection to vocabulary

```
x (B, T, C)
   │
   ▼
ln_f: LayerNorm(C)                  params: 2C = 768
   │
   ▼
lm_head: Linear(C → V, bias=False)  params: V*C = 24,960
   │                                (TIED with wte.weight, so 0 new params)
   ▼
logits (B, T, V)
```

**Weight tying.** `self.transformer.wte.weight = self.lm_head.weight` makes the input embedding matrix and the output projection matrix the same tensor. This:

- Halves the embedding-related parameter count (saves ~25k here, but ~38M with GPT-2's 50,257 vocab).
- Forces consistency: a vector that maps token *i* into the residual stream is the same vector used to score "is the next token *i*?" at the output.

### Step 4 — loss

If `targets` is provided (training), the loss is computed inside `forward`:

```
logits.view(-1, V)             ── (B*T, V)
targets.view(-1)               ── (B*T,)

loss = cross_entropy(logits, targets)    ── scalar
```

`cross_entropy` combines `log_softmax` + `NLLLoss`. The model's prediction at every position contributes to the loss; this is what makes training data-efficient — one batch yields `B*T = 16,384` supervised examples.

If `targets` is `None` (inference), only `logits` are returned and the loss step is skipped.

## Parameter budget

For default `n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=65`:

| Component | Calculation | Params |
|-----------|-------------|--------|
| Token embeddings (`wte`, tied with `lm_head`) | 65 × 384 | 25 K |
| Position embeddings (`wpe`) | 256 × 384 | 98 K |
| Per-block: 2× LayerNorm | 2 × (2 × 384) | 1.5 K |
| Per-block: attention `c_attn` | 384 × 1152 + 1152 | 443 K |
| Per-block: attention `c_proj` | 384 × 384 + 384 | 148 K |
| Per-block: MLP `c_fc` | 384 × 1536 + 1536 | 591 K |
| Per-block: MLP `c_proj` | 1536 × 384 + 384 | 590 K |
| Per-block subtotal | | **1.77 M** |
| 6 blocks | 6 × 1.77 M | 10.6 M |
| Final LayerNorm | 2 × 384 | 768 |
| **Total** | | **~10.78 M** |

The model is **~10× larger than its training data is informative** (10M params vs ~1M characters). That ratio guarantees overfitting — and that is the central pedagogical point of [Part 3](../03-training-loop.md#overfitting-train-loss-vs-val-loss).

## Memory: where does it go?

For `B=64, T=256, C=384, n_layer=6` on a recent GPU (fp32):

| Allocation | Approx. size |
|------------|-------------:|
| Model parameters (10.8 M × 4 bytes) | ~43 MB |
| Adam moment buffers (2 × params, fp32) | ~86 MB |
| Activations checkpointed for backward (~ B*T*C * n_layer * 4) | ~150 MB |
| Mini-batch (`x`, `y`, both `int64`) | ~0.3 MB |
| **Total working set** | **~300 MB** |

Comfortable on any 4 GB+ GPU. CPU-only training also fits easily but runs ~10-30× slower because of cache misses, no fused GEMM kernels, and no parallel attention.

If you run out of memory, the dominant lever is `batch_size` (linear), then `block_size` (linear in attention activations), then `n_embd` (super-linear because most layers scale as `C²`).

## The training loop, step by step

```
for step in range(max_steps):

    if step % val_every == 0:
        # Evaluate on held-out 10% of the data
        model.eval()
        with torch.no_grad():
            avg_val_loss = mean(model(*get_val())[1].item() for _ in range(20))
        log["val"].append((step, avg_val_loss))
        model.train()

    # Cosine LR schedule with linear warmup
    lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # Training step
    x, y = get_train()                                  # (B, T), (B, T)
    _, loss = model(x, y)                                # scalar loss
    optimizer.zero_grad()                                # clear .grad on every parameter
    loss.backward()                                      # populate .grad via autograd
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()                                     # AdamW update

    if step > 0 and step % sample_every == 0:
        # Generate a snippet so you can see the model improve over time
        sample = generate(model, "To be or not", stoi, itos,
                          max_new_tokens=100, temperature=0.8)
        tqdm.write(sample)

    if step > 0 and step % ckpt_every == 0:
        # Persist weights + tokenizer + config so generation can resume cold
        torch.save({"step": step, "model_state_dict": model.state_dict(),
                    "config": config, "stoi": stoi, "itos": itos},
                   f"checkpoint_{step}.pt")
```

### What each line is doing

- **`model.train()` / `model.eval()`** toggle dropout and batch-norm modes. Our model has neither, but `eval()` is still important: `scaled_dot_product_attention` may take a faster path, and (most relevant in the future) any future dropout would be disabled.
- **`with torch.no_grad():`** skips building the autograd graph during validation, saving memory and time.
- **`get_lr`** implements warmup + cosine decay (see [Part 3 → Step 3](../03-training-loop.md#step-3-learning-rate-schedule)).
- **`optimizer.zero_grad()`** is required because gradients accumulate by default.
- **`loss.backward()`** runs reverse-mode autodiff, populating `.grad` on every parameter.
- **`clip_grad_norm_`** rescales all gradients so their global L2 norm ≤ 1.0. This stops occasional huge gradients from blowing up the weights.
- **`optimizer.step()`** applies AdamW: `p ← p - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * p`.

## Inference: the autoregressive loop

```
prompt: str
    │
    ▼
tokens = [stoi[c] for c in prompt]             (already encoded)
    │
    ▼
idx = tensor(tokens).unsqueeze(0)              (1, len(prompt))
    │
    ▼
for _ in range(max_new_tokens):
    idx_cond = idx[:, -block_size:]            keep only last T tokens
    logits, _ = model(idx_cond)                (1, t, V)
    next_logits = logits[:, -1, :]             (1, V)   only last position matters
    next_logits = next_logits / temperature
    next_logits = topk_filter(next_logits, k)  set < kth largest to -inf
    probs = softmax(next_logits, dim=-1)
    next_id = multinomial(probs, num_samples=1)
    idx = cat([idx, next_id], dim=1)
    │
    ▼
text = "".join(itos[i] for i in idx[0].tolist())
```

Two important subtleties:

1. **`idx[:, -block_size:]`** — the model has no positional embeddings beyond `block_size = 256`. If you generate longer outputs than that, you must keep only the last 256 tokens of context. Earlier tokens fall out of the model's window.

2. **`logits[:, -1, :]`** — the model returns a distribution at every position, but during generation only the last position is informative. The earlier positions are conditioned on prefixes you already sampled.

## Why each piece is needed

A useful exercise is asking what happens if you remove each piece. Brief answers:

| Remove | Result |
|--------|--------|
| Position embeddings (`wpe`) | The model can't tell `"abc"` from `"cba"` — sequence order is lost. |
| Causal mask (`is_causal=True`) | Becomes a non-causal encoder; trivially perfect train loss because each position can see its own target. |
| LayerNorm | Activations explode/vanish; deeper than 2 layers becomes untrainable. |
| Residual connections | Gradients vanish at depth; loss plateaus near initial value. |
| Multi-head (use one head with full C) | Same parameter count but worse — multiple heads let the model attend to different relationships in parallel. |
| Weight tying | Vocab-related params double; matters more with large vocabularies. |
| Cross-entropy loss | No useful gradient signal for next-token prediction. |
| Gradient clipping | Occasional NaN losses when batch hits an outlier. |
| Cosine LR decay (use constant LR) | Converges noisily; final loss ~0.05-0.1 worse. |
| Warmup (start at full LR) | First few steps may diverge, especially with `lr=1e-3+`. |
| Validation set | You cannot detect overfitting; you would happily train past the optimum. |

## What lives where

| File | Defines | Imports from |
|------|---------|--------------|
| `model.py` | `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` | `torch`, `torch.nn` |
| `train.py` | `get_device`, `load_data`, `get_lr`, `train`, `__main__` entry | `model`, `generate`, `torch`, `tqdm`, `math`, `json`, `os` |
| `generate.py` | `generate`, `__main__` entry | `model`, `torch`, `argparse` |

There is one circular-looking import: `train.py` imports `generate` (to print samples mid-training), and `generate.py` imports from `model`. That works because Python only executes `generate.py`'s top level when imported; `generate.main()` runs only when `generate.py` is the entry point.

## Key invariants

These should hold at all times — if any breaks, something is wrong:

- `set(text) == set(stoi.keys())` — no character encountered at training that isn't in the tokenizer
- `len(stoi) == len(itos) == config.vocab_size`
- `all(0 <= idx < config.vocab_size)` for every input token
- `idx.shape[-1] <= config.block_size` whenever `model(idx)` is called
- `loss.requires_grad and loss.is_leaf is False` during training
- After `optimizer.step()`, every parameter's `.grad` is non-None
- The first batch's loss is in `[ln(vocab_size) - 0.3, ln(vocab_size) + 0.3]` ≈ `[3.87, 4.47]` for vocab=65

The smoke test in [`04-test-run.md → Pass criteria`](04-test-run.md#pass-criteria) checks the last invariant.

## Next

- Tweak parameters and understand tradeoffs: [06-configuration.md](06-configuration.md)
- Look up an unfamiliar term: [07-glossary.md](07-glossary.md)
- Things broke: [08-troubleshooting.md](08-troubleshooting.md)
