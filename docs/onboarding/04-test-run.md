# 04 — Test Run

This is the most important onboarding doc. It takes you from an empty `scratchpad/` to a trained model that generates Shakespeare-style text.

The repo intentionally ships no Python source files (see [`01-what-is-this.md`](01-what-is-this.md#why-the-repo-looks-empty)). To **test run** the codebase you must first scaffold three files yourself: `model.py`, `train.py`, and `generate.py`. This doc gives you two ways to do that:

- **Fast path** — paste the assembled snippets below, run the smoke test, see text generated. ~5-10 minutes.
- **Full path** — work through Parts 1-4 in order, writing each file as you read. ~2-3 hours.

Either way, the same three files end up in `scratchpad/`.

> The code in this document is the canonical assembled version derived from [`docs/01-tokenization.md`](../01-tokenization.md), [`docs/02-the-transformer.md`](../02-the-transformer.md), [`docs/03-training-loop.md`](../03-training-loop.md), and [`docs/04-text-generation.md`](../04-text-generation.md). If anything diverges, those chapters are the source of truth.

---

## 0. Prerequisites for this doc

- You have completed [`03-setup.md`](03-setup.md).
- `uv run python -c "import torch; print(torch.__version__)"` prints a version.
- You are in the project root (`llm-from-scratch/`) and `scratchpad/` exists.

```bash
cd llm-from-scratch
mkdir -p scratchpad
ls scratchpad
# (empty)
```

---

## 1. Scaffold `model.py`

Create `scratchpad/model.py` with the following exact content. This is the model from [Part 2](../02-the-transformer.md).

```python
# scratchpad/model.py
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss
```

### Verify `model.py`

```bash
cd scratchpad
uv run python -c "from model import GPT, GPTConfig; m = GPT(GPTConfig()); print(f'{sum(p.numel() for p in m.parameters())/1e6:.2f}M params')"
# 10.78M params
```

Anything between **10.5M** and **11.0M** is correct.

---

## 2. Scaffold `generate.py`

Create `scratchpad/generate.py`. This is the inference code from [Part 4](../04-text-generation.md).

```python
# scratchpad/generate.py
import argparse
import torch

from model import GPT, GPTConfig  # noqa: F401  (GPTConfig referenced via checkpoint)


@torch.no_grad()
def generate(model, prompt, stoi, itos, max_new_tokens=200, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    tokens = [stoi[c] for c in prompt if c in stoi]
    if not tokens:
        tokens = [0]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return "".join([itos[i] for i in idx[0].tolist()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--prompt", default="To be or not")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ck = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = GPT(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    out = generate(
        model, args.prompt, ck["stoi"], ck["itos"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(out)


if __name__ == "__main__":
    main()
```

### Verify `generate.py` imports

```bash
uv run python -c "import generate; print('ok')"
# ok
```

You cannot fully test it yet — there is no checkpoint. We get one in step 4.

---

## 3. Scaffold `train.py`

Create `scratchpad/train.py`. This combines the data loader and training loop from [Part 3](../03-training-loop.md).

```python
# scratchpad/train.py
import json
import math
import os
import torch
from tqdm import tqdm

from model import GPT, GPTConfig
from generate import generate


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(filepath, block_size, batch_size, device):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Dataset: {len(tokens):,} chars, vocab size: {vocab_size}")

    def get_batch(split_tokens):
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
        x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y

    n = int(0.9 * len(tokens))
    return (lambda: get_batch(tokens[:n])), (lambda: get_batch(tokens[n:])), vocab_size, stoi, itos


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(
    data_path="../data/shakespeare.txt",
    max_steps=5000,
    batch_size=64,
    n_layer=6,
    n_head=6,
    n_embd=384,
    block_size=256,
    max_lr=1e-3,
    warmup_steps=100,
    sample_every=100,
    val_every=100,
    ckpt_every=1000,
    out_dir=".",
):
    device = get_device()
    print(f"Using device: {device}")
    get_train, get_val, vocab_size, stoi, itos = load_data(
        data_path, block_size, batch_size, device
    )
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_layer}L/{n_head}H/{n_embd}D, {n_params/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    min_lr = max_lr * 0.1

    log = {"steps": [], "train": [], "val_steps": [], "val": []}
    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                vlosses = []
                for _ in range(20):
                    x, y = get_val()
                    _, l = model(x, y)
                    vlosses.append(l.item())
                vloss = sum(vlosses) / len(vlosses)
                log["val_steps"].append(step)
                log["val"].append(vloss)
                tqdm.write(f"Step {step:5d} | val loss: {vloss:.4f}")
            model.train()

        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_train()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        log["steps"].append(step)
        log["train"].append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        if step > 0 and step % sample_every == 0:
            model.eval()
            sample = generate(model, "To be or not", stoi, itos,
                              max_new_tokens=100, temperature=0.8)
            tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
            model.train()

        if step > 0 and step % ckpt_every == 0:
            path = os.path.join(out_dir, f"checkpoint_{step}.pt")
            torch.save({
                "step": step, "model_state_dict": model.state_dict(),
                "config": config, "stoi": stoi, "itos": itos,
            }, path)
            tqdm.write(f"Saved {path}")

    final_path = os.path.join(out_dir, "checkpoint_final.pt")
    torch.save({
        "step": max_steps, "model_state_dict": model.state_dict(),
        "config": config, "stoi": stoi, "itos": itos,
    }, final_path)
    with open(os.path.join(out_dir, "loss_log.json"), "w") as f:
        json.dump(log, f)
    print(f"Saved {final_path}")
    return model, stoi, itos


if __name__ == "__main__":
    train()
```

### Verify `train.py` imports

```bash
uv run python -c "import train; print('ok')"
# ok
```

---

## 4. Run the smoke test (the fast test run)

This is a **2-5 minute** training run with a tiny model. The point is to confirm the whole pipeline works — not to get good output.

From inside `scratchpad/`:

```bash
uv run python -c "from train import train; train(max_steps=200, n_layer=2, n_head=2, n_embd=128, batch_size=16, sample_every=100, val_every=50, ckpt_every=200)"
```

What you should see, in order:

```
Using device: mps                                ← or cuda / cpu
Dataset: 1,115,394 chars, vocab size: 65
Model: 2L/2H/128D, 0.5M params
Step     0 | val loss: 4.1701                    ← starts near ln(65) ≈ 4.17
Training:   0%|                  | 0/200 ...
Step    50 | val loss: 2.7xxx
Step   100 | val loss: 2.5xxx
--- Step 100 sample ---
To be or not[50-100 chars of mostly-gibberish]
---
Step   150 | val loss: 2.4xxx
Saved ./checkpoint_200.pt
Saved ./checkpoint_final.pt
```

### Pass criteria

The smoke test passes if **all three** of these are true:

1. **Loss starts near 4.17.** That's `ln(vocab_size) = ln(65)`, the baseline for a uniformly random predictor. Anything below 3.5 at step 0 is suspicious; anything above 5 means data was loaded wrong.
2. **Loss is decreasing.** By step 200 you should see val loss ≤ 2.6. If it is stuck near 4.17, the model is not learning — see [`08-troubleshooting.md → "Loss not decreasing"`](08-troubleshooting.md#loss-not-decreasing).
3. **A checkpoint file appears.** `ls scratchpad/checkpoint_*.pt` should show at least `checkpoint_200.pt` and `checkpoint_final.pt`.

Loss starting above 4.17 (say, at 6.3) usually means the tokenizer is mismatched — see Part 1's note about BPE on small data.

---

## 5. Generate text from your smoke-test checkpoint

```bash
uv run python generate.py checkpoint_final.pt --prompt "To be or not" --seed 42 --max_new_tokens 200
```

Expected output: 200 characters of mostly-gibberish containing recognizable letters and the prompt at the start. **It will not be coherent yet** — 200 steps is far too few. But you have proven the full pipeline works end to end:

```
data → tokenize → batches → forward → loss → backward → optimizer → checkpoint → load → generate
```

If this prints text, **you have successfully test-run the codebase.** Everything below is about training a model that actually produces good Shakespeare.

---

## 6. Train the Medium model (the full run)

The default `train()` call uses the Medium config (6L/6H/384D, ~10M params, 5,000 steps, batch_size=64). On the supported devices:

| Device | Time | Steps/sec |
|--------|------|-----------|
| Apple M3 Pro | ~45 min | ~1.9 |
| NVIDIA Colab T4 | ~15 min | ~5.5 |
| NVIDIA RTX 3080 | ~5-8 min | ~10-15 |
| Modern CPU only | 2-4 hours | ~0.5 |

```bash
cd scratchpad
uv run python train.py
```

What you will see during the run:

- A `tqdm` progress bar with current `loss` and `lr`.
- Every 100 steps: a validation-loss line and a 100-character sample starting with `"To be or not"`.
- Every 1,000 steps: `Saved ./checkpoint_1000.pt` (then `_2000.pt`, `_3000.pt`, …).
- At the end: `checkpoint_final.pt` and `loss_log.json`.

The shape of the val-loss curve is documented in [Part 3 → Watching the Model Learn](../03-training-loop.md#watching-the-model-learn). The short version:

| Approx. val loss | Quality |
|------------------|---------|
| 4.17 | random |
| 3.3 | char frequencies learned |
| 2.5 | common bigrams (`th`, `he`) |
| 1.8 | recognizable words |
| 1.5-1.7 | Shakespeare-like — **this is your best model** |
| < 1.0 | overfitting (memorizing the training data) |

The lowest val loss usually occurs around **step 1500-2500**. After that, training loss keeps falling but val loss climbs — that is overfitting.

---

## 7. Generate Shakespeare from your trained checkpoint

```bash
uv run python generate.py checkpoint_2000.pt --prompt "To be or not" --temperature 0.8 --top_k 40 --seed 42
```

Realistic output at val loss ~1.6:

```
To be or not to be some of you shall know
That everlature by Romeo: what news,
Which you had knock'd my part to speak
```

Try other prompts and settings:

```bash
# more deterministic
uv run python generate.py checkpoint_2000.pt --prompt "ROMEO:" --temperature 0.5 --seed 0

# more creative (will be wilder)
uv run python generate.py checkpoint_2000.pt --prompt "JULIET:" --temperature 1.0 --top_k 20 --seed 0

# longer
uv run python generate.py checkpoint_2000.pt --prompt "Friends, Romans" --max_new_tokens 800 --seed 0
```

---

## 8. Plot the loss curve (optional)

```bash
uv run pip install matplotlib
uv run python -c "
import json, matplotlib.pyplot as plt
log = json.load(open('loss_log.json'))
plt.figure(figsize=(10,6))
plt.plot(log['steps'], log['train'], alpha=0.3, label='train')
plt.plot(log['val_steps'], log['val'], 'o-', label='val')
plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.savefig('loss_curve.png')
print('wrote loss_curve.png')
"
```

A healthy run shows train loss falling smoothly with val loss tracking it for a while, then val loss flattening or rising while train loss continues to fall. The minimum of the val curve is your best checkpoint.

---

## Quick reference: full test-run sequence

For copy-paste use:

```bash
# from llm-from-scratch/
uv sync
mkdir -p scratchpad
# (paste model.py, generate.py, train.py from sections 1-3 above)

cd scratchpad
uv run python -c "from train import train; train(max_steps=200, n_layer=2, n_head=2, n_embd=128, batch_size=16, sample_every=100, val_every=50, ckpt_every=200)"
uv run python generate.py checkpoint_final.pt --prompt 'To be or not' --seed 42

# once smoke test passes, full run:
uv run python train.py
uv run python generate.py checkpoint_2000.pt --prompt 'To be or not' --seed 42
```

## Next

- For deeper understanding of what just ran: [05-architecture.md](05-architecture.md)
- For tweaking parameters: [06-configuration.md](06-configuration.md)
- If something broke: [08-troubleshooting.md](08-troubleshooting.md)
