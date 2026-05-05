# 01 — What is this?

This is a workshop, not a library. By the end of it you will have written a working ~10M-parameter GPT in roughly 300 lines of PyTorch and trained it on a laptop in under an hour. The model will generate Shakespeare-style text from a prompt.

## One-line answer

A guided exercise in which you re-implement the four building blocks of a GPT training pipeline — tokenizer, transformer model, training loop, and text generator — by reading six conceptual docs and typing the code into your own files.

## Who this is for

You should be comfortable with:

- Reading Python (you do not need ML experience)
- Running terminal commands
- Editing source files in any editor

You do **not** need:

- A GPU (Apple Silicon MPS, NVIDIA CUDA, and pure CPU all work)
- Prior knowledge of transformers, attention, or PyTorch
- Cloud credentials or an API key

The minimum hardware target is a laptop with **16 GB of RAM**. Anything weaker still works, just slower.

## Why the repo looks empty

When you clone this repo, the only Python you see is in `pyproject.toml`. There is no `model.py`, `train.py`, or `generate.py`. **This is intentional.** You write those files yourself, in a folder called `scratchpad/` that is in `.gitignore`.

This is the workshop's whole point. nanoGPT and other from-scratch tutorials hand you working code and explain it after the fact. Here, you build the code line by line as the docs explain why each piece exists. The only files committed to the repo are:

```
llm-from-scratch/
├── README.md                  ← project overview
├── pyproject.toml             ← Python project metadata + deps
├── uv.lock                    ← pinned dependency versions
├── .python-version            ← target Python version (3.12)
├── data/
│   └── shakespeare.txt        ← ~1.1 MB of works of Shakespeare
└── docs/
    ├── 01-tokenization.md     ← chapter 1
    ├── 02-the-transformer.md  ← chapter 2
    ├── 03-training-loop.md    ← chapter 3
    ├── 04-text-generation.md  ← chapter 4
    ├── 05-putting-it-together.md
    ├── 06-competition.md
    └── onboarding/            ← you are here
```

After you finish the workshop, your worktree will look like this:

```
llm-from-scratch/
├── ... (everything above)
└── scratchpad/                ← gitignored — your code lives here
    ├── model.py               ← from Part 2
    ├── train.py               ← from Parts 1 + 3
    ├── generate.py            ← from Part 4
    ├── checkpoint_1000.pt     ← saved by train.py
    ├── checkpoint_2000.pt
    ├── ...
    └── loss_log.json          ← optional, for plotting
```

## What you will build, four ways

### 1. As a system

```
┌──────────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────┐
│ shakespeare  │ → │  Tokenize  │ → │   Train GPT  │ → │  Generate  │
│   1.1 MB     │   │  65 chars  │   │ 5,000 steps  │   │   text     │
└──────────────┘   └────────────┘   └──────────────┘   └────────────┘
                       Part 1          Parts 2 & 3        Part 4
```

### 2. As components

| Component | What it does | Lines of code | File |
|-----------|--------------|---------------|------|
| **Tokenizer** | Map every unique character to an integer ID and back | ~10 | inside `train.py` |
| **Model** | A 6-layer GPT (~10M parameters) that predicts the next token | ~120 | `model.py` |
| **Training loop** | Loads batches, computes loss, backpropagates, schedules learning rate, saves checkpoints | ~80 | `train.py` |
| **Generator** | Autoregressive sampling with temperature and top-k filtering | ~30 | `generate.py` |

### 3. As a function call

```python
text = generate(
    model=load_checkpoint("checkpoint_2000.pt"),
    prompt="To be or not",
    temperature=0.8,
    top_k=40,
    seed=42,
)
print(text)
# → "To be or not to be some of you shall know\n
#    That everlature by Romeo: what news,\n
#    Which you had knock'd my part to speak"
```

### 4. As a learning experience

The workshop walks you through five mental shifts most ML beginners stumble on:

1. **Tokens are integers, not letters.** A model sees `[20, 43, 50, 50, 53]`, not `"hello"`. (Part 1)
2. **A transformer is just a stack of identical blocks.** Once you understand one block, you understand the whole model. (Part 2)
3. **Training is a loop, not a function.** Forward → loss → backward → step, repeat. The architecture matters less than the loop. (Part 3)
4. **Generation is one token at a time.** The model only ever predicts the next token; long outputs are made of many short predictions. (Part 4)
5. **Bigger is not better.** A 10M-parameter model overfits Shakespeare in 30 minutes. The interesting tradeoff is *capacity vs. data size*. (Part 5)

## Mental model: the workshop loop

```
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │   Read a docs/0X-*.md chapter                       │
  │     │                                               │
  │     ▼                                               │
  │   Type the code into scratchpad/<file>.py           │
  │     │                                               │
  │     ▼                                               │
  │   Run it, watch the output, fix what's wrong        │
  │     │                                               │
  │     ▼                                               │
  │   Move to the next chapter                          │
  │                                                     │
  └─────────────────────────────────────────────────────┘
```

This is the cycle for the entire workshop. Skipping the "type it yourself" step is allowed (each doc has copy-paste-ready snippets) but defeats the purpose.

## What this workshop is *not*

- **Not a state-of-the-art model.** ~10M params trained on 1 MB of Shakespeare cannot compete with anything modern. The point is to understand the pipeline.
- **Not a reproduction of GPT-2.** GPT-2 (124M params, 50,257 vocab, BPE) was the inspiration, but we use a smaller architecture, a 65-character vocabulary, and 256-token context.
- **Not a fine-tuning tutorial.** There is no pre-trained checkpoint anywhere in this repo. You train from random initialization.
- **Not production-ready.** No mixed precision, no model parallelism, no compiled kernels, no deepspeed, no LoRA. The training loop is intentionally simple.

## How long will this take?

| Path | Time |
|------|------|
| Read all docs without writing code | 30-45 min |
| Read + paste code + run smoke test (Tiny config) | 60-75 min |
| Read + write code yourself + train Medium config | 2-3 hours |
| Read + write + experiment + competition | 4-6 hours |

The original live workshop ran 90 minutes. The full thing is a comfortable afternoon project.

## The four building blocks at a glance

| Block | Job | Where it lives |
|-------|-----|----------------|
| **Tokenizer** | text ↔ integer IDs | character map (`stoi`/`itos`) inside `train.py` |
| **Model architecture** | maps token IDs to a probability distribution over the next token | `model.py` (the `GPT` class and its parts) |
| **Training loop** | nudges the model's weights so its predictions match the data | `train()` in `train.py` |
| **Inference** | uses the trained weights to produce new text | `generate()` in `generate.py` |

Everything else — gradient clipping, learning-rate schedules, validation loss, checkpointing — is supporting machinery for the four blocks above.

## Next

[02-prerequisites.md →](02-prerequisites.md)
