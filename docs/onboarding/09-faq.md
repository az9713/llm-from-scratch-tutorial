# 09 — FAQ

Questions asked at the live workshop ("Train an LLM from scratch on your laptop", presented by Angelos Perivolaropoulos at ElevenLabs), with the answers — condensed and edited for clarity.

For technical setup or error problems, see [`08-troubleshooting.md`](08-troubleshooting.md). For term definitions, see [`07-glossary.md`](07-glossary.md).

---

## About the workshop

### Why train from scratch instead of fine-tuning?

You see what every component does and why it exists. A fine-tuning tutorial calls `from_pretrained(...)` and you never learn how the model was built. Once you've built it once, fine-tuning, post-training, RLHF, distillation — they all become variations on the same loop you already understand.

### How does this compare to nanoGPT?

nanoGPT (Karpathy) targets reproducing GPT-2 (124M params, BPE). This workshop strips that down: ~10M params, character-level tokenization, ~45 minutes on a laptop. The architecture is otherwise the same — same blocks, same attention, same training loop. You can read nanoGPT after this workshop and feel right at home.

### Is this still relevant given how big modern models are?

Yes. The pre-training architecture used by GPT-4, Gemini, and Claude is largely the same decoder-only transformer with attention + MLP + residual + LayerNorm. The improvements at the frontier are **post-training** (instruction tuning, RLHF, reasoning fine-tuning) and **engineering tricks** for scaling (mixture of experts, longer context, faster attention). The base architecture you build here is ~80% of what runs at the frontier.

---

## Tokenization

### In Python source code, what is a token?

It depends on the tokenizer, not on the language. With our character-level tokenizer, every character — letter, digit, brace, semicolon, newline — is its own token.

With BPE (GPT-2's tokenizer), the tokenizer **discovers patterns from training data**. If your training corpus contains a lot of Python, you'd likely see `for `, `def `, `enumerate`, `len(`, etc. as single tokens. There is no human in the loop deciding "this should be a token" — the algorithm finds frequent byte pairs and merges them.

For unusual variable names or non-Latin characters, BPE falls back to byte-level tokens — each byte its own token. This works for any input but is inefficient.

### Why character-level for this workshop?

Because Shakespeare is only ~1 MB. With BPE's 50,257-token vocab, most token bigrams would never appear in the data, so the model can't learn sequential statistics. With a 65-character vocab, all 4,225 possible bigrams are well-represented. We tested it: BPE on Shakespeare plateaus at val loss ~6.3 (unigram frequency); char-level reaches ~1.5.

### Will character-level work for, say, web-scale text?

No. Two problems:

1. Sequences become 3-5× longer (you spend more compute per sample).
2. The model has to learn spelling from scratch, wasting capacity on patterns BPE captures for free.

For real-world LLMs, BPE-class tokenizers are standard.

---

## Architecture

### Is `block_size` (sequence length) related to the number of parameters?

No. You can have small models with long context, or large models with short context. `block_size` mainly controls the position-embedding table size (linear) and the per-layer attention compute (quadratic).

The thing that *does* break when you crank up `block_size` is memory. Attention scales as `O(block_size² × n_embd)`. On 16 GB hardware, `block_size=1024` is roughly the wall.

### If I had more compute, what would I increase to make the model "reason better"?

In rough priority order:

1. **`n_embd`** — width matters most (most parameters live in the MLPs, which scale as `n_embd²`).
2. **`n_layer`** — depth lets the model chain transformations.
3. **More data** — without it, more parameters just mean faster overfitting.
4. **`block_size`** — useful only if your data has long-range structure.

A real production model balances all of these against its data corpus.

### Why not increase `block_size` to 1 million for our model?

You'd hit memory and compute walls instantly — attention is quadratic. Frontier labs make 1M context work by changing the architecture (sliding window attention, sparse attention, hierarchical attention, KV-cache compression). The base transformer doesn't scale to that on its own.

### Is each transformer block the same architecture?

Same architecture, **different weights**. Each `Block` instance has its own `c_attn`, `c_proj`, `c_fc`, etc., and learns its own role. Early blocks tend to learn local syntactic patterns; later blocks tend to track longer-range structure.

### Why pre-norm (LayerNorm before attention/MLP)?

Stability. Post-norm (the original 2017 transformer) trains fine for shallow models but becomes harder to optimize past ~6-8 layers. Pre-norm is now standard since GPT-2 (2019).

### What does "residual connection" actually do?

It lets gradients flow directly through the network during backpropagation. Each layer learns a *small adjustment* to its input rather than reinventing the representation from scratch. Without residuals, deep networks become numerically intractable to train.

---

## Training

### Why next-token prediction as the objective?

Three reasons:

1. **It's self-supervised.** No human labels. Every piece of text is simultaneously input and target.
2. **It scales.** The amount of training data is bounded only by how much text you can find.
3. **It generalizes.** A model that can predict any next token from any context implicitly learns syntax, factual knowledge, reasoning patterns — anything that helps reduce loss.

### What's a good loss number?

For our setup (vocab=65, char-level Shakespeare):

| Val loss | Meaning |
|---------:|---------|
| 4.17 | random initialization (= `ln(65)`) |
| 3.3 | learned character frequencies |
| 2.5 | learned common bigrams |
| 1.5-1.7 | best — Shakespeare-like text |
| < 1.0 | overfitting |

The 4.17 baseline is universal: a model with no information about the data should assign uniform probability `1/V` to every token, which gives cross-entropy `ln(V)`.

### Why does the model start at exactly `ln(vocab_size)`?

A randomly-initialized model produces near-uniform output distributions (the embeddings haven't learned anything yet). Cross-entropy on a uniform distribution over `V` classes is exactly `ln(V)`.

### What is overfitting and how do I tell?

Train loss decreasing while val loss increases. The model is memorizing training data instead of learning patterns. With a 10M-param model on 1 MB of Shakespeare, this is **inevitable** — the question is just when. For our default config it starts around step 1500-2500.

### What's the best checkpoint?

The one with the **lowest validation loss**, not the lowest train loss. Save your best-val-loss checkpoint and use it for generation.

### Why do you use AdamW instead of Adam or SGD?

Compared to Adam: AdamW applies weight decay directly to weights, not via the gradient. Cleaner regularization.

Compared to SGD: AdamW adapts per-parameter learning rates from gradient statistics. For transformers with very different gradient scales across parameters (embeddings vs. attention vs. MLP), this matters a lot.

### What does gradient clipping do?

Caps the L2 norm of all gradients combined at some threshold (1.0 in our code). Occasional large gradients from a hard batch can otherwise blow up the weights. With clipping, even bad batches can't push the model into a divergent regime.

### Why warm up the learning rate?

Adam estimates per-parameter moment statistics over time. At step 0, those estimates are garbage. Starting at full LR can push parameters into bad regions before the moments settle. Linear warmup over the first 100 steps gives Adam's accumulators time to stabilize.

### Why cosine decay?

Empirical: it's the schedule that consistently works for transformer training. Other schedules (step decay, linear decay) work too but cosine is the de facto default. The intuition: large updates early to explore, small updates late to refine.

---

## Generation

### Why not always pick the highest-probability token (greedy)?

Greedy decoding for LLMs produces boring, repetitive, often degenerate text. The model gets stuck in high-probability loops. Sampling from the distribution — controlled by temperature and top-k — produces more natural text.

For some other model types (transcription, translation), greedy or beam search **is** the right choice because there's only one correct answer. For open-ended generation, sample.

### What does the seed do?

Initializes the random number generator. `torch.manual_seed(42)` makes all subsequent random ops (including `torch.multinomial` in our generation) deterministic. Same seed + same checkpoint + same args → same output. This is what makes the workshop competition reproducible.

### If I use greedy decoding, do I still need a seed?

Probably yes. Even "greedy" inference can have non-determinism from non-deterministic CUDA kernels, mixed-precision rounding, etc. Setting a seed is cheap insurance.

### What temperature should I use?

Sweet spot: 0.7-0.9. Below 0.5, output gets repetitive. Above 1.0, output gets wild and incoherent. Our default (0.8) is a reasonable midpoint.

### What does top-k do?

Filters out low-probability tokens before sampling. With `top_k=40`, the model only samples from the 40 most likely tokens at each step. This prevents the rare-but-harmful case of sampling a tail token that derails the sequence (e.g., an unexpected line break or end-of-text).

---

## Reasoning models, multimodal, audio

### How are reasoning models (o1, R1) different from this?

They use the same base architecture. The difference is **post-training**: high-quality chain-of-thought data teaches the model to spend extra tokens on intermediate reasoning before producing the final answer. The training architecture is the same; the data and the loss-shaping are different.

### Do reasoning models share the base with non-reasoning models?

Yes, often. Many labs release a "base" model and one or more fine-tuned variants. The base is the same pre-trained model; the variants differ in their post-training.

### How do you build the chain-of-thought training data?

Mostly humans. Companies like Scale AI hire domain experts (PhDs in physics, math, chemistry, etc.) to write step-by-step reasoning traces for hard problems. This is the most expensive ingredient of a reasoning model — far more than the compute. Quality matters enormously: bad reasoning data trains the model to reason badly.

### Is audio different from text?

Surprisingly similar. The architecture stays largely the same. What changes:

- **Tokenizer**: instead of BPE on text, you tokenize audio (mel-spectrograms → discrete codes).
- **Loss**: cross-entropy works for tokenized audio, but L2 loss between mel-spectrograms is also common for waveform-level training.
- **Modality fusion**: multimodal models put a separate encoder (e.g., for video/audio) on the front, then feed its output vectors into the main transformer's embedding stream.

The interesting difficulty is the tokenizer: how do you represent audio as a discrete vocabulary? That's an entire research area on its own.

### How do multimodal models work?

Each modality has its own encoder. You feed video through a video encoder, audio through an audio encoder, and text through the regular embedding layer. The encoders all output vectors in the same dimensionality (`n_embd`), so the main transformer treats them uniformly: it just sees a sequence of vectors, not knowing which came from which modality.

---

## Practical / extending the workshop

### Should I copy-paste the code or write it line by line?

Both are fine. Most people learn more by typing it themselves at least once. If you're short on time, copy-paste, run, then experiment with modifications — that's also good learning.

### What can I tweak first to improve generation quality?

In rough priority:

1. **Train longer** (but stop at lowest val loss, not last step).
2. **Try different prompts** — they matter more than you'd think.
3. **Tune temperature and top-k** — different settings give different vibes.
4. **Get more data** — switch to a curated poetry dataset for the competition.
5. **Add dropout** — `nn.Dropout(0.1)` in the attention and MLP blocks delays overfitting.

### Can I train this on a phone / Raspberry Pi / etc.?

CPU training works on anything that can run PyTorch and has 4 GB+ of RAM. On a Pi 4 or M1 iPad in a Python environment, the Tiny config would train in hours. The Medium config would take days. It's not the workshop's intended path, but possible.

### Does this generalize to fine-tuning my own LLM?

Yes. You'd start from a pre-trained checkpoint instead of random initialization, use a smaller learning rate, and possibly freeze early layers. Otherwise the loop is identical.

### How do I submit to the competition?

See [Part 6](../06-competition.md). In short: train your best model, generate your best poem with a fixed seed, and submit:

1. The poem.
2. Your final checkpoint (`.pt` file).
3. The exact `python generate.py ...` command (with `--seed`) that reproduces the poem.

---

## Questions specifically about your environment

### Why is `scratchpad/` empty when I cloned the repo?

By design — see [`01-what-is-this.md`](01-what-is-this.md#why-the-repo-looks-empty). You write the code yourself.

### What's the difference between `python train.py` and `uv run python train.py`?

`uv run` activates the `.venv` for that one command and then deactivates. Plain `python` requires you to have already activated the env (`source .venv/bin/activate`). Both end up running the same interpreter from `.venv/bin/python`.

### Can I use Jupyter for this?

Sure. Paste the contents of each file into cells and call `train()` directly. On Colab this is the standard workflow. Locally, you'd just `uv pip install jupyter` and `uv run jupyter lab`.

---

## Workshop competition rules

### Are pretrained models allowed?

No. You must train from random initialization on your own machine (or Colab).

### Can I edit the model output?

You can trim trailing partial sentences. You cannot rewrite, reorder, or correct anything. What the model produced is what you submit.

### Can I cherry-pick from many generations?

Yes. Generate 100 poems and pick your favorite — that's fair game. Just record the seed of your chosen one so it's reproducible.

### Can I change the architecture?

Yes. Within the constraint of training from scratch on your own hardware, you can change the model, the tokenizer, the data, the training loop — anything.

### How is the winner judged?

Coherence, creativity, structure, emotion. Subjective, voted on by participants. See [Part 6](../06-competition.md).

---

## Next

You've reached the end of the onboarding. From here:

- Start building: [`04-test-run.md`](04-test-run.md).
- Read the chapters: [Part 1](../01-tokenization.md) → [Part 2](../02-the-transformer.md) → [Part 3](../03-training-loop.md) → [Part 4](../04-text-generation.md) → [Part 5](../05-putting-it-together.md) → [Part 6](../06-competition.md).
- Or revisit the index: [`README.md`](README.md).
