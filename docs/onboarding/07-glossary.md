# 07 — Glossary

Every term that appears across the docs, defined once, with an example. Skim once, return when stuck.

Sorted alphabetically.

---

**AdamW** — The optimizer used by `train.py`. A variant of Adam where weight decay is applied directly to weights instead of being mixed into the gradient. The de facto standard for training transformers. Constructed in `train.py` as `torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)`.

**Attention head** — One of `n_head` parallel attention computations per layer. Each head has its own query/key/value subspace of size `n_embd / n_head` and can specialize in a different relational pattern. With `n_embd=384, n_head=6`, each head is 64-dimensional.

**Autoregressive** — A model that generates one token at a time, conditioning each prediction on all previously generated tokens. GPT is autoregressive. The opposite (e.g., BERT) processes all positions in parallel and is bidirectional.

**Backpropagation** — Reverse-mode automatic differentiation. After `loss.backward()`, every learnable parameter has a `.grad` attribute holding `∂loss/∂param`. The optimizer uses these gradients to nudge parameters in the direction that reduces loss.

**Batch / `batch_size`** — Number of sequences processed in parallel per training step. Default 64. Larger batches give smoother gradients but use more memory. Each batch is `(batch_size, block_size)` integers.

**Bigram** — A two-token sequence (e.g., `"th"`, `"he"`). Character-level Shakespeare has 65² = 4,225 possible bigrams, all of which appear many times in 1.1 MB of text. With BPE's 50,257-token vocab, most bigrams would never appear — making it impossible to learn from.

**Block (transformer block)** — One layer of the transformer stack: `LayerNorm → Self-Attention → residual → LayerNorm → MLP → residual`. Implemented as `Block` in `model.py`. The model is `n_layer` of these stacked.

**`block_size`** — Maximum sequence length the model can process. Default 256. Sets the size of the position-embedding table and the attention window. Generation longer than `block_size` must crop earlier tokens.

**BPE (Byte Pair Encoding)** — A subword tokenization algorithm. Repeatedly merges the most-common adjacent token pair into a new token, building a vocabulary tuned to the training corpus. GPT-2 uses BPE with 50,257 tokens. We do **not** use BPE in the workshop — it's overkill for 1 MB of Shakespeare. See `tiktoken`.

**Causal mask** — In self-attention, the mask that hides future positions. Token at position *i* can only attend to positions `0..i`. Implemented in our code via `is_causal=True` in `scaled_dot_product_attention`. This is what makes a transformer GPT-style (decoder-only) instead of BERT-style (encoder).

**Checkpoint** — A saved snapshot of the model state. Includes `model_state_dict`, `config`, `stoi`, `itos`, and the step number. Saved by `torch.save({...}, "checkpoint_NNNN.pt")` and loaded with `torch.load(..., weights_only=False)`.

**Colab (Google Colab)** — A free Jupyter notebook environment with optional GPU/TPU access. Useful when you don't have a local GPU. Free tier gives you an NVIDIA T4. See [`03-setup.md → Step 8`](03-setup.md#8-optional-enable-gpu-on-google-colab).

**Cosine decay** — A learning-rate schedule that smoothly decreases the LR following a half-cosine curve from `max_lr` down to `min_lr` over `max_steps - warmup_steps` steps. Implemented in `get_lr` in `train.py`.

**CPU / CUDA / MPS** — The three device backends `torch` can use. CUDA is NVIDIA GPUs, MPS is Apple Silicon GPUs, CPU is the fallback. `get_device()` in `train.py` auto-selects in that order: MPS → CUDA → CPU.

**Cross-entropy loss** — The standard loss for classification. Measures how surprised the model is by the correct label. For next-token prediction with vocab size 65, a uniformly-random model has cross-entropy `ln(65) ≈ 4.17` — your initial loss.

**Decoder-only** — A transformer that uses only the decoder half of the original encoder-decoder architecture from "Attention Is All You Need". Combined with causal masking, this is the GPT family. (Encoder-only is BERT; encoder-decoder is T5/the original Transformer.)

**Embedding** — A learned mapping from a discrete index (token ID, position) to a dense vector. We have two: `wte` (token embeddings, shape `[vocab_size, n_embd]`) and `wpe` (position embeddings, shape `[block_size, n_embd]`).

**Epoch** — One full pass through the training data. We don't actually train in epochs; we sample random `(batch_size, block_size)` windows from the corpus. With 1 M chars, batch=64, block=256, one "epoch-equivalent" is `1_000_000 / (64 * 256) ≈ 60` steps — so 5,000 steps is ~80 epochs.

**Feed-forward / FFN / MLP** — The two-linear-layer subnetwork inside each transformer block: `Linear(C, 4C) → GELU → Linear(4C, C)`. Operates on each position independently. In our code, the class is named `MLP` but you'll see `FFN` in some other codebases.

**GELU** — Gaussian Error Linear Unit. A smooth alternative to ReLU used in GPT-2. We use the tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))`.

**GPT (Generative Pre-trained Transformer)** — The model family this workshop builds. Decoder-only, autoregressive, trained with next-token prediction.

**Gradient clipping** — `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` rescales gradients so their global L2 norm is ≤ 1.0. Prevents occasional huge gradients from blowing up the weights. Standard transformer-training practice.

**`head_dim`** — `n_embd / n_head`. The dimension each attention head operates in. For default config: `384 / 6 = 64`.

**`itos`** — `int → str` dict mapping token IDs back to characters. Built in `load_data` as `{i: c for c, i in stoi.items()}`. Used during generation to detokenize.

**LayerNorm** — Normalizes a tensor along its last dimension to have mean 0 and variance 1, then applies a learned per-feature scale and bias. Stabilizes activations between layers. Two per `Block` (`ln_1`, `ln_2`); one final (`ln_f`).

**Learning rate (`lr`)** — Step size for the optimizer. Too high → diverges; too low → barely learns. Our schedule: ramp from ~0 to `1e-3` over the first 100 steps, then cosine decay to `1e-4` by step 5000.

**Logits** — Pre-softmax scores. `model.forward()` returns shape `(B, T, vocab_size)`. Each `logits[b, t, v]` is "how confident the model is that token `v` should follow the prefix at batch `b`, position `t`". Softmax converts logits to probabilities.

**Loss** — A scalar measuring how wrong the model is. We use cross-entropy. For our setup, expected initial loss is `ln(65) ≈ 4.17`; "good" final val loss is around 1.5-1.7.

**`lm_head`** — The final `Linear(n_embd, vocab_size, bias=False)` that converts the residual stream to logits over the vocabulary. Its weight matrix is **tied** to the token embedding matrix (`wte`).

**MPS (Metal Performance Shaders)** — Apple's GPU backend. `torch.device("mps")` runs PyTorch ops on Apple Silicon GPUs. Roughly 2-3× faster than CPU on a MacBook Pro for our workload.

**Multi-head attention** — Running several independent attention computations in parallel and concatenating their outputs. Lets the model attend to different relationships simultaneously (e.g., one head tracks line breaks, another tracks character names).

**Multinomial sampling** — `torch.multinomial(probs, num_samples=1)` draws one token from a probability distribution over the vocab. Used in `generate.py` to pick the next token after softmax.

**`n_embd`** — Embedding dimension. The "width" of the model — every hidden state is a vector of `n_embd` floats. Default 384.

**`n_head`** — Number of attention heads per layer. Default 6. Must divide `n_embd`.

**`n_layer`** — Number of transformer blocks stacked. Default 6. The "depth" of the model.

**Next-token prediction** — The objective. Given tokens `[t_0, t_1, …, t_n-1]`, predict `[t_1, t_2, …, t_n]`. The model produces a distribution at every position, and cross-entropy compares to the actual next token.

**Optimizer** — The algorithm that updates weights from gradients. We use AdamW.

**Overfitting** — When the model memorizes training data instead of learning general patterns. Detected by train loss continuing to decrease while val loss starts increasing. Almost guaranteed for our 10M-param model on 1MB of Shakespeare.

**`pos_emb` / `wpe`** — Position embeddings. A learned vector per position `0..block_size-1`. Added to token embeddings so the model knows where each token sits in the sequence.

**Pre-norm / Post-norm** — Two ways to place LayerNorm in a block. Pre-norm (what we use): `x + sublayer(LN(x))`. Post-norm (original 2017 paper): `LN(x + sublayer(x))`. Pre-norm is more stable for deep models; standard since GPT-2.

**Q / K / V (Query, Key, Value)** — The three projections inside attention. For each token: `q` asks a question, `k` advertises content, `v` is the content itself. Attention weights are `softmax(Q @ K^T / sqrt(d))`, applied to `V`.

**Residual connection** — `x = x + sublayer(x)`. Lets gradients flow directly through the network during backprop. Without it, deep networks become untrainable.

**Scaled dot-product attention** — The core attention computation: `softmax(Q @ K^T / sqrt(head_dim)) @ V`. The `sqrt(head_dim)` keeps softmax inputs in a reasonable range. Implemented by `torch.nn.functional.scaled_dot_product_attention`, which uses fused/optimized kernels when available.

**Scratchpad** — The `scratchpad/` directory at the project root. Where you write your `model.py`, `train.py`, and `generate.py`. It is in `.gitignore` so your in-progress code isn't committed.

**Seed** — An integer that initializes the RNG. `torch.manual_seed(42)` makes all subsequent random operations deterministic. Used to reproduce generation outputs (and required for the workshop competition).

**Self-attention** — Attention where Q, K, and V all come from the same input sequence (each token attends to other tokens in the same sequence). Cross-attention (used in encoder-decoder models) has K, V come from a different source than Q.

**Softmax** — Converts a vector of logits into a probability distribution: `softmax(x)_i = exp(x_i) / sum_j exp(x_j)`.

**`stoi`** — `str → int` dict mapping characters to token IDs. Built in `load_data` from `sorted(set(text))`. Used during encoding (text → tokens).

**Step** — One iteration of the training loop: forward, backward, optimizer update. Default 5,000 steps. Don't confuse "step" with "epoch" — they're different units.

**Temperature** — A divisor applied to logits before softmax during generation. Lower → more deterministic, higher → more random. Default 0.8. At T → 0 you get greedy decoding; at T → ∞ you get uniform random.

**Tiktoken** — OpenAI's BPE tokenizer library. Listed as a dependency for the optional competition phase but unused in the core workshop.

**Token** — One unit the model processes. In this workshop, one token = one character. With BPE it's a subword unit.

**Top-k sampling** — A generation filter. Keep only the top-k most likely tokens; set everything else to `-inf` before softmax. Default 40. With our vocab of 65, `top_k=40` is mild; with vocab of 50k, `top_k=40` is aggressive.

**`tqdm`** — A progress-bar library. The training loop wraps `range(max_steps)` in `tqdm(...)` to show throughput, ETA, and live-updated `loss` and `lr`.

**Validation set** — The 10% of `shakespeare.txt` (the last ~111k chars) the model never trains on. Used to compute `val_loss`, which detects overfitting. Train/val split is contiguous, not shuffled — this is fine for character-level Shakespeare but not for general datasets.

**`vocab_size`** — Number of unique tokens. For Shakespeare char-level it's 65. Determines the size of the embedding table (`vocab_size × n_embd`) and the output layer (`n_embd × vocab_size`).

**Warmup** — The first phase of the LR schedule. LR ramps linearly from ~0 to `max_lr` over `warmup_steps` (default 100). Gives Adam's moment estimates time to stabilize before large updates.

**Weight decay** — L2-style regularization that shrinks parameters every step. With AdamW, this is applied directly (not via the gradient). Default `weight_decay=0.01`.

**Weight tying** — Sharing the parameter tensor between the token embedding (`wte`) and the output projection (`lm_head`). Done in `model.py` with `self.transformer.wte.weight = self.lm_head.weight`. Saves parameters and forces input/output token representations to be consistent.

**`wpe`** — *word position embedding.* The position-embedding matrix (`block_size × n_embd`).

**`wte`** — *word token embedding.* The token-embedding matrix (`vocab_size × n_embd`). Tied with `lm_head`.

---

## Symbols cheat sheet

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| `B` | batch size | 64 |
| `T` | sequence length | 256 (= `block_size`) |
| `C` | embedding dimension | 384 (= `n_embd`) |
| `V` | vocabulary size | 65 |
| `H` | number of heads | 6 |
| `D` | head dimension | 64 (= `C/H`) |
| `L` | number of layers | 6 |

These show up in tensor-shape comments throughout the code: `x: (B, T, C)`, `logits: (B, T, V)`, etc.

## Next

- See how all these terms compose in code: [05-architecture.md](05-architecture.md)
- Or fix something that broke: [08-troubleshooting.md](08-troubleshooting.md)
