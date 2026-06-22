# Spider Web SLM

A small language model built from scratch that uses **no token-to-token self-attention**. Instead of attention, tokens are routed through a fixed topology of processing nodes using a routing signal derived from the **Lorenz-63 chaotic system**, and context is carried in a persistent **orbital memory** module rather than a growing key–value cache.

This is a **proof-of-concept for an alternative architecture class** — a demonstration that a non-attention, chaos-routed model can learn coherent language at small scale. It is not a state-of-the-art language model, and it does not claim to match transformer perplexity at this size. The point is to explore whether the inductive biases of attention are necessary for language modeling, or merely sufficient.

## What's novel here

- **Lorenz Router** — routing decisions (a token stays at its node, moves inward, or exits) are produced by integrating the Lorenz-63 chaotic system, giving an input-sensitive, high-dimensional routing signal from a fixed, parameter-free dynamical system. A Jacobian-guard term keeps the token's identity recoverable through the chaotic transformation.
- **Solar Ring Memory (SRM)** — a persistent, slot-based memory module whose cost is **constant in sequence length**, in contrast to the linearly growing KV cache of a transformer. Context is accumulated into a fixed set of slots and updated in place.
- **3-Axis Polar RoPE** — positional encoding generalized from a single sequence axis to three: temporal (sequence position), angular (node position within a ring), and radial (ring depth), reflecting that a token occupies a position in a topology, not just a sequence.
- **Node topology** — 4 rings × 8 nodes = 32 processing units, with node-to-node lateral attention *within* a ring (not over the token sequence), SwiGLU feed-forward blocks, and spectral normalization on projections for stability.

## Results

Trained from scratch on [TinyStories](https://arxiv.org/abs/2305.07759) at 3.48M parameters (dim=64, hidden=256, vocab=5000, seq_len=128):

| Metric | Value |
|---|---|
| Average eval CE (50 batches) | 4.77 |
| Best checkpoint CE | 4.42 |
| Random baseline (ln 5000) | 8.52 |
| Training steps | 20,000 |
| Hardware | Single RTX 5050 (8GB) |

The model generates coherent TinyStories-style text — named characters, story structure, cause-and-effect — with grammatical imperfections consistent with its scale and loss level. Sample output (prompt in **bold**):

> **Once upon a time**, and said he had to the park. They were happy and her mommy was so happy for his friends and the big man in the old and they went to be proud of a time...

This is clearly a different class of output from an untrained or broken model (which produces disconnected word fragments).

## A note on training: bfloat16 freezes normalization layers

During development we found a subtle but important bug worth documenting, because it applies to any heavily-normalized architecture. Training naively in bfloat16 caused **every RMSNorm weight to freeze at its initial value of 1.0**. The reason: an AdamW update of ~2e-4 is roughly 39× below bfloat16's unit of least precision at magnitude 1.0 (~7.8e-3), so the update rounds to zero on every step. This architecture has far more normalization than a standard transformer (3 RMSNorms per node × 32 nodes, plus a final norm), so it is unusually exposed to this failure — the result was a model that plateaued far above its true floor.

The fix is standard mixed precision done correctly: **keep float32 master weights** and use `autocast(bfloat16)` only around the forward/loss compute, so sub-epsilon optimizer updates accumulate at full precision. After the fix, the norm weights train freely (final-norm weights drifted ~270% from initialization over training).

## Repository structure

```
core/
  web.py        # SpiderWeb — the full model
  node.py       # WebNode — per-node processing unit
  lorenz.py     # LorenzRouter — chaos-based routing
  memory.py     # SolarRingMemory — orbital persistent memory
  rope.py       # 3-axis Polar RoPE
  tokenizer.py  # tokenizer wrapper
train/
  loss.py       # CE + routing entropy + load-balance loss
  scheduler.py  # cosine LR + temperature annealing
  curriculum.py # entropy/routing schedule
  dataloader.py # TinyStories data pipeline
config.py       # model + training configuration
train_main.py   # training entry point
infer.py        # text generation
tests/          # component tests
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train (expects TinyStories text in data/raw/ and a SentencePiece tokenizer in data/)
python train_main.py

# Generate text from a checkpoint
python infer.py --prompt "Once upon a time" --max_new_tokens 80 --temperature 0.7
```

Note: trained checkpoints and the dataset are not included in the repository (size). The tokenizer model is included for reproducibility.

## Status and limitations

- Single dataset (TinyStories); generality to other text is untested.
- Does not match same-scale transformer perplexity — this is a proof of concept, not a competitive model.
- No efficiency benchmark against transformers; the constant-memory property of SRM is structural (by construction) and not yet measured empirically.
- Component ablations (chaos router vs. linear router; with/without SRM) are in progress and will quantify how much each novel component contributes.

## License

MIT

