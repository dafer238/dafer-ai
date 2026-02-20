Week 14 — Transformers

## Prerequisites

- **Week 13** — scaled dot-product attention implemented from scratch. The transformer is attention + positional encoding + feed-forward layers + residual connections, assembled from components. Week 13 gives you the hardest component.
- **Week 09–10** — PyTorch training loops; the transformer encoder you build here is a `nn.Module` trained with the same pipeline.
- **Week 12** — LayerNorm is the transformer's normalisation of choice; its relationship to BatchNorm was covered there.

## What this week delivers

Transformers are the dominant architecture in NLP, vision, and increasingly time-series modelling. Building one from scratch — even a minimal version — removes the black-box nature. After this week you will be able to read the "Attention Is All You Need" paper and understand every component. Week 15 fine-tunes pretrained transformers, which only makes sense if you know what is inside them.

Overview
Study the transformer architecture, positional encodings, multi-head attention, and practical scaling considerations.

Study
- Transformer blocks (self-attention, feed-forward), positional encodings
- Multi-head attention and parallelization

Practical libraries & tools
- PyTorch / HuggingFace Transformers for reference implementations

Datasets & examples
- Small language modeling tasks (toy text) or time-series forecasting examples

Exercises
1) Build a minimal transformer encoder (single layer) and train on a toy task.

2) Experiment with positional encodings (sinusoidal vs learned).

3) Profile memory and compute for different sequence lengths and head counts.

Reading
- "Attention Is All You Need" and accessible walkthrough blogs; HuggingFace tutorials.

Deliverable
- Notebook implementing a minimal transformer, experiments with positional encodings, and a short profiling report.

## What comes next

- **Week 15** fine-tunes a pretrained transformer from HuggingFace on a downstream task. The architecture you built this week is what you are adapting.
- **Week 16** (Capstone) — the transformer or attention-based model is the most likely candidate for your capstone deployment.
