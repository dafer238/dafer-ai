Week 13 — Sequential Models & Attention

## Prerequisites

- **Week 09–10** — PyTorch fluency; you will implement attention as a `torch.nn.Module`.
- **Week 00a** — the problem taxonomy entry on "sequential data" and the note that RNNs exist as a precursor to transformers.

## What this week delivers

Attention is the building block of transformers (Week 14). Implementing it from scratch in this week, on small toy tasks, means Week 14 is assembling known components rather than absorbing an alien architecture.

Overview
Learn attention mechanisms and their role in modeling sequences. Build small attention modules and visualize weight patterns.

Study
- Scoring functions (dot-product, additive), scaled dot-product attention
- Sequence modeling basics: RNNs vs attention

Practical libraries & tools
- NumPy/PyTorch for implementations; visualization libraries for attention maps

Datasets & examples
- Synthetic sequence tasks (copy task, adding problem) and small time-series datasets

Exercises
1) Implement scaled dot-product attention and visualise attention matrices.

2) Build a small seq-to-seq model using attention and test on toy tasks.

3) Compare RNNs vs attention on learning long-range dependencies.

Code hint
  # attention weights (simplified)
  scores = Q @ K.T / np.sqrt(dk)
  weights = softmax(scores, axis=-1)
  out = weights @ V

Reading
- Attention is All You Need (attention sections) and practical tutorials.

Deliverable
- Notebook implementing attention, attention visualizations, and experiments contrasting RNNs and attention.

## What comes next

- **Week 14** assembles multiple attention heads + positional encoding + feed-forward blocks into a full transformer.
