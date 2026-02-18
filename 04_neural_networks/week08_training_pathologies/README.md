Week 08 — Training Pathologies

## Prerequisites

- **Week 07** — neural networks from scratch. You need to have implemented backprop to understand why gradients vanish or explode across layers.

## What this week delivers

Knowing that a network trains badly is not the same as knowing _why_. This week gives you practical diagnostics: gradient norm tracking, activation distribution visualization, and a toolbox of fixes (He init, BatchNorm, gradient clipping). These diagnostic skills carry through to every deep learning week.

Overview
Study common failure modes (vanishing/exploding gradients, saturated activations) and practical fixes (initialization, normalization, architecture choices).

Study
- Vanishing/exploding gradients, activation saturation, gradient clipping
- BatchNorm, LayerNorm, and initialization strategies

Practical libraries & tools
- NumPy/PyTorch for experiments; TensorBoard/Matplotlib for tracking gradients

Datasets & examples
- Train small nets on toy tasks (deep MLP on MNIST subset or synthetic tasks) to observe pathologies

Exercises
1) Track gradient norms across layers
   - Log gradient L2 norms per layer during training and plot trends.

2) Activation comparisons
   - Compare sigmoid/tanh/ReLU/LeakyReLU and their effect on gradient propagation.

3) Apply fixes
   - Try Xavier/He init, BatchNorm, gradient clipping and measure improvements.

Code hint
  # compute gradient norms
  norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]

Reading
- Papers on initialization, BatchNorm paper, and practical notes on training stability.

Deliverable
- Notebook demonstrating diagnosis and fixes for at least one pathology with plots and a short write-up.

## What comes next

- **Week 09** moves to PyTorch, where gradient norm monitoring is done with one line: `[p.grad.norm() for p in model.parameters()]`. The concepts are identical; only the syntax changes.
- **Week 12** revisits BatchNorm in the context of CNNs and deep regularization.
