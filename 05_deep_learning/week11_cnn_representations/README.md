Week 11 — Representation Learning (CNNs)

## Prerequisites

- **Week 09–10** — PyTorch fluency and efficient training pipelines. CNNs are `nn.Module`s trained with the same loop you built there.
- **Week 00a** — the concept of _inductive bias_ (architecture encodes assumptions about data structure) introduced in the family tree section.
- **Week 07** — fully connected NNs. A CNN is a constrained MLP with weight sharing; understanding the unconstrained version makes the motivation for convolution clear.

## What this week delivers

Convolutions are weight-shared linear operations — not magic. This week makes that concrete by building CNNs, visualizing their learned filters, and studying what "representation" means in practice. The transfer learning preview here is expanded fully in Week 15.

Overview
Understand convolutional inductive biases and how hierarchical features emerge. Practice building CNNs and inspecting learned features.

Study
- Convolutions, receptive fields, pooling, and feature hierarchies
- Transfer learning basics and pretrained features

Practical libraries & tools
- PyTorch / torchvision, matplotlib for visualizations
- Tools to visualize activations (e.g., hooks in PyTorch)

Datasets & examples
- CIFAR-10 subset, small custom sensor-window datasets for 1D convs

Exercises
1) Build a small CNN and visualize filters and activations

2) Transfer learning
   - Extract features from a pretrained model and train a small classifier on top.

3) Ablation studies
   - Vary depth/width and document representation changes.

Code hint
  # hook example to capture activations
  activations = {}
  def hook(mod, inp, out): activations['layer'] = out.detach()
  handle = layer.register_forward_hook(hook)

Reading
- CS231n convolutional networks notes and transfer learning tutorials.

Deliverable
- Notebook building CNNs, visualizing filters/activations, and a short ablation study report.

## What comes next

- **Week 12** adds Dropout, BatchNorm, and data augmentation to the CNN built here; the ablation discipline introduced this week continues.
- **Week 15** revisits the pretrained-feature extraction exercise from this week and formalises it as transfer learning with adapters.
