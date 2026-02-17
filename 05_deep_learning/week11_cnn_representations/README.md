Week 11 — Representation Learning (CNNs)

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
