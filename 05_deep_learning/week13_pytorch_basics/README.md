Week 09 — PyTorch Fundamentals

## Prerequisites

- **Week 07–08** — neural network from scratch in NumPy. This week's central exercise is re-implementing that exact network in PyTorch and verifying identical outputs. You must have the NumPy version first.

## What this week delivers

PyTorch is the tool every remaining week uses. After Week 09 you should find working in PyTorch as natural as working in NumPy. The key insight is that autograd is just the chain rule you derived in Week 07 — it is now automated.

Overview
Get practical with a modern DL framework: tensors, autograd, `nn.Module`, and training loops. Re-implement a from-scratch model using PyTorch primitives.

Study
- Tensors, broadcasting, autograd internals, computational graphs
- `torch.nn`, `DataLoader`, optimizers, and checkpointing

Practical libraries & tools
- PyTorch, torchvision (small datasets), TensorBoard or `torch.utils.tensorboard`

Datasets & examples
- MNIST / FashionMNIST small experiments; tiny custom regression dataset for verification

Exercises
1) Re-implement your from-scratch NN in PyTorch and verify identical behavior on small data.

2) Compare gradients
   - Compare autograd gradients to your numeric gradient checks.

3) Checkpointing and saving
   - Save/load model state_dict and optimizer state; resume training.

Code hint
  import torch
  x = torch.randn(16, 10, requires_grad=True)
  y = x.sum()
  y.backward()

Reading/resources
- Official PyTorch tutorials (60-minute blitz) and docs on autograd

Deliverable
- Notebook showing PyTorch rewrite, gradient comparisons, and checkpointing example.

## What comes next

- **Week 10** adds DataLoaders, schedulers, and efficient training — the production harness around the model you built this week.
- **Week 11–16** all use the PyTorch patterns established here.
