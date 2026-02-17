Week 09 — PyTorch Fundamentals

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
