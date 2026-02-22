# PyTorch Fundamentals

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Tensors](#2-tensors)
    - 2.1 [Creating Tensors](#21-creating-tensors)
    - 2.2 [Data Types](#22-data-types)
    - 2.3 [Shape Manipulation](#23-shape-manipulation)
    - 2.4 [Indexing and Slicing](#24-indexing-and-slicing)
    - 2.5 [Broadcasting](#25-broadcasting)
3. [Automatic Differentiation (Autograd)](#3-automatic-differentiation-autograd)
    - 3.1 [The Computational Graph](#31-the-computational-graph)
    - 3.2 [Forward and Backward](#32-forward-and-backward)
    - 3.3 [Gradient Accumulation](#33-gradient-accumulation)
    - 3.4 [Detaching and No-Grad Context](#34-detaching-and-no-grad-context)
    - 3.5 [Autograd Is Just the Chain Rule](#35-autograd-is-just-the-chain-rule)
4. [Building Models with nn.Module](#4-building-models-with-nnmodule)
    - 4.1 [The Module Contract](#41-the-module-contract)
    - 4.2 [Parameters and Buffers](#42-parameters-and-buffers)
    - 4.3 [Common Layer Types](#43-common-layer-types)
    - 4.4 [nn.Sequential](#44-nnsequential)
5. [Loss Functions](#5-loss-functions)
    - 5.1 [Cross-Entropy Loss](#51-cross-entropy-loss)
    - 5.2 [MSE Loss](#52-mse-loss)
    - 5.3 [Choosing a Loss Function](#53-choosing-a-loss-function)
6. [Optimisers](#6-optimisers)
    - 6.1 [SGD](#61-sgd)
    - 6.2 [Adam](#62-adam)
    - 6.3 [The Optimiser API](#63-the-optimiser-api)
7. [Data Loading: Dataset and DataLoader](#7-data-loading-dataset-and-dataloader)
    - 7.1 [The Dataset Interface](#71-the-dataset-interface)
    - 7.2 [TensorDataset](#72-tensordataset)
    - 7.3 [Custom Datasets](#73-custom-datasets)
    - 7.4 [DataLoader](#74-dataloader)
8. [The Training Loop](#8-the-training-loop)
    - 8.1 [Anatomy of One Step](#81-anatomy-of-one-step)
    - 8.2 [Training vs. Evaluation Mode](#82-training-vs-evaluation-mode)
    - 8.3 [A Complete Training Function](#83-a-complete-training-function)
    - 8.4 [Why `optimizer.zero_grad()` Is Necessary](#84-why-optimizerzero_grad-is-necessary)
9. [Checkpointing: Save and Load](#9-checkpointing-save-and-load)
    - 9.1 [What to Save](#91-what-to-save)
    - 9.2 [Saving and Loading](#92-saving-and-loading)
    - 9.3 [Resuming Training](#93-resuming-training)
10. [Device Management: CPU and GPU](#10-device-management-cpu-and-gpu)
11. [Learning Rate Schedulers](#11-learning-rate-schedulers)
12. [From NumPy to PyTorch: A Translation Table](#12-from-numpy-to-pytorch-a-translation-table)
13. [Connections to the Rest of the Course](#13-connections-to-the-rest-of-the-course)
14. [Notebook Reference Guide](#14-notebook-reference-guide)
15. [Symbol Reference](#15-symbol-reference)
16. [References](#16-references)

---

## 1. Scope and Purpose

[[Weeks 11](../../04_neural_networks/week11_nn_from_scratch/theory.md)](../../04_neural_networks/week11_nn_from_scratch/theory.md)–[12](../../04_neural_networks/week12_training_pathologies/theory.md) built and diagnosed neural networks using NumPy. Starting this week, **every remaining week uses PyTorch**. The goal is not to learn a new framework for its own sake — it is to automate the two things that make hand-coded deep learning painful:

1. **Gradient computation.** PyTorch's autograd implements the chain rule ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md)) automatically for arbitrary computational graphs.
2. **Boilerplate management.** Parameter tracking, mini-batching, GPU transfer, checkpointing — all standardised.

After this week you should find writing a PyTorch model as natural as writing NumPy. The concepts are identical; only the tooling changes.

**Prerequisites.** [Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md) (backpropagation, chain rule), [Week 12](../../04_neural_networks/week12_training_pathologies/theory.md) (training pathologies — you'll use `nn.BatchNorm1d`, gradient clipping, etc. in later weeks via their PyTorch APIs).

---

## 2. Tensors

A **tensor** is a multi-dimensional array — the fundamental data structure in PyTorch. It behaves almost identically to a NumPy `ndarray`, with two additions: it can live on a GPU, and it can track gradients.

### 2.1 Creating Tensors

```python
import torch

# From Python data
x = torch.tensor([1.0, 2.0, 3.0])            # shape (3,)
M = torch.tensor([[1, 2], [3, 4]])            # shape (2, 2)

# Factory functions
z = torch.zeros(3, 4)                          # shape (3, 4), all 0s
o = torch.ones(2, 3)                           # shape (2, 3), all 1s
r = torch.randn(5, 10)                         # shape (5, 10), standard normal
u = torch.rand(5, 10)                          # shape (5, 10), uniform [0, 1)
e = torch.eye(3)                               # 3×3 identity matrix
a = torch.arange(0, 10, 2)                     # [0, 2, 4, 6, 8]
l = torch.linspace(0, 1, 5)                    # [0.00, 0.25, 0.50, 0.75, 1.00]

# From NumPy (shares memory by default)
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)                      # shared memory — changes in arr affect t
t_copy = torch.tensor(arr)                     # copy — independent
```

> **Notebook reference.** Cell 1 demonstrates tensor creation and basic operations.

---

### 2.2 Data Types

| dtype                            | Description    | When to use                                 |
| -------------------------------- | -------------- | ------------------------------------------- |
| `torch.float32` (`torch.float`)  | 32-bit float   | Default for model weights and inputs        |
| `torch.float64` (`torch.double`) | 64-bit float   | Numerical verification; rarely for training |
| `torch.float16` (`torch.half`)   | 16-bit float   | Mixed-precision training ([Week 14](../week14_training_at_scale/theory.md))          |
| `torch.int64` (`torch.long`)     | 64-bit integer | Class labels for `CrossEntropyLoss`         |
| `torch.int32` (`torch.int`)      | 32-bit integer | Index tensors                               |
| `torch.bool`                     | Boolean        | Masks                                       |

```python
x = torch.tensor([1.0, 2.0])       # float32 by default
x_long = x.long()                   # cast to int64
x_half = x.half()                   # cast to float16
x_explicit = torch.tensor([1, 2], dtype=torch.float32)
```

> **Common bug:** `CrossEntropyLoss` requires targets of type `torch.long`. Passing `float32` targets causes a runtime error.

---

### 2.3 Shape Manipulation

| Operation   | Code                                | Notes                                                     |
| ----------- | ----------------------------------- | --------------------------------------------------------- |
| Reshape     | `x.view(2, 3)` or `x.reshape(2, 3)` | `view` requires contiguous memory; `reshape` always works |
| Flatten     | `x.view(-1)` or `x.flatten()`       | `-1` infers the dimension                                 |
| Squeeze     | `x.squeeze(dim)`                    | Remove dimension of size 1                                |
| Unsqueeze   | `x.unsqueeze(dim)`                  | Add dimension of size 1                                   |
| Transpose   | `x.T` or `x.transpose(0, 1)`        | 2-D transpose                                             |
| Permute     | `x.permute(2, 0, 1)`                | Arbitrary axis reordering                                 |
| Stack       | `torch.stack([a, b], dim=0)`        | Creates new dimension                                     |
| Concatenate | `torch.cat([a, b], dim=0)`          | Along existing dimension                                  |

```python
x = torch.randn(3, 4)         # shape (3, 4)
x_flat = x.view(-1)           # shape (12,)
x_3d = x.unsqueeze(0)         # shape (1, 3, 4)
x_back = x_3d.squeeze(0)      # shape (3, 4)
```

---

### 2.4 Indexing and Slicing

PyTorch indexing mirrors NumPy:

```python
x = torch.randn(5, 10)

x[0]            # first row, shape (10,)
x[:, 0]         # first column, shape (5,)
x[1:3, 4:7]     # rows 1–2, columns 4–6, shape (2, 3)
x[x > 0]        # boolean indexing, 1-D result

# Advanced indexing
idx = torch.tensor([0, 2, 4])
x[idx]           # rows 0, 2, 4, shape (3, 10)
```

---

### 2.5 Broadcasting

PyTorch follows NumPy broadcasting rules. Two tensors are broadcastable if, for each trailing dimension, the sizes are equal or one of them is 1.

**Rules (applied right-to-left):**

1. If tensors have different numbers of dimensions, prepend 1s to the shape of the smaller.
2. For each dimension, if sizes differ, the size-1 dimension is stretched to match.
3. If neither is 1 and they differ, broadcasting fails.

**Example:**

```
A: shape (3, 1)
B: shape    (4,)

Step 1: B → (1, 4)
Step 2: A (3, 1) + B (1, 4) → (3, 4)
```

Formally, if $A \in \mathbb{R}^{m \times 1}$ and $B \in \mathbb{R}^{1 \times n}$, then:

$$(A + B)_{ij} = A_{i1} + B_{1j} \qquad \text{for } i = 1,\ldots,m;\; j = 1,\ldots,n$$

This produces an outer-sum of shape $(m, n)$.

> **Notebook reference.** Cell 1 demonstrates broadcasting: a `(3,1)` tensor + a `(3,)` tensor → a `(3,3)` result.

---

## 3. Automatic Differentiation (Autograd)

### 3.1 The Computational Graph

When you create a tensor with `requires_grad=True` and perform operations on it, PyTorch builds a **computational graph** — a directed acyclic graph (DAG) where:

- **Leaf nodes** are input tensors (parameters, data).
- **Internal nodes** are operations (`+`, `*`, `relu`, `matmul`, ...).
- **Edges** represent data flow.

```python
x = torch.tensor(2.0, requires_grad=True)   # leaf
y = x ** 2 + 3 * x + 1                       # y is an internal node
# The graph records: y = Add(Pow(x, 2), Add(Mul(3, x), 1))
```

Each non-leaf tensor stores a reference to its `grad_fn` — the function that created it. This is the entry point for backward traversal.

---

### 3.2 Forward and Backward

**Forward pass:** compute the output from inputs, building the graph.
**Backward pass:** traverse the graph in reverse (topological order), applying the chain rule at each node.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1       # forward

y.backward()                  # backward: compute dy/dx

print(x.grad)                 # tensor(7.) because d/dx(x²+3x+1) = 2x+3 = 7
```

For a scalar loss $\mathcal{L}$ and parameter vector $\boldsymbol{\theta}$:

$$\texttt{loss.backward()} \quad\Longrightarrow\quad \theta\texttt{.grad} = \frac{\partial\mathcal{L}}{\partial\boldsymbol{\theta}}$$

> **Notebook reference.** Cell 1 computes $y = x^2 + 3x + 1$ and verifies that autograd gives $\mathrm{d}y/\mathrm{d}x = 2x + 3 = 7$ at $x = 2$.

---

### 3.3 Gradient Accumulation

**Critical detail:** `tensor.grad` is **accumulated** (summed), not replaced, on each `.backward()` call.

```python
x = torch.tensor(3.0, requires_grad=True)
for _ in range(3):
    y = x * 2
    y.backward()

print(x.grad)   # tensor(6.) — accumulated 2 + 2 + 2
```

This is why `optimizer.zero_grad()` is mandatory before each training step — without it, gradients from the previous step leak into the current step.

---

### 3.4 Detaching and No-Grad Context

Sometimes you need to stop gradient tracking:

```python
# Method 1: detach() — creates a tensor sharing data but detached from the graph
z = y.detach()   # z has no grad_fn, no gradient flows through it

# Method 2: torch.no_grad() context — disables grad tracking for efficiency
with torch.no_grad():
    pred = model(x_test)   # faster, uses less memory (no graph built)

# Method 3: model.eval() — changes layer behaviour (BatchNorm, Dropout)
# but does NOT disable gradient tracking; combine with torch.no_grad()
```

| Mechanism         | Disables grad tracking? | Changes layer behaviour? | When to use                    |
| ----------------- | ----------------------- | ------------------------ | ------------------------------ |
| `detach()`        | Yes (one tensor)        | No                       | Isolate a value from the graph |
| `torch.no_grad()` | Yes (all ops in block)  | No                       | Inference / evaluation         |
| `model.eval()`    | No                      | Yes (BN, Dropout)        | Always before evaluation       |

> **Best practice for evaluation:** use both `model.eval()` and `torch.no_grad()`.

---

### 3.5 Autograd Is Just the Chain Rule

The key insight connecting this week to [Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md): PyTorch autograd computes exactly the same derivatives you computed by hand. For a two-layer network:

$$\hat{y} = W_2\,\text{relu}(W_1 x + b_1) + b_2$$

$$\frac{\partial\mathcal{L}}{\partial W_1} = \frac{\partial\mathcal{L}}{\partial\hat{y}}\cdot\frac{\partial\hat{y}}{\partial h}\cdot\frac{\partial h}{\partial z_1}\cdot\frac{\partial z_1}{\partial W_1}$$

In [Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md) you computed each factor manually. In PyTorch, `loss.backward()` does this automatically by traversing the graph. The mathematics is identical; only the implementation changes.

> **Notebook reference.** Cell 6 verifies this: for $y = w_1 x_1 + w_2 x_2 + b$, autograd gives $\partial y/\partial w_1 = x_1$, matching the analytic result.

---

## 4. Building Models with nn.Module

### 4.1 The Module Contract

Every PyTorch model inherits from `nn.Module` and must implement:

1. **`__init__`**: define the layers (sub-modules and parameters).
2. **`forward`**: define the computation (how inputs become outputs).

```python
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

**Never call `model.forward(x)` directly.** Use `model(x)`, which calls `forward` plus any registered hooks.

---

### 4.2 Parameters and Buffers

| Attribute     | Registered via                                     | Learned by optimiser? | Example                |
| ------------- | -------------------------------------------------- | --------------------- | ---------------------- |
| **Parameter** | `nn.Parameter(tensor)` or a sub-module's parameter | Yes                   | Weights, biases        |
| **Buffer**    | `self.register_buffer('name', tensor)`             | No                    | BatchNorm running mean |

```python
# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

For a `nn.Linear(10, 50)` layer:

$$\text{Parameters} = \underbrace{10 \times 50}_{W} + \underbrace{50}_{b} = 550$$

For the notebook's `SimpleMLP(10, 50, 2)`:

$$\text{Total} = (10 \times 50 + 50) + (50 \times 2 + 2) = 550 + 102 = 652$$

> **Notebook reference.** Cell 2 builds `SimpleMLP(10, 50, 2)` and prints the parameter count.

---

### 4.3 Common Layer Types

| Layer           | Constructor                | Computation                                          |
| --------------- | -------------------------- | ---------------------------------------------------- |
| Fully connected | `nn.Linear(in, out)`       | $y = Wx + b$                                         |
| ReLU            | `nn.ReLU()`                | $y = \max(0, x)$                                     |
| Sigmoid         | `nn.Sigmoid()`             | $y = 1/(1+e^{-x})$                                   |
| Tanh            | `nn.Tanh()`                | $y = \tanh(x)$                                       |
| Dropout         | `nn.Dropout(p)`            | Zeroes elements with probability $p$ during training |
| BatchNorm       | `nn.BatchNorm1d(features)` | Normalise across batch ([Week 12](../../04_neural_networks/week12_training_pathologies/theory.md))                     |
| LayerNorm       | `nn.LayerNorm(features)`   | Normalise across features ([Week 12](../../04_neural_networks/week12_training_pathologies/theory.md))                  |

Activations can also be used as functions:

```python
import torch.nn.functional as F

x = F.relu(self.fc1(x))      # functional form (stateless)
# vs.
x = self.relu(self.fc1(x))   # module form (same result)
```

Use `nn.Module` form when the layer has state (Dropout, BatchNorm). Use `F.` form for pure functions if you prefer.

---

### 4.4 nn.Sequential

For simple linear stacks of layers:

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)
```

Equivalent to writing a custom `nn.Module` with these layers in `forward`. Use `nn.Sequential` for simple cases; write a custom module when you need branching, skip connections, or custom logic.

---

## 5. Loss Functions

### 5.1 Cross-Entropy Loss

For classification with $C$ classes. PyTorch's `nn.CrossEntropyLoss` combines `log_softmax` and `NLLLoss` in a single step:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{z_{i, y_i}}}{\sum_{c=1}^{C}e^{z_{i,c}}}$$

where $z_{i,c}$ are the raw **logits** (model outputs before softmax) and $y_i$ is the true class index.

```python
criterion = nn.CrossEntropyLoss()

logits = model(x_batch)           # shape (N, C), raw scores
loss = criterion(logits, y_batch) # y_batch shape (N,), dtype=torch.long
```

> **Important:** do NOT apply softmax before `CrossEntropyLoss`. The loss function expects raw logits and applies `log_softmax` internally for numerical stability.

**Equivalence to negative log-likelihood.** For one sample:

$$\mathcal{L}_i = -\log p(y_i\mid x_i) = -\log\text{softmax}(z_i)_{y_i} = -z_{i, y_i} + \log\sum_{c=1}^{C}e^{z_{i,c}}$$

This is exactly the NLL from [Week 07](../../03_probability/week07_likelihood/theory.md) (Likelihood), now applied to neural network outputs.

---

### 5.2 MSE Loss

For regression:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

```python
criterion = nn.MSELoss()
loss = criterion(predictions, targets)   # both shape (N, d) or (N,)
```

---

### 5.3 Choosing a Loss Function

| Task                       | Loss function                        | Model output                |
| -------------------------- | ------------------------------------ | --------------------------- |
| Binary classification      | `nn.BCEWithLogitsLoss()`             | 1 logit per sample          |
| Multi-class classification | `nn.CrossEntropyLoss()`              | $C$ logits per sample       |
| Regression                 | `nn.MSELoss()`                       | 1 or more continuous values |
| Regression (robust)        | `nn.L1Loss()` or `nn.SmoothL1Loss()` | Continuous values           |

---

## 6. Optimisers

### 6.1 SGD

$$\theta_{t+1} = \theta_t - \eta\,\nabla_\theta\mathcal{L}$$

With momentum $\mu$:

$$v_{t+1} = \mu\, v_t + \nabla_\theta\mathcal{L}, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}$$

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

### 6.2 Adam

Combines momentum and per-parameter adaptive learning rates ([[Weeks 01](../../02_fundamentals/week01_optimization/theory.md)](../../02_fundamentals/week01_optimization/theory.md)–[02](../../02_fundamentals/week02_advanced_optimizers/theory.md)):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
# defaults: betas=(0.9, 0.999), eps=1e-8, weight_decay=0
```

> **Practical default:** Adam with `lr=0.001` is a safe starting point for most problems. SGD with momentum can generalise better but requires more tuning.

---

### 6.3 The Optimiser API

All optimisers share the same interface:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)

# In the training loop:
optimizer.zero_grad()     # 1. Clear old gradients
loss.backward()           # 2. Compute new gradients (populates .grad)
optimizer.step()          # 3. Update parameters using .grad
```

The three calls always appear in this order. Swapping them causes bugs:
- `.step()` before `.backward()` → updates using stale gradients.
- Forgetting `.zero_grad()` → gradients accumulate across steps.

---

## 7. Data Loading: Dataset and DataLoader

### 7.1 The Dataset Interface

Any custom dataset must implement:

```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, ...):
        # Load / store data
        pass

    def __len__(self):
        return N          # total number of samples

    def __getitem__(self, idx):
        return x, y       # one (input, target) pair
```

---

### 7.2 TensorDataset

The simplest case — when data is already in tensors:

```python
from torch.utils.data import TensorDataset

X = torch.FloatTensor(X_np)
y = torch.LongTensor(y_np)
dataset = TensorDataset(X, y)
# dataset[i] returns (X[i], y[i])
```

---

### 7.3 Custom Datasets

For real-world data that doesn't fit in memory, or that requires on-the-fly transforms:

```python
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = load_image(self.paths[idx])   # load from disk on demand
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
```

> **Notebook reference.** Exercise 1 asks you to implement a custom `MyTabularDataset` class.

---

### 7.4 DataLoader

Wraps a `Dataset` to provide mini-batches, shuffling, and parallel loading:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,       # samples per mini-batch
    shuffle=True,        # randomise order each epoch
    num_workers=0,       # parallel data loading workers (>0 for disk I/O)
    drop_last=False,     # drop last incomplete batch?
)

for batch_x, batch_y in train_loader:
    # batch_x shape: (32, n_features)
    # batch_y shape: (32,)
    ...
```

**Why mini-batches?** ([Week 01](../../02_fundamentals/week01_optimization/theory.md) review)

| Batch size       | Gradient estimate       | Update speed        | Memory                   |
| ---------------- | ----------------------- | ------------------- | ------------------------ |
| $N$ (full batch) | Low variance, high cost | 1 update/epoch      | Entire dataset in memory |
| 1 (stochastic)   | High variance, low cost | $N$ updates/epoch   | 1 sample                 |
| $B$ (mini-batch) | Moderate variance       | $N/B$ updates/epoch | $B$ samples              |

Mini-batch gradient descent (typically $B = 32$–$256$) balances noise and throughput.

> **Notebook reference.** Cell 3 creates `DataLoader` objects with `batch_size=32`, showing `len(train_loader)` = number of batches per epoch.

---

## 8. The Training Loop

### 8.1 Anatomy of One Step

```python
# One gradient descent step
outputs = model(batch_x)          # 1. Forward pass
loss = criterion(outputs, batch_y) # 2. Compute loss
optimizer.zero_grad()              # 3. Zero gradients
loss.backward()                    # 4. Backward pass (compute gradients)
optimizer.step()                   # 5. Update parameters
```

In diagram form:

```
Input → [Forward] → Loss → [Backward] → Gradients → [Step] → Updated Parameters
  ↑                                                              |
  └──────────────────── next batch ──────────────────────────────┘
```

---

### 8.2 Training vs. Evaluation Mode

```python
model.train()    # Enable dropout, use batch statistics for BatchNorm
model.eval()     # Disable dropout, use running statistics for BatchNorm
```

These calls change the behaviour of **stateful layers** like `Dropout` and `BatchNorm`. They do not affect gradient computation — you still need `torch.no_grad()` for that.

**Correct evaluation pattern:**

```python
model.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        ...
```

---

### 8.3 A Complete Training Function

```python
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_x, batch_y in loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / len(loader), correct / total
```

Key details:

- `loss.item()` extracts the scalar value without retaining the graph (avoids memory leak).
- `torch.max(outputs, 1)` returns `(values, indices)` — `indices` are the predicted classes.
- `(predicted == batch_y).sum().item()` counts correct predictions.

> **Notebook reference.** Cell 4 implements `train_epoch` and `evaluate`, training a `SimpleMLP` for 50 epochs with Adam.

---

### 8.4 Why `optimizer.zero_grad()` Is Necessary

PyTorch **accumulates** gradients by default (Section 3.3). Without zeroing:

$$\texttt{param.grad} = \sum_{t=1}^{T}\nabla_\theta\mathcal{L}_t$$

After $T$ steps without zeroing, the gradient reflects the sum of all past batches — not the current batch. This almost always leads to incorrect training behaviour.

> **Exception:** gradient accumulation is intentionally used to simulate larger batch sizes when GPU memory is limited. In that case, you call `optimizer.zero_grad()` every $k$ steps and `optimizer.step()` every $k$ steps.

---

## 9. Checkpointing: Save and Load

### 9.1 What to Save

A complete checkpoint contains:

| Component                | Why                                                          |
| ------------------------ | ------------------------------------------------------------ |
| `model.state_dict()`     | All learned parameters (weights, biases, BN running stats)   |
| `optimizer.state_dict()` | Momentum buffers, adaptive learning rate accumulators (Adam) |
| Epoch number             | To resume from the right point                               |
| Loss / metrics           | For logging and comparison                                   |
| Random state (optional)  | For exact reproducibility: `torch.random.get_rng_state()`    |

---

### 9.2 Saving and Loading

```python
# Save
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'test_loss': test_loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

> **Notebook reference.** Cell 5 saves and loads a checkpoint, verifying that the reloaded model's parameters are identical.

---

### 9.3 Resuming Training

```python
# 1. Create model and optimizer (same architecture and hyperparameters)
model = SimpleMLP(10, 64, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Load checkpoint
ckpt = torch.load('checkpoint.pth')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch']

# 3. Continue training
for epoch in range(start_epoch, start_epoch + 50):
    train_epoch(model, train_loader, criterion, optimizer)
```

> **Why save the optimiser?** Adam maintains per-parameter momentum ($m_t$) and variance ($v_t$) estimates. If you restart with a fresh optimiser, these are reset to zero, causing a "learning rate warm-up" artifact and potentially degraded fine-tuning.

---

## 10. Device Management: CPU and GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model
model = model.to(device)

# Move data (in training loop)
for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    outputs = model(batch_x)
    ...
```

**Rules:**

1. **All tensors in a computation must be on the same device.** A CPU tensor cannot be multiplied with a GPU tensor.
2. **Model and data must be on the same device.** Call `.to(device)` on both.
3. **Move data inside the loop, not before.** DataLoader produces CPU tensors; move each batch as needed.
4. **Use `.item()` or `.cpu()` to bring results back** for printing, logging, or NumPy operations.

```python
# Common pattern
loss_val = loss.item()            # scalar → Python float (CPU)
preds_np = preds.cpu().numpy()    # tensor → NumPy (must be on CPU first)
```

> **Notebook reference.** Exercise 4 asks you to write a device-agnostic training loop.

---

## 11. Learning Rate Schedulers

A **scheduler** adjusts the learning rate during training. PyTorch provides many strategies:

| Scheduler                                     | Formula / behaviour                                                                  | When to use                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------- |
| `StepLR(step_size, gamma)`                    | Multiply LR by $\gamma$ every `step_size` epochs                                     | Simple decay                                        |
| `ExponentialLR(gamma)`                        | $\eta_t = \eta_0\cdot\gamma^t$                                                       | Smooth continuous decay                             |
| `CosineAnnealingLR(T_max)`                    | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t/T_{\max}))$ | Widely used; gentle decay and warm restart friendly |
| `ReduceLROnPlateau(patience)`                 | Reduce LR when metric stops improving for `patience` epochs                          | Adaptive; good default                              |
| `OneCycleLR(max_lr, epochs, steps_per_epoch)` | Warm-up → high LR → anneal                                                           | Super-convergence ([Week 14](../week14_training_at_scale/theory.md))                         |

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(50):
    train_epoch(model, train_loader, criterion, optimizer)
    scheduler.step()         # update LR after each epoch
    current_lr = scheduler.get_last_lr()[0]
```

**Cosine annealing** (often the default in modern training):

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi\, t}{T_{\max}}\right)\right)$$

At $t = 0$: $\eta = \eta_{\max}$. At $t = T_{\max}$: $\eta = \eta_{\min}$. The decay is slow at the start and end, fast in the middle.

> **Notebook reference.** Exercise 2 asks you to add `CosineAnnealingLR` and compare to the constant-LR baseline.

---

## 12. From NumPy to PyTorch: A Translation Table

| NumPy                            | PyTorch                             | Notes                                    |
| -------------------------------- | ----------------------------------- | ---------------------------------------- |
| `np.array([1, 2])`               | `torch.tensor([1, 2])`              |                                          |
| `np.zeros((3, 4))`               | `torch.zeros(3, 4)`                 | No tuple needed                          |
| `np.random.randn(3, 4)`          | `torch.randn(3, 4)`                 |                                          |
| `a @ b` or `np.dot(a, b)`        | `a @ b` or `torch.matmul(a, b)`     |                                          |
| `a.T`                            | `a.T` or `a.t()`                    |                                          |
| `np.sum(a, axis=0)`              | `a.sum(dim=0)`                      | `dim` not `axis`                         |
| `np.max(a, axis=1)`              | `a.max(dim=1)`                      | Returns `(values, indices)`              |
| `np.concatenate([a, b], axis=0)` | `torch.cat([a, b], dim=0)`          |                                          |
| `a.reshape(2, 3)`                | `a.view(2, 3)` or `a.reshape(2, 3)` |                                          |
| `np.exp(a)`                      | `torch.exp(a)`                      |                                          |
| `np.log(a)`                      | `torch.log(a)`                      |                                          |
| —                                | `a.requires_grad_(True)`            | Enable gradient tracking                 |
| —                                | `a.backward()`                      | Compute gradients                        |
| `a` (ndarray)                    | `a.numpy()`                         | Tensor → NumPy (CPU only, shared memory) |
| `a` (ndarray)                    | `torch.from_numpy(a)`               | NumPy → Tensor (shared memory)           |

> **Key difference.** NumPy operations never track gradients. PyTorch operations with `requires_grad=True` tensors build a computational graph. Everything else is nearly interchangeable.

---

## 13. Connections to the Rest of the Course

| Week                                     | Connection                                                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md) (NN from Scratch)**            | The NumPy network you built — PyTorch automates its gradient computation                                     |
| **[Week 12](../../04_neural_networks/week12_training_pathologies/theory.md) (Training Pathologies)**       | `nn.BatchNorm1d`, `nn.init.kaiming_normal_`, `clip_grad_norm_` are one-line PyTorch calls                    |
| **[Week 14](../week14_training_at_scale/theory.md) (Training at Scale)**          | DataLoader workers, LR schedulers, mixed precision, distributed training — all build on this week's patterns |
| **[Week 15](../week15_cnn_representations/theory.md) (CNNs)**                       | `nn.Conv2d`, `nn.MaxPool2d` — same `nn.Module` API, just new layer types                                     |
| **[Week 16](../week16_regularization_dl/theory.md) (Regularisation DL)**          | `nn.Dropout`, weight decay in optimiser, data augmentation in DataLoader transforms                          |
| **[[Week 17](../../06_sequence_models/week17_attention/theory.md)](../../06_sequence_models/week17_attention/theory.md)–[18](../../06_sequence_models/week18_transformers/theory.md) (Attention, Transformers)** | `nn.MultiheadAttention`, custom `forward` with masks — built on Module patterns                              |
| **[Week 19](../../07_transfer_learning/week19_finetuning/theory.md) (Fine-Tuning)**                | Loading pre-trained `state_dict`, freezing parameters, saving checkpoints                                    |
| **[Week 20](../../08_deployment/week20_deployment/theory.md) (Deployment)**                 | `torch.jit`, ONNX export, model serialisation — extensions of `torch.save`                                   |

---

## 14. Notebook Reference Guide

| Cell                    | Section               | What it demonstrates                                                              | Theory reference |
| ----------------------- | --------------------- | --------------------------------------------------------------------------------- | ---------------- |
| 1 (Tensors & Autograd)  | Basics                | Tensor creation, broadcasting `(3,1)+(3,)→(3,3)`, autograd for $y = x^2 + 3x + 1$ | Sections 2, 3    |
| 2 (nn.Module)           | Model building        | `SimpleMLP(10, 50, 2)`, parameter count, forward pass                             | Section 4        |
| 3 (DataLoader)          | Data pipeline         | `TensorDataset`, `DataLoader(batch_size=32)`, train/test split                    | Section 7        |
| 4 (Training loop)       | Full training         | `train_epoch`, `evaluate`, 50 epochs with Adam, loss/accuracy plots               | Section 8        |
| 5 (Checkpointing)       | Save/load             | `torch.save`/`torch.load`, verify parameter equality                              | Section 9        |
| 6 (Gradient comparison) | Autograd verification | Manual vs. autograd for $y = w_1 x_1 + w_2 x_2 + b$                               | Section 3.5      |
| Ex. 1 (Custom Dataset)  | Data pipeline         | Implement `MyTabularDataset(Dataset)`                                             | Section 7.3      |
| Ex. 2 (LR Scheduler)    | Scheduler             | `CosineAnnealingLR(T_max=50)` vs. constant LR                                     | Section 11       |
| Ex. 4 (GPU portability) | Device                | Device-agnostic loop with `torch.device`                                          | Section 10       |

**Suggested modifications:**

| Modification                                                                                     | What it reveals                                                                              |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| Replace `Adam` with `SGD(lr=0.01, momentum=0.9)`                                                 | Demonstrates Adam's faster initial convergence vs. SGD's potential for better final accuracy |
| Add `nn.BatchNorm1d(64)` after `fc1`                                                             | Shows immediate reduction in loss variance between epochs ([Week 12](../../04_neural_networks/week12_training_pathologies/theory.md) applied in PyTorch)       |
| Double the hidden size to 128                                                                    | More parameters → faster fitting but check for overfitting by comparing train/test accuracy  |
| Use `BCEWithLogitsLoss` with a single output neuron                                              | Binary classification with logits — compare to 2-class `CrossEntropyLoss`                    |
| Add `nn.Dropout(0.3)` between layers and compare with/without `model.eval()`                     | Shows why `eval()` matters: dropout active at test time worsens accuracy                     |
| Try `ReduceLROnPlateau` with patience=5                                                          | Adaptive schedule; observe when LR drops and how loss responds                               |
| Save checkpoint at epoch 25, resume from it, train to 50 — compare to uninterrupted 50-epoch run | Verifies that checkpointing preserves training trajectory exactly                            |
| Remove `optimizer.zero_grad()` and observe what happens                                          | Gradients accumulate → loss initially drops then diverges or oscillates                      |

---

## 15. Symbol Reference

| Symbol                     | Name               | Meaning                                                       |
| -------------------------- | ------------------ | ------------------------------------------------------------- |
| $x$                        | Input tensor       | Shape $(N, d)$, batch of $N$ samples with $d$ features        |
| $W$                        | Weight matrix      | Shape $(d_{\text{out}}, d_{\text{in}})$ in PyTorch convention |
| $b$                        | Bias vector        | Shape $(d_{\text{out}},)$                                     |
| $z$                        | Pre-activation     | $z = Wx + b$ (before nonlinearity)                            |
| $a$                        | Activation         | $a = \sigma(z)$ (after nonlinearity)                          |
| $\hat{y}$                  | Model prediction   | Output of the final layer                                     |
| $\mathcal{L}$              | Loss               | Scalar function of predictions and targets                    |
| $\nabla_\theta\mathcal{L}$ | Gradient           | `param.grad` after `loss.backward()`                          |
| $\eta$                     | Learning rate      | Step size for parameter updates                               |
| $B$                        | Batch size         | Number of samples per mini-batch                              |
| $N$                        | Dataset size       | Total training samples                                        |
| $C$                        | Number of classes  | Output dimension for classification                           |
| $m_t$                      | Adam first moment  | Exponential moving average of gradient                        |
| $v_t$                      | Adam second moment | Exponential moving average of squared gradient                |
| $\beta_1, \beta_2$         | Adam decay rates   | Defaults: 0.9, 0.999                                          |
| $T_{\max}$                 | Annealing period   | Total epochs for cosine schedule                              |
| `state_dict`               | Model state        | Dictionary mapping parameter names to tensors                 |

---

## 16. References

1. Paszke, A., Gross, S., Massa, F., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*. — The PyTorch system paper.
2. PyTorch Documentation. *Autograd mechanics.* https://pytorch.org/docs/stable/notes/autograd.html — Official guide to computational graphs and gradient tracking.
3. PyTorch Documentation. *torch.nn.* https://pytorch.org/docs/stable/nn.html — Complete API reference for layers, losses, and modules.
4. PyTorch Tutorials. *Deep Learning with PyTorch: A 60 Minute Blitz.* https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html — Recommended first tutorial.
5. Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR*. — Adam optimiser (Section 6.2).
6. Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*. — Cosine annealing and warm restarts.
7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 8 (Optimization). MIT Press. — Theory behind SGD, momentum, and adaptive methods.
8. Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks." *WACV*. — Learning rate range test and cyclical schedules.
