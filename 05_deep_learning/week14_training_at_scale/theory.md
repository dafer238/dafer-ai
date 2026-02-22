# Training at Scale

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [The Data Pipeline](#2-the-data-pipeline)
   - 2.1 [Dataset and DataLoader Recap](#21-dataset-and-dataloader-recap)
   - 2.2 [Multi-Worker Loading](#22-multi-worker-loading)
   - 2.3 [Pinned Memory](#23-pinned-memory)
   - 2.4 [Prefetching](#24-prefetching)
   - 2.5 [Profiling the Pipeline](#25-profiling-the-pipeline)
3. [Batching Strategies](#3-batching-strategies)
   - 3.1 [Batch Size and Generalisation](#31-batch-size-and-generalisation)
   - 3.2 [Gradient Accumulation](#32-gradient-accumulation)
   - 3.3 [The Linear Scaling Rule](#33-the-linear-scaling-rule)
4. [Learning Rate Schedules](#4-learning-rate-schedules)
   - 4.1 [Why Schedule the Learning Rate?](#41-why-schedule-the-learning-rate)
   - 4.2 [Step Decay](#42-step-decay)
   - 4.3 [Exponential Decay](#43-exponential-decay)
   - 4.4 [Cosine Annealing](#44-cosine-annealing)
   - 4.5 [Reduce on Plateau](#45-reduce-on-plateau)
   - 4.6 [Warmup](#46-warmup)
   - 4.7 [Warmup + Cosine: The Modern Default](#47-warmup--cosine-the-modern-default)
   - 4.8 [One-Cycle Policy](#48-one-cycle-policy)
5. [Robust Checkpointing](#5-robust-checkpointing)
   - 5.1 [What to Save](#51-what-to-save)
   - 5.2 [Checkpoint Rotation](#52-checkpoint-rotation)
   - 5.3 [Best-Model Tracking](#53-best-model-tracking)
   - 5.4 [Resuming Training Correctly](#54-resuming-training-correctly)
6. [Mixed Precision Training](#6-mixed-precision-training)
   - 6.1 [The Idea](#61-the-idea)
   - 6.2 [Loss Scaling](#62-loss-scaling)
   - 6.3 [PyTorch AMP](#63-pytorch-amp)
7. [Distributed Training Overview](#7-distributed-training-overview)
   - 7.1 [Data Parallelism](#71-data-parallelism)
   - 7.2 [Model Parallelism](#72-model-parallelism)
   - 7.3 [PyTorch APIs](#73-pytorch-apis)
8. [Putting It Together: The Production Training Loop](#8-putting-it-together-the-production-training-loop)
9. [Connections to the Rest of the Course](#9-connections-to-the-rest-of-the-course)
10. [Notebook Reference Guide](#10-notebook-reference-guide)
11. [Symbol Reference](#11-symbol-reference)
12. [References](#12-references)

---

## 1. Scope and Purpose

[Week 13](../week13_pytorch_basics/theory.md) built a working PyTorch training loop. This week wraps that loop in **production tooling**: efficient data loading, learning rate schedules, checkpointing, and an introduction to mixed precision and distributed training.

The difference between a research prototype and a reproducible experiment is almost entirely in this engineering layer. None of it changes the model's mathematics — it changes whether training finishes at all, finishes in reasonable time, and whether you can resume when it fails.

**Prerequisites.** [Week 13](../week13_pytorch_basics/theory.md) (PyTorch: `nn.Module`, DataLoader, optimisers, `state_dict`). [[Weeks 01](../../02_fundamentals/week01_optimization/theory.md)](../../02_fundamentals/week01_optimization/theory.md)–[02](../../02_fundamentals/week02_advanced_optimizers/theory.md) (learning rate intuition, SGD, Adam).

---

## 2. The Data Pipeline

### 2.1 Dataset and DataLoader Recap

From [Week 13](../week13_pytorch_basics/theory.md):

```python
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

This works, but for real workloads (large images, data on disk, augmentation) the pipeline becomes the bottleneck. The goal is to ensure the GPU never waits for data.

---

### 2.2 Multi-Worker Loading

```python
loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)
```

| Parameter       | Effect                                                |
| --------------- | ----------------------------------------------------- |
| `num_workers=0` | Single-process loading (main process does everything) |
| `num_workers=N` | $N$ subprocesses prepare batches in parallel          |

**How it works.** With $N$ workers, the DataLoader spawns $N$ subprocesses. Each worker calls `dataset.__getitem__()` independently. The main process consumes ready batches from a queue. While the GPU trains on batch $t$, workers are already loading batch $t + 1, t + 2, \ldots$

**Choosing `num_workers`:**

$$N_{\text{workers}} \approx \min\!\left(\text{CPU cores},\; 4 \times \text{num GPUs}\right)$$

Too many workers waste memory and CPU context-switching; too few leave the GPU idle.

> **Windows/Jupyter caveat.** On Windows, multi-worker DataLoaders use `spawn` (not `fork`). The `Dataset` class must be importable from a module (not defined inline in a notebook cell). The notebook imports `SyntheticDataset` from `utils.datasets` for this reason.

---

### 2.3 Pinned Memory

```python
loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```

**Pinned (page-locked) memory** is host (CPU) memory that the operating system guarantees will not be swapped to disk. GPU DMA transfers from pinned memory are faster because the GPU can read directly without an extra copy.

```
Normal:   Dataset → CPU memory → (copy to pinned) → GPU
Pinned:   Dataset → pinned CPU memory → GPU  (one fewer copy)
```

| Setting            | When to use                           |
| ------------------ | ------------------------------------- |
| `pin_memory=False` | CPU-only training; debugging          |
| `pin_memory=True`  | GPU training (default recommendation) |

After pinning, use non-blocking transfers in the loop:

```python
batch_x = batch_x.to(device, non_blocking=True)
batch_y = batch_y.to(device, non_blocking=True)
```

This overlaps the CPU→GPU transfer with computation on the previous batch.

---

### 2.4 Prefetching

Prefetching means preparing the next batch while the current batch is being processed. PyTorch's DataLoader implements this automatically via its internal queue when `num_workers > 0`. The `prefetch_factor` parameter (default 2) controls how many batches each worker pre-loads:

```python
loader = DataLoader(dataset, batch_size=64, num_workers=4, prefetch_factor=2)
# Each of the 4 workers has 2 batches ready → 8 batches in the queue at any time
```

---

### 2.5 Profiling the Pipeline

The first optimisation question is always: **where is the bottleneck?**

| Bottleneck           | Symptom                            | Fix                                            |
| -------------------- | ---------------------------------- | ---------------------------------------------- |
| **Data loading**     | GPU utilisation < 50%; CPU at 100% | More workers, faster storage, prefetching      |
| **CPU→GPU transfer** | GPU idle between batches           | `pin_memory=True`, `non_blocking=True`         |
| **GPU compute**      | GPU utilisation ~100%; CPU idle    | Larger batch, mixed precision, better GPU      |
| **Python overhead**  | Neither CPU nor GPU saturated      | `torch.compile()`, TorchScript, reduce logging |

```python
import time

start = time.time()
for i, (batch_x, _) in enumerate(loader):
    if i >= 50: break
    _ = batch_x.mean()  # minimal "compute"
elapsed = time.time() - start
throughput = 50 / elapsed  # batches/sec
```

> **Notebook reference.** Cell 2 benchmarks four DataLoader configurations (0/2/4 workers, ± pinned memory) on a synthetic dataset with simulated I/O delay. Multi-worker loading significantly improves throughput.

---

## 3. Batching Strategies

### 3.1 Batch Size and Generalisation

From [Week 01](../../02_fundamentals/week01_optimization/theory.md), the mini-batch gradient is a noisy estimate of the full-batch gradient:

$$g_B = \frac{1}{|B|}\sum_{i \in B}\nabla_\theta\mathcal{L}_i \qquad\text{where}\qquad \mathbb{E}[g_B] = \nabla_\theta\mathcal{L}$$

The variance of this estimate scales as:

$$\text{Var}(g_B) = \frac{\sigma^2}{|B|}$$

| Batch size $     | B        | $                                     | Gradient noise        | Generalisation | Throughput |
| ---------------- | -------- | ------------------------------------- | --------------------- |
| Small (32–64)    | High     | Often better (noise ≈ regularisation) | Lower GPU utilisation |
| Medium (128–256) | Moderate | Good                                  | Good utilisation      |
| Large (1024+)    | Low      | Can degrade without tuning            | High utilisation      |

**Key insight (Keskar et al., 2017).** Large batches tend to converge to sharp minima with poor generalisation. Small batches introduce noise that helps escape sharp minima and find flat minima.

> **Practical default.** Start with $|B| = 32$–$128$. Increase if the GPU is underutilised. If you increase the batch size, increase the learning rate proportionally (linear scaling rule).

---

### 3.2 Gradient Accumulation

When GPU memory limits the batch size, simulate larger batches by accumulating gradients over $K$ steps before updating:

```python
accumulation_steps = 4   # effective batch = 4 × 32 = 128

optimizer.zero_grad()
for i, (batch_x, batch_y) in enumerate(train_loader):
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y) / accumulation_steps  # scale loss
    loss.backward()                                           # accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Why divide by $K$?** Without the division, the accumulated gradient is:

$$g_{\text{accum}} = \sum_{k=1}^{K}\nabla_\theta\mathcal{L}_{B_k}$$

This is $K$ times larger than the single-batch gradient, effectively multiplying the learning rate by $K$. Dividing the loss by $K$ normalises:

$$g_{\text{accum}} = \frac{1}{K}\sum_{k=1}^{K}\nabla_\theta\mathcal{L}_{B_k} \approx \nabla_\theta\mathcal{L}$$

> **Notebook reference.** Exercise 1 asks you to implement gradient accumulation with $K = 4$ and compare the training curve to a true large-batch baseline.

---

### 3.3 The Linear Scaling Rule

When you multiply the batch size by $k$, multiply the learning rate by $k$ (Goyal et al., 2017):

$$|B| \to k|B| \implies \eta \to k\eta$$

**Intuition.** A larger batch gives a more accurate gradient estimate with lower variance. The larger step size compensates for making fewer updates per epoch.

**Caveat.** The linear scaling rule works well up to a point (often $|B| \leq 8{,}192$). Beyond that, training can diverge — warmup (Section 4.6) is essential.

---

## 4. Learning Rate Schedules

### 4.1 Why Schedule the Learning Rate?

A fixed learning rate faces a dilemma:
- **Too large:** training oscillates or diverges, especially near the minimum.
- **Too small:** convergence is painfully slow.

The solution: start with a large $\eta$ (fast progress in the early phase) and reduce it over time (fine convergence near the minimum).

Formally, the optimal constant learning rate depends on the **local curvature** of the loss landscape (the largest eigenvalue of the Hessian). As optimisation progresses, the landscape around the minimum has higher curvature, demanding a smaller step:

$$\eta_{\text{optimal}} \propto \frac{1}{\lambda_{\max}(H)}$$

Since $\lambda_{\max}$ increases near sharp features of the loss surface, reducing $\eta$ over training matches the geometry.

---

### 4.2 Step Decay

Multiply $\eta$ by a factor $\gamma < 1$ every $S$ epochs:

$$\eta_t = \eta_0\cdot\gamma^{\lfloor t/S\rfloor}$$

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

| Epoch | $\eta$ (with $\eta_0 = 0.1, \gamma = 0.1, S = 30$) |
| ----- | -------------------------------------------------- |
| 0–29  | 0.1                                                |
| 30–59 | 0.01                                               |
| 60–89 | 0.001                                              |

**Pros:** simple, predictable.
**Cons:** abrupt drops can destabilise training momentarily; requires choosing $S$ and $\gamma$ by hand.

---

### 4.3 Exponential Decay

Smooth continuous decay:

$$\eta_t = \eta_0\cdot\gamma^t$$

```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

After 100 epochs with $\gamma = 0.95$: $\eta_{100} = 0.1\cdot 0.95^{100} \approx 5.9 \times 10^{-4}$.

**Pros:** smooth; no hyperparameter for step frequency.
**Cons:** decay can be too fast or too slow depending on $\gamma$.

---

### 4.4 Cosine Annealing

$$\boxed{\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi\, t}{T_{\max}}\right)\right)}$$

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

At $t = 0$: $\eta = \eta_{\max}$. At $t = T_{\max}$: $\eta = \eta_{\min}$. The decay is **slow at the start** (model is still in a noisy regime), **fast in the middle** (rapid convergence), and **slow at the end** (fine-tuning near the minimum).

**Cosine annealing with warm restarts** (SGDR, Loshchilov & Hutter 2017) resets the schedule periodically, allowing the model to explore new regions:

```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
# First cycle: 10 epochs; second: 20; third: 40; ...
```

---

### 4.5 Reduce on Plateau

Adaptive: reduce $\eta$ when a monitored metric stops improving.

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# After each epoch:
scheduler.step(val_loss)   # pass the metric, not just call .step()
```

If `val_loss` hasn't improved for 10 consecutive epochs, $\eta \to 0.5\eta$.

**Pros:** no need to choose a schedule a priori.
**Cons:** reactive, not proactive — by the time the plateau is detected, some epochs have been wasted.

> **Notebook reference.** Cell 3 visualises StepLR, ExponentialLR, CosineAnnealingLR, and ReduceLROnPlateau side by side on a log-scale plot.

---

### 4.6 Warmup

Start with $\eta \approx 0$ and linearly increase to the target $\eta_{\max}$ over $T_w$ steps:

$$\eta_t = \eta_{\max}\cdot\frac{t}{T_w} \qquad \text{for } t \leq T_w$$

**Why warmup?**

1. **Adam's estimates are unreliable at $t = 0$.** The bias-corrected moments $\hat{m}_t, \hat{v}_t$ are noisy when $t$ is small, so the adaptive step size is erratic. A small $\eta$ during this phase prevents premature divergence.

2. **Large-batch training.** With the linear scaling rule ($\eta \propto |B|$), the initial $\eta$ can be very large. Warmup gives the model a few steps at moderate $\eta$ to find a reasonable basin before ramping up.

3. **BatchNorm running statistics.** The running mean and variance are initialised to $\mu = 0, \sigma^2 = 1$. During the first few epochs, these stabilise; large updates during this phase can cause instability.

---

### 4.7 Warmup + Cosine: The Modern Default

The most common schedule in modern deep learning:

$$\eta_t = \begin{cases}\eta_{\max}\cdot\dfrac{t}{T_w} & t \leq T_w\\[6pt]\eta_{\min} + \dfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\dfrac{\pi(t - T_w)}{T_{\max} - T_w}\right)\right) & t > T_w\end{cases}$$

```
LR
|   /‾‾‾‾\
|  /        \
| /           ‾‾‾‾\_____
|/________________________ epoch
  warmup       cosine decay
```

PyTorch provides this via `torch.optim.lr_scheduler.SequentialLR` or manual implementation:

```python
def warmup_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

> **Notebook reference.** Exercise 2 asks you to implement warmup + cosine from scratch and plot the curve.

---

### 4.8 One-Cycle Policy

Smith (2018): ramp $\eta$ from low → high → low in a single cycle, with a corresponding inverse cycle for momentum.

$$\eta: \quad \eta_{\min} \xrightarrow{\text{phase 1}} \eta_{\max} \xrightarrow{\text{phase 2}} \eta_{\min} \to 0$$

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, epochs=100, steps_per_epoch=len(train_loader)
)
# Call scheduler.step() after every BATCH, not every epoch
```

**Why it works (super-convergence).** The high-LR phase acts as a regulariser (the noise from large updates explores broadly), while the low-LR phases at the start and end stabilise convergence. This often trains faster than cosine annealing alone.

---

## 5. Robust Checkpointing

### 5.1 What to Save

A production checkpoint includes:

| Component             | Why                            | PyTorch API                                             |
| --------------------- | ------------------------------ | ------------------------------------------------------- |
| Model parameters      | The learned weights            | `model.state_dict()`                                    |
| Optimiser state       | Momentum, Adam moments         | `optimizer.state_dict()`                                |
| Scheduler state       | Step count, internal variables | `scheduler.state_dict()`                                |
| Epoch / step number   | Resume from correct position   | Integer                                                 |
| Metrics               | Compare runs, track best model | Dictionary                                              |
| RNG states (optional) | Exact reproducibility          | `torch.random.get_rng_state()`, `np.random.get_state()` |

---

### 5.2 Checkpoint Rotation

Saving every epoch accumulates disk space. A checkpoint manager keeps only the last $N$ checkpoints:

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(exist_ok=True)
        self.keep_last_n = keep_last_n

    def save(self, state, epoch, is_best=False):
        path = self.dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(state, path)
        if is_best:
            torch.save(state, self.dir / 'best_model.pth')
        self._cleanup()

    def _cleanup(self):
        ckpts = sorted(self.dir.glob('checkpoint_epoch_*.pth'))
        for old in ckpts[:-self.keep_last_n]:
            old.unlink()
```

> **Notebook reference.** Cell 4 implements a full `CheckpointManager` with metadata, rotation, and best-model tracking.

---

### 5.3 Best-Model Tracking

Track the validation metric across epochs and save a special checkpoint when it improves:

```python
best_val_loss = float('inf')

for epoch in range(n_epochs):
    train_loss = train_epoch(...)
    val_loss = evaluate(...)

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss

    ckpt_manager.save(
        {'epoch': epoch, 'model': model.state_dict(), ...},
        epoch=epoch,
        is_best=is_best,
    )
```

At the end of training, `best_model.pth` contains the checkpoint with the best validation performance — not necessarily the last epoch (which may be overfit).

---

### 5.4 Resuming Training Correctly

```python
# 1. Recreate architecture (same hyperparameters)
model = SimpleMLP(10, 64, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 2. Load checkpoint
ckpt = torch.load('checkpoint.pth')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
scheduler.load_state_dict(ckpt['scheduler_state_dict'])
start_epoch = ckpt['epoch'] + 1

# 3. Resume
for epoch in range(start_epoch, 100):
    train_epoch(model, ...)
    scheduler.step()
```

**Critical detail:** you must load the **scheduler** state too. Otherwise, the learning rate restarts from $\eta_0$ instead of continuing from where it left off. Similarly, the **optimiser** state (Adam's $m_t, v_t$) must be restored — without it, Adam's adaptive estimates reset, causing a learning rate spike.

> **Notebook reference.** Exercise 4 asks you to train for 25 epochs, "crash" (reinitialise model), load the checkpoint, resume for 25 more, and verify a smooth 50-epoch loss curve.

---

## 6. Mixed Precision Training

### 6.1 The Idea

Neural network computations don't need full 32-bit precision everywhere. **Mixed precision** runs the forward pass and gradient computation in **float16** (half precision, 16 bits) while keeping master weights and parameter updates in **float32**.

| Precision | Bits | Dynamic range                      | Use                                  |
| --------- | ---- | ---------------------------------- | ------------------------------------ |
| float32   | 32   | $\sim 10^{\pm 38}$                 | Master weights, loss computation     |
| float16   | 16   | $\sim 10^{\pm 5}$                  | Matrix multiplications, convolutions |
| bfloat16  | 16   | $\sim 10^{\pm 38}$ (same as fp32!) | Preferred on modern hardware (A100+) |

**Benefits:**
- **Speed.** GPU tensor cores process float16 operations 2–8× faster than float32.
- **Memory.** Activations stored in float16 use half the GPU memory, allowing larger batch sizes or models.

---

### 6.2 Loss Scaling

Float16 has limited dynamic range. Small gradient values (e.g., $10^{-6}$) underflow to zero. **Loss scaling** multiplies the loss by a large constant $S$ before the backward pass, then divides gradients by $S$ after:

$$\text{scaled gradients} = S \cdot \nabla_\theta\mathcal{L} \qquad\text{(computed in fp16)}$$
$$\text{true gradients} = \frac{1}{S}\cdot\text{scaled gradients} \qquad\text{(used for fp32 update)}$$

This shifts the gradient distribution into the representable range of float16.

**Dynamic loss scaling:** start with a large $S$ (e.g., $2^{16}$) and halve it whenever an overflow (NaN/Inf) is detected; double it when $N$ consecutive steps succeed. PyTorch's `GradScaler` implements this automatically.

---

### 6.3 PyTorch AMP

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()

    with autocast():                          # forward in fp16
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

    scaler.scale(loss).backward()             # backward with scaled loss
    scaler.step(optimizer)                    # unscale → clip → step
    scaler.update()                           # adjust scale factor
```

| Function                 | What it does                                                          |
| ------------------------ | --------------------------------------------------------------------- |
| `autocast()`             | Context manager: casts eligible ops to float16/bfloat16               |
| `scaler.scale(loss)`     | Multiplies loss by the current scale factor                           |
| `scaler.step(optimizer)` | Unscales gradients, checks for NaN/Inf, then calls `optimizer.step()` |
| `scaler.update()`        | Adjusts the scale factor based on NaN history                         |

> **Notebook reference.** Exercise 1 in the exercises section mentions mixed precision as an extension.

---

## 7. Distributed Training Overview

### 7.1 Data Parallelism

The most common form: replicate the model on $G$ GPUs, split each mini-batch across GPUs, compute gradients independently, then **all-reduce** (average) the gradients.

```
Batch (N samples)
  ├─ GPU 0: N/G samples → gradients₀
  ├─ GPU 1: N/G samples → gradients₁
  └─ GPU 2: N/G samples → gradients₂
              ↓ all-reduce ↓
       averaged gradient → update (all GPUs)
```

Effective batch size = $|B| \times G$. Apply the linear scaling rule: $\eta \to G\eta$.

---

### 7.2 Model Parallelism

When the model doesn't fit on a single GPU, split it across devices:

- **Pipeline parallelism:** different layers on different GPUs; micro-batches pipelined through.
- **Tensor parallelism:** individual weight matrices split across GPUs (e.g., Megatron-LM).

Model parallelism is more complex and harder to scale efficiently. Data parallelism is the starting point for almost all distributed training.

---

### 7.3 PyTorch APIs

| API                                     | Use case                | Notes                                              |
| --------------------------------------- | ----------------------- | -------------------------------------------------- |
| `nn.DataParallel`                       | Single-node, multi-GPU  | Simple but inefficient (GIL bottleneck)            |
| `DistributedDataParallel` (DDP)         | Multi-node or multi-GPU | Preferred; overlaps communication with compute     |
| `FullyShardedDataParallel` (FSDP)       | Large models            | Shards parameters, gradients, and optimiser states |
| `torchrun` / `torch.distributed.launch` | Launch DDP processes    | One process per GPU                                |

```python
# Basic DDP pattern (one process per GPU)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model.to(local_rank), device_ids=[local_rank])
```

Full distributed training is beyond this week's scope but the concepts appear again in [Week 19](../../07_transfer_learning/week19_finetuning/theory.md) (fine-tuning) and [Week 20](../../08_deployment/week20_deployment/theory.md) (deployment).

---

## 8. Putting It Together: The Production Training Loop

Combining all the pieces from this week:

```python
# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()               # mixed precision
ckpt_mgr = CheckpointManager('checkpoints', keep_last_n=3)

# DataLoader (efficient)
train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True, prefetch_factor=2,
)

# Training
best_val = float('inf')
for epoch in range(start_epoch, 100):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    # Evaluate
    val_loss = evaluate(model, val_loader, criterion, device)
    is_best = val_loss < best_val
    if is_best:
        best_val = val_loss

    # Checkpoint
    ckpt_mgr.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'val_loss': val_loss,
    }, epoch=epoch, is_best=is_best)
```

**Checklist for a production loop:**

- [x] DataLoader with `num_workers > 0`, `pin_memory=True`
- [x] Device-agnostic (CPU/GPU) with `non_blocking=True`
- [x] Learning rate schedule (warmup + cosine)
- [x] Mixed precision (`autocast` + `GradScaler`)
- [x] Checkpoint rotation with best-model tracking
- [x] Scheduler and scaler state saved for correct resumption

---

## 9. Connections to the Rest of the Course

| Week                               | Connection                                                                                                                               |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **[[Week 01](../../02_fundamentals/week01_optimization/theory.md)](../../02_fundamentals/week01_optimization/theory.md)–[02](../../02_fundamentals/week02_advanced_optimizers/theory.md) (Optimisation)**      | LR intuition; SGD variance analysis; Adam's adaptive rates — the theory behind schedulers                                                |
| **[Week 12](../../04_neural_networks/week12_training_pathologies/theory.md) (Training Pathologies)** | Gradient clipping integrates with `scaler.step()` in mixed precision; BatchNorm running stats need warmup                                |
| **[Week 13](../week13_pytorch_basics/theory.md) (PyTorch Basics)**       | DataLoader, `state_dict`, training loop — this week extends them to production quality                                                   |
| **[Week 15](../week15_cnn_representations/theory.md) (CNNs)**                 | Convolutional models are expensive; multi-worker loading and mixed precision are essential                                               |
| **[Week 16](../week16_regularization_dl/theory.md) (Regularisation DL)**    | Data augmentation happens in the DataLoader pipeline (transforms); dropout interacts with checkpointing (`model.train()`/`model.eval()`) |
| **[Week 18](../../06_sequence_models/week18_transformers/theory.md) (Transformers)**         | Transformers require warmup + cosine schedule; mixed precision is standard for large models                                              |
| **[Week 19](../../07_transfer_learning/week19_finetuning/theory.md) (Fine-Tuning)**          | Loading pre-trained checkpoints; freezing layers; discriminative LR schedules build on this week's scheduler patterns                    |
| **[Week 20](../../08_deployment/week20_deployment/theory.md) (Deployment)**           | Model serialisation, ONNX export, TorchScript — extensions of `torch.save` and `state_dict`                                              |

---

## 10. Notebook Reference Guide

| Cell                      | Section       | What it demonstrates                                                                                    | Theory reference |
| ------------------------- | ------------- | ------------------------------------------------------------------------------------------------------- | ---------------- |
| 1 (Custom Dataset)        | Data pipeline | `SyntheticDataset` with configurable I/O delay; imported from `utils.datasets` for worker compatibility | Section 2        |
| 2 (DataLoader benchmark)  | Profiling     | 4 configs (0/2/4 workers ± pinned memory); throughput bar chart                                         | Section 2.2–2.5  |
| 3 (LR schedulers)         | Scheduling    | StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau visualised on log scale                     | Section 4        |
| 4 (CheckpointManager)     | Checkpointing | Full class with save/load/cleanup/best-model tracking; 5-epoch demo                                     | Section 5        |
| Ex. 1 (Grad accumulation) | Batching      | $K = 4$ accumulation vs. true large-batch baseline                                                      | Section 3.2      |
| Ex. 2 (Warmup + Cosine)   | Scheduling    | Custom scheduler function: linear warmup 10 epochs + cosine 90 epochs                                   | Section 4.7      |
| Ex. 4 (Resume training)   | Checkpointing | Train 25 epochs, "crash", resume from checkpoint, verify smooth 50-epoch curve                          | Section 5.4      |

**Suggested modifications:**

| Modification                                                                  | What it reveals                                                                           |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Increase `num_workers` from 4 to 8 or 16                                      | Diminishing returns; at some point, workers compete for CPU cores and slow down           |
| Benchmark with real disk I/O (load images) instead of synthetic delay         | Real bottleneck is disk speed, not simulated sleep; SSDs vs HDDs matter enormously        |
| Try `OneCycleLR` instead of `CosineAnnealingLR`                               | Often converges faster; observe the LR ramp-up phase                                      |
| Remove `pin_memory` and add timing around `.to(device)`                       | Measures the actual CPU→GPU transfer overhead                                             |
| Set `accumulation_steps=16` and compare to `K=1`                              | Very large effective batch: smoother gradients but fewer updates per epoch                |
| Corrupt a checkpoint file and test recovery                                   | The `CheckpointManager` should gracefully fall back to the previous checkpoint            |
| Add `torch.backends.cudnn.benchmark = True` and time convolution-heavy models | cuDNN auto-tunes kernel selection; first epoch is slower but subsequent epochs are faster |

---

## 11. Symbol Reference

| Symbol                   | Name                       | Meaning                                              |
| ------------------------ | -------------------------- | ---------------------------------------------------- |
| $                        | B                          | $                                                    | Batch size | Number of samples per mini-batch    |
| $G$                      | Number of GPUs             | For data-parallel scaling                            |
| $K$                      | Accumulation steps         | Number of mini-batches before one `optimizer.step()` |
| $\eta$                   | Learning rate              | Step size; scheduled over training                   |
| $\eta_0$ / $\eta_{\max}$ | Initial / peak LR          | Starting or maximum value before decay               |
| $\eta_{\min}$            | Minimum LR                 | Floor of the schedule (often 0 or $10^{-6}$)         |
| $T_{\max}$               | Total epochs               | Cosine annealing period                              |
| $T_w$                    | Warmup epochs              | Duration of linear warmup phase                      |
| $\gamma$                 | Decay factor               | Step/exponential multiplier ($< 1$)                  |
| $S$                      | Loss scale                 | Mixed-precision scaling factor                       |
| $g_B$                    | Mini-batch gradient        | $\frac{1}{                                           | B          | }\sum_{i \in B}\nabla\mathcal{L}_i$ |
| $\sigma^2$               | Gradient variance          | Per-sample gradient variance                         |
| $\lambda_{\max}(H)$      | Largest Hessian eigenvalue | Determines optimal step size                         |
| $m_t, v_t$               | Adam moments               | First/second moment estimates ([Week 02](../../02_fundamentals/week02_advanced_optimizers/theory.md))              |
| `state_dict`             | Model/optimiser state      | Serialised dictionary of parameters                  |
| `pin_memory`             | Pinned CPU memory          | Page-locked for fast GPU transfer                    |
| `non_blocking`           | Async transfer             | Overlap CPU→GPU copy with computation                |

---

## 12. References

1. Goyal, P., Dollár, P., Girshick, R., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." *arXiv:1706.02677*. — Linear scaling rule and warmup for large-batch training.
2. Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*. — Cosine annealing with warm restarts.
3. Smith, L. N. (2018). "A Disciplined Approach to Neural Network Hyper-Parameters." *arXiv:1803.09820*. — One-cycle policy and super-convergence.
4. Micikevicius, P., Narang, S., Alben, J., et al. (2018). "Mixed Precision Training." *ICLR*. — Loss scaling and mixed-precision methodology.
5. Keskar, N. S., Mudigere, D., Nocedal, J., et al. (2017). "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." *ICLR*. — Batch size and generalisation trade-off.
6. Li, S., Zhao, Y., Varma, R., et al. (2020). "PyTorch Distributed: Experiences on Accelerating Data Parallel Training." *VLDB*. — DistributedDataParallel system design.
7. PyTorch Documentation. *Automatic Mixed Precision.* https://pytorch.org/docs/stable/amp.html — AMP API reference.
8. PyTorch Documentation. *torch.utils.data.* https://pytorch.org/docs/stable/data.html — DataLoader, Dataset, Sampler reference.
9. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 8. MIT Press. — Optimisation, learning rate schedules, and batch size analysis.
10. You, Y., Gitman, I., & Ginsburg, B. (2017). "Large Batch Training of Convolutional Networks." *arXiv:1708.03888*. — LARS optimiser; layer-wise adaptive rates for very large batches.
