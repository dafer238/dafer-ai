Week 01 — Optimization Intuition (Loss as Energy)

## Prerequisites

- **Week 00a** — you need the training loop concept (forward → loss → backward → update), the definition of _loss function_, and the definition of _parameters_ before this week makes sense.
- **Week 00b** — you need NumPy fluency, partial derivatives, and the chain rule. The update rule $w \leftarrow w - \eta \nabla L$ uses all three.

## What this week delivers

By the end of Week 01 you can implement gradient descent from scratch and understand _why_ it works. This is the engine that every later week will modify (Week 02 adds momentum and Adam) or apply at scale (Week 09 via PyTorch autograd).

Overview
Build physical intuition for loss landscapes and optimization dynamics. Focus on experiments that reveal how hyperparameters shape training trajectories.

Study
- Loss functions as energy landscapes — visualize minima and basins
- Gradients as forces; relation to physics intuition
- Convex vs non-convex geometry and saddle points

Practical libraries & tools
- NumPy, Matplotlib/Seaborn for visualizations
- Jupyter notebooks for interactive experiments

Datasets & examples
- Synthetic 2D losses: simple quadratic, donut, multimodal mixtures
- Toy regression datasets (sine + noise) to visualize fit and loss surface

Exercises (step-by-step)
1) Visualize simple loss landscapes
   - Create 2D grids and plot loss contours for quadratic and multimodal functions.

2) Gradient descent dynamics
   - Implement vanilla gradient descent and SGD in NumPy; plot parameter trajectories.

3) Momentum and learning-rate sweeps
   - Implement momentum; run LR sweeps and plot divergence vs convergence.

Concrete snippets
  import numpy as np
  def quadratic(x, y):
      return x**2 + 3*y**2

  # gradient descent step
  x, y = 2.0, -1.0
  lr = 0.1
  gx, gy = 2*x, 6*y
  x -= lr * gx

Recommended readings
- Sections on optimization in Goodfellow et al. and Murphy's optimization chapters
- LR range test and Leslie Smith's cyclical LR notes

Deliverables
- Notebook: loss visualizations, GD/SGD/momentum implementations, LR sweep plots, short conclusions.

Notes
- Emphasize clean plots and reproducible notebooks; include commentary on failures (divergence, oscillation).

## What comes next

- **Week 02** extends the optimizer: momentum, RMSProp, Adam — all built on the same update-rule idea.
- **Week 03** applies gradient descent to a real supervised problem (linear regression) for the first time.
- **Week 07** re-derives this loop for multi-layer networks via backpropagation.
