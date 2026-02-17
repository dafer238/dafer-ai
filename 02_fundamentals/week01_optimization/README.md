Week 01 — Optimization Intuition (Loss as Energy)

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
