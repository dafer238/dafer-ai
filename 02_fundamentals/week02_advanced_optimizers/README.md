Week 02 — Advanced Optimization

## Prerequisites

- **Week 01** — vanilla gradient descent, learning rate effects, and loss landscapes. Week 02 answers "what is wrong with plain gradient descent?" — which requires having seen plain gradient descent first.

## What this week delivers

You will understand _why_ modern training uses Adam rather than vanilla SGD, and when each optimizer is appropriate. This understanding will inform every training experiment from Week 03 onwards.

Overview
Dive into practical optimizer design and tuning. Compare algorithms empirically and learn diagnostics for optimizer choice.

Study
- SGD vs Momentum vs RMSProp vs Adam; bias corrections and effective step sizes
- LR schedules: step, cosine, warmup; LR range tests

Practical libraries & tools
- NumPy and PyTorch for implementing optimizers and running experiments
- TensorBoard/Matplotlib for plotting optimization traces

Datasets & examples
- Synthetic quadratic and logistic loss problems
- Small neural net on MNIST or a tiny regression dataset

Exercises
1) Implement Momentum and Adam in NumPy/PyTorch
   - Compare trajectories on the same convex and non-convex losses.

2) LR schedules and sweeps
   - Run LR range tests and compare final performance with different schedules.

3) Hyperparameter sensitivity
   - Grid search over beta/epsilon values and summarize effects.

Code hint
  # Simple Adam update pseudocode
  m = beta1*m + (1-beta1)*g
  v = beta2*v + (1-beta2)*(g*g)
  m_hat = m/(1-beta1**t)
  v_hat = v/(1-beta2**t)
  param -= lr * m_hat / (np.sqrt(v_hat)+eps)

Readings
- Kingma & Ba (Adam paper), RMSProp references, and practical blog posts comparing optimizers.

Deliverable
- Notebook comparing optimizers on at least two problems with clear plots and a short write-up.

## What comes next

- **Week 03** applies the optimizer to the first real supervised learning problem (linear/logistic regression).
- **Week 09** re-visits optimizer choice using PyTorch's `torch.optim` rather than NumPy implementations.
