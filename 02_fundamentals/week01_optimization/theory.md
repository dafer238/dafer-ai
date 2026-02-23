# Optimisation Intuition: Loss as Energy

## Table of Contents

- [Optimisation Intuition: Loss as Energy](#optimisation-intuition-loss-as-energy)
  - [Table of Contents](#table-of-contents)
  - [1. Scope and Purpose](#1-scope-and-purpose)
  - [2. The Optimisation Problem](#2-the-optimisation-problem)
  - [3. Loss Functions](#3-loss-functions)
    - [3.1 What Is a Loss Function?](#31-what-is-a-loss-function)
    - [3.2 Mean Squared Error](#32-mean-squared-error)
    - [3.3 Cross-Entropy Loss](#33-cross-entropy-loss)
    - [3.4 Loss Landscapes](#34-loss-landscapes)
  - [4. Gradient Descent](#4-gradient-descent)
    - [4.1 The Gradient](#41-the-gradient)
    - [4.2 The Update Rule](#42-the-update-rule)
    - [4.3 Convergence Analysis for Quadratic Losses](#43-convergence-analysis-for-quadratic-losses)
    - [4.4 The Learning Rate](#44-the-learning-rate)
    - [4.5 Full-Batch Gradient Descent in Matrix Form](#45-full-batch-gradient-descent-in-matrix-form)
  - [5. Stochastic Gradient Descent (SGD)](#5-stochastic-gradient-descent-sgd)
    - [5.1 Motivation](#51-motivation)
    - [5.2 Mini-Batch SGD](#52-mini-batch-sgd)
    - [5.3 Variance and the Noise–Speed Trade-off](#53-variance-and-the-noisespeed-trade-off)
  - [6. Momentum](#6-momentum)
    - [6.1 The Physics Analogy](#61-the-physics-analogy)
    - [6.2 The Momentum Update Rule](#62-the-momentum-update-rule)
    - [6.3 Why Momentum Helps](#63-why-momentum-helps)
    - [6.4 Nesterov Accelerated Gradient](#64-nesterov-accelerated-gradient)
  - [7. The Geometry of Loss Landscapes](#7-the-geometry-of-loss-landscapes)
    - [7.1 Convex Functions](#71-convex-functions)
    - [7.2 Non-Convex Landscapes, Local Minima, and Saddle Points](#72-non-convex-landscapes-local-minima-and-saddle-points)
    - [7.3 Conditioning and the Hessian](#73-conditioning-and-the-hessian)
    - [7.4 Saddle Points in High Dimensions](#74-saddle-points-in-high-dimensions)
  - [8. Learning Rate Selection](#8-learning-rate-selection)
    - [8.1 The Divergence Threshold](#81-the-divergence-threshold)
    - [8.2 Learning Rate Schedules](#82-learning-rate-schedules)
    - [8.3 Leslie Smith's LR Range Test](#83-leslie-smiths-lr-range-test)
  - [9. Convergence Diagnostics](#9-convergence-diagnostics)
  - [10. The Physical Analogy: Loss as Energy](#10-the-physical-analogy-loss-as-energy)
    - [The Gradient Descent Analogy](#the-gradient-descent-analogy)
    - [Simulated Annealing Connection](#simulated-annealing-connection)
  - [11. Notebook Reference Guide](#11-notebook-reference-guide)
  - [12. Symbol Reference](#12-symbol-reference)
  - [13. References](#13-references)

---

## 1. Scope and Purpose

This week develops the core engine of machine learning: **optimisation**. Every model in this course — from a 2-parameter linear regression to a multi-million-parameter transformer — is trained by the same fundamental procedure: measuring how wrong the model is (loss), computing the direction of improvement (gradient), and adjusting parameters to be less wrong (update step).

The goal is threefold:
1. **Formal understanding** of gradient descent and its variants, grounded in calculus and linear algebra.
2. **Geometric and physical intuition** for loss landscapes, convergence, and failure modes.
3. **Practical fluency** in implementing optimisers from scratch and diagnosing their behaviour.

**Prerequisites.** This document assumes familiarity with:
- Partial derivatives, the gradient vector, and the chain rule ([Week 00b](../../01_intro/week00b_math_and_data/theory.md#3-part-ii-calculus-and-optimisation), Part II).
- Matrix–vector multiplication and norms ([Week 00b](../../01_intro/week00b_math_and_data/theory.md#2-part-i-linear-algebra), Part I).
- The training loop concept: forward → loss → backward → update ([Week 00a](../../01_intro/week00_ai_landscape/theory.md#7-the-training-loop)).

---

## 2. The Optimisation Problem

Machine learning is, at its mathematical core, an optimisation problem. Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ and a parameterised model $f_\theta$, we seek the parameters $\theta^*$ that minimise a loss function:

$$\theta^* = \arg\min_{\theta \in \Theta} \mathcal{L}(\theta)$$

| Symbol                    | Meaning                                                                  |
| ------------------------- | ------------------------------------------------------------------------ |
| $\theta \in \mathbb{R}^p$ | Parameter vector (all learnable weights and biases)                      |
| $\Theta$                  | Parameter space — the set of all possible $\theta$                       |
| $p$                       | Number of parameters                                                     |
| $\mathcal{L}(\theta)$     | Loss function — a scalar measuring prediction error                      |
| $\theta^*$                | Optimal parameters — the minimiser of $\mathcal{L}$                      |
| $\arg\min$                | "The argument that minimises" — returns the $\theta$, not the loss value |

The loss function $\mathcal{L}(\theta)$ is a scalar-valued function of $p$ variables. Visualising it as a **surface** (a "landscape") over the parameter space provides powerful intuition: the height at each point is the loss, and the goal is to descend to the lowest valley.

> **Analogy.** Imagine standing on a hilly terrain in fog — you cannot see the global shape, only the slope at your feet. Gradient descent is the strategy of always stepping downhill. It works remarkably well, but it can get trapped in valleys that are not the deepest one (local minima) or stall on flat ridges (saddle points).

---

## 3. Loss Functions

### 3.1 What Is a Loss Function?

A **loss function** $\mathcal{L} : \Theta \to \mathbb{R}$ maps a parameter configuration to a scalar that quantifies how badly the model fits the data. A good loss function satisfies:

1. $\mathcal{L}(\theta) \geq 0$ (non-negative — zero means perfect fit).
2. $\mathcal{L}(\theta) = 0$ if and only if the model perfectly predicts all training targets (for most losses).
3. $\mathcal{L}$ is differentiable with respect to $\theta$ (required for gradient-based optimisation).

The loss is typically an average over individual per-sample losses:

$$\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\ell(y_i, f_\theta(\mathbf{x}_i))$$

| Symbol                   | Meaning                                                          |
| ------------------------ | ---------------------------------------------------------------- |
| $\ell(y, \hat{y})$       | Per-sample loss: compares true value $y$ to prediction $\hat{y}$ |
| $f_\theta(\mathbf{x}_i)$ | Model prediction for sample $i$, written $\hat{y}_i$             |
| $n$                      | Number of training samples                                       |

> **Why an average?** Dividing by $n$ makes the loss scale-invariant with respect to dataset size. Without this normalisation, the gradient magnitude would grow with $n$, requiring the learning rate to be adjusted whenever the dataset size changes.

---

### 3.2 Mean Squared Error

The most fundamental regression loss:

$$\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{n}\sum_{i=1}^{n}(y_i - f_\theta(\mathbf{x}_i))^2 = \frac{1}{n}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$$

| Symbol                               | Meaning                         |
| ------------------------------------ | ------------------------------- |
| $y_i$                                | True target for sample $i$      |
| $\hat{y}_i = f_\theta(\mathbf{x}_i)$ | Model prediction for sample $i$ |
| $\mathbf{y} - \hat{\mathbf{y}}$      | Residual vector                 |

**Properties:**
- Each squared term $(y_i - \hat{y}_i)^2$ is non-negative and zero when the prediction is exact.
- Squaring penalises large errors more than small ones (quadratic penalty).
- The MSE is differentiable everywhere, which is essential for gradient descent.
- Its gradient has a simple closed form (derived below).

**Probabilistic justification.** MSE is the negative log-likelihood under a Gaussian noise model: $y_i = f_\theta(\mathbf{x}_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ (proved in [Week 07](../../03_probability/week07_likelihood/theory.md#51-gaussian-noise-mse)). If the noise is instead Laplace-distributed, the corresponding loss is the **Mean Absolute Error** $\frac{1}{n}\sum |y_i - \hat{y}_i|$, which is more robust to outliers.

**Gradient of the MSE.** Let $r_i = y_i - \hat{y}_i$ denote the residual. For a linear model $\hat{y}_i = \mathbf{w}^\top \mathbf{x}_i + b$:

$$\frac{\partial \mathcal{L}}{\partial w_j} = -\frac{2}{n}\sum_{i=1}^{n} r_i \, x_{ij}$$

In matrix form:

$$\nabla_{\mathbf{w}} \mathcal{L} = -\frac{2}{n}X^\top \mathbf{r}$$

where $X \in \mathbb{R}^{n \times d}$ is the data matrix and $\mathbf{r} = \mathbf{y} - X\mathbf{w} \in \mathbb{R}^n$ is the residual vector.

| Symbol              | Shape    | Meaning                                         |
| ------------------- | -------- | ----------------------------------------------- |
| $X$                 | $(n, d)$ | Data matrix: $n$ samples, $d$ features          |
| $\mathbf{w}$        | $(d,)$   | Weight vector                                   |
| $\mathbf{r}$        | $(n,)$   | Residual vector: $\mathbf{y} - X\mathbf{w}$     |
| $X^\top \mathbf{r}$ | $(d,)$   | Gradient direction (same shape as $\mathbf{w}$) |

> **Intuition.** The gradient $-\frac{2}{n}X^\top \mathbf{r}$ is the correlation between each feature and the prediction error. If feature $j$ is positively correlated with the residual (the model under-predicts when feature $j$ is large), then $w_j$ should increase — which is exactly the direction the negative gradient points.

> **Notebook reference.** In `starter.ipynb` Cell 5 (Gradient Descent Dynamics), the gradient of the quadratic loss `quadratic(x, y) = x² + 3y²` is computed analytically as `(2x, 6y)`. This is a special case of the MSE gradient where the "data" is a single point at the origin.

---

### 3.3 Cross-Entropy Loss

For classification, where the model outputs a probability $\hat{y}_i \in (0, 1)$ for binary tasks:

$$\mathcal{L}_{\text{BCE}}(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i)\right]$$

For multi-class classification with $K$ classes and softmax output $\hat{y}_{ik} = P(\text{class } k \mid \mathbf{x}_i)$:

$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \ln \hat{y}_{ik}$$

| Symbol         | Meaning                                                         |
| -------------- | --------------------------------------------------------------- |
| $y_{ik}$       | 1 if sample $i$ belongs to class $k$, else 0 (one-hot encoding) |
| $\hat{y}_{ik}$ | Predicted probability that sample $i$ belongs to class $k$      |
| $\ln$          | Natural logarithm (base $e$)                                    |

**Properties:**
- $\mathcal{L}_{\text{CE}} \geq 0$, and $\mathcal{L}_{\text{CE}} = 0$ only if $\hat{y}_{ik} = y_{ik}$ for all $i, k$.
- The logarithm heavily penalises confident wrong predictions: $-\ln(0.01) \approx 4.6$ vs $-\ln(0.5) \approx 0.69$.
- Cross-entropy equals the KL divergence between the true label distribution and the predicted distribution, plus the entropy of the true distribution (a constant).
- Probabilistically, it is the negative log-likelihood under a Bernoulli (binary) or Categorical (multi-class) model ([Week 07](../../03_probability/week07_likelihood/theory.md#52-bernoulli-noise-binary-cross-entropy)).

> **When to use which loss.** MSE for regression (continuous targets). Cross-entropy for classification (discrete targets). Using MSE for classification is technically possible but leads to pathological gradient behaviour: the gradient vanishes when the model is confidently wrong, precisely when it should be largest. Cross-entropy avoids this by the logarithmic penalty.

---

### 3.4 Loss Landscapes

A **loss landscape** is the graph of $\mathcal{L}(\theta)$ over the parameter space $\Theta$. For $p = 2$ parameters, it is a surface in $\mathbb{R}^3$ (two parameter axes + one loss axis). For $p > 3$, direct visualisation is impossible, but the same geometric concepts (valleys, ridges, saddle points) apply.

**Key landscape features:**

| Feature            | Definition                                                                                            | Implication for optimisation  |
| ------------------ | ----------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Global minimum** | The point(s) with the lowest loss over all of $\Theta$                                                | The ideal target              |
| **Local minimum**  | A point where loss is lower than all nearby points, but not necessarily globally                      | GD can get trapped here       |
| **Saddle point**   | A point where $\nabla \mathcal{L} = 0$ but it is a minimum in some directions and a maximum in others | GD stalls; noise helps escape |
| **Plateau**        | A flat region where $\|\nabla \mathcal{L}\| \approx 0$                                                | GD makes negligible progress  |
| **Ravine**         | A narrow valley where curvature differs greatly across directions                                     | GD zigzags; momentum helps    |

> **Notebook reference.** Cell 3 of `starter.ipynb` visualises two landscapes: a **quadratic** $\mathcal{L}(x, y) = x^2 + 3y^2$ (convex, single minimum at the origin) and a **multimodal** $\mathcal{L}(x, y) = \sin(x)\cos(y) + 0.1(x^2 + y^2)$ (non-convex, multiple minima). The contour plots are critical for developing intuition about how optimisers navigate these surfaces.
>
> **Suggested modification.** Change the quadratic to $\mathcal{L}(x, y) = x^2 + 30y^2$ (increasing the $y$ coefficient from 3 to 30). Observe how the contours become extremely elongated ellipses. Then run gradient descent on both versions and compare the number of steps to convergence. The elongated version will exhibit severe zigzagging — this demonstrates the effect of **conditioning** on convergence (Section 7.3).

---

## 4. Gradient Descent

### 4.1 The Gradient

The **gradient** of the loss with respect to the parameters is:

$$\nabla_\theta \mathcal{L} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial \theta_1} \\ \frac{\partial \mathcal{L}}{\partial \theta_2} \\ \vdots \\ \frac{\partial \mathcal{L}}{\partial \theta_p} \end{bmatrix} \in \mathbb{R}^p$$

The gradient has two fundamental properties:
1. **Direction:** it points in the direction of steepest ascent of $\mathcal{L}$.
2. **Magnitude:** $\|\nabla_\theta \mathcal{L}\|$ is the rate of increase in that direction.

These follow from the first-order Taylor expansion: for a small perturbation $\boldsymbol{\delta}$,

$$\mathcal{L}(\theta + \boldsymbol{\delta}) \approx \mathcal{L}(\theta) + \nabla_\theta \mathcal{L}^\top \boldsymbol{\delta}$$

The change $\nabla_\theta \mathcal{L}^\top \boldsymbol{\delta}$ is maximised (for fixed $\|\boldsymbol{\delta}\|$) when $\boldsymbol{\delta}$ is parallel to $\nabla_\theta \mathcal{L}$ (by the Cauchy–Schwarz inequality). Therefore, to **decrease** the loss maximally, one should step in the direction $-\nabla_\theta \mathcal{L}$.

---

### 4.2 The Update Rule

**Gradient descent** is the iterative algorithm:

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)$$

| Symbol                                | Meaning                                      |
| ------------------------------------- | -------------------------------------------- |
| $\theta_t$                            | Parameter vector at iteration $t$            |
| $\eta > 0$                            | Learning rate (step size)                    |
| $\nabla_\theta \mathcal{L}(\theta_t)$ | Gradient evaluated at the current parameters |
| $\theta_{t+1}$                        | Updated parameters                           |

Each iteration:
1. Evaluates the gradient at the current point $\theta_t$.
2. Takes a step of size $\eta$ in the negative gradient direction.
3. Repeats until a stopping criterion is met (convergence, maximum iterations, or validation loss increase).

**Implementation.** The following is the gradient descent loop from `starter.ipynb` (Cell 5), annotated:

```python
def gradient_descent(x0, y0, grad_fn, lr=0.1, n_steps=50):
    """
    Vanilla gradient descent on a 2D loss function.
    
    Parameters
    ----------
    x0, y0 : float       — Initial parameter values (θ₀)
    grad_fn : callable    — Returns (∂L/∂x, ∂L/∂y) at (x, y)
    lr : float            — Learning rate η
    n_steps : int         — Number of iterations T
    
    Returns
    -------
    trajectory : list of (x, y) tuples — the path through parameter space
    """
    trajectory = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n_steps):
        gx, gy = grad_fn(x, y)       # Step 1: compute gradient
        x -= lr * gx                   # Step 2: update x ← x - η·∂L/∂x
        y -= lr * gy                   # Step 2: update y ← y - η·∂L/∂y
        trajectory.append((x, y))
    return trajectory
```

> **Key observation.** The gradient always has the **same shape** as the parameter vector. If $\theta$ has $p$ components, the gradient has $p$ components, and the update is a component-wise subtraction. This is true whether $p = 2$ (as above) or $p = 10^9$ (a large language model).

---

### 4.3 Convergence Analysis for Quadratic Losses

To build precise intuition, consider the simplest non-trivial case: a quadratic loss with two parameters.

$$\mathcal{L}(w_1, w_2) = \frac{1}{2}\left(\lambda_1 w_1^2 + \lambda_2 w_2^2\right)$$

where $\lambda_1, \lambda_2 > 0$ are the eigenvalues of the Hessian (the curvatures along each axis). The minimum is at $\mathbf{w}^* = (0, 0)$.

| Symbol                                     | Meaning                                                   |
| ------------------------------------------ | --------------------------------------------------------- |
| $\lambda_1, \lambda_2$                     | Eigenvalues of the Hessian — curvatures along each axis   |
| $\kappa = \lambda_{\max} / \lambda_{\min}$ | Condition number — ratio of largest to smallest curvature |

The gradient is:

$$\nabla \mathcal{L} = \begin{bmatrix} \lambda_1 w_1 \\ \lambda_2 w_2 \end{bmatrix}$$

The gradient descent update for each component is:

$$w_{j}^{(t+1)} = w_{j}^{(t)} - \eta \lambda_j w_{j}^{(t)} = (1 - \eta\lambda_j) w_{j}^{(t)}$$

This is a geometric sequence. Starting from $w_j^{(0)}$:

$$w_j^{(t)} = (1 - \eta\lambda_j)^t \, w_j^{(0)}$$

**Convergence condition.** The sequence converges to 0 if and only if:

$$|1 - \eta\lambda_j| < 1 \quad \Longleftrightarrow \quad 0 < \eta\lambda_j < 2 \quad \Longleftrightarrow \quad \eta < \frac{2}{\lambda_j}$$

For convergence along **all** directions simultaneously:

$$\boxed{\eta < \frac{2}{\lambda_{\max}}}$$

This is the **maximum stable learning rate**. Beyond this threshold, the parameter along the direction of highest curvature oscillates with growing amplitude — the algorithm **diverges**.

**Convergence rate.** The slowest-converging direction is the one with the smallest eigenvalue. The convergence factor per step is:

$$\rho = \max_j |1 - \eta \lambda_j|$$

The optimal learning rate (minimising $\rho$) is:

$$\eta^* = \frac{2}{\lambda_{\max} + \lambda_{\min}}, \qquad \rho^* = \frac{\kappa - 1}{\kappa + 1}$$

where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number. When $\kappa$ is large (ill-conditioned problem), $\rho^*$ is close to 1 and convergence is slow.

> **Example.** For the notebook's quadratic $\mathcal{L}(x, y) = x^2 + 3y^2$:
> - $\lambda_1 = 2$ (the Hessian of $x^2$ is $2$), $\lambda_2 = 6$ (the Hessian of $3y^2$ is $6$).
> - Maximum stable learning rate: $\eta < 2/6 \approx 0.333$.
> - Condition number: $\kappa = 6/2 = 3$.
> - Optimal learning rate: $\eta^* = 2/(2 + 6) = 0.25$, giving $\rho^* = (3-1)/(3+1) = 0.5$.
>
> **Notebook reference.** In `starter.ipynb` Cell 8, the learning rate sweep uses $\eta \in \{0.01, 0.05, 0.1, 0.2, 0.25\}$ and plots the resulting trajectories. Note that $\eta = 0.25$ is the theoretically optimal rate. Exercise 5 asks you to test $\eta \in \{0.3, 0.4, 0.5, 0.6\}$ — since $0.333$ is the divergence threshold, $\eta \geq 0.4$ should diverge.
>
> **Suggested experiment.** Change the loss to $\mathcal{L}(x, y) = x^2 + 100y^2$ (condition number $\kappa = 100$). Run the LR sweep and observe that the working range of $\eta$ shrinks to $(0, 2/200) = (0, 0.01)$ and convergence within that range is extremely slow. This demonstrates concretely that **ill-conditioning is the central challenge of gradient descent** — and motivates the adaptive methods of [Week 02](../week02_advanced_optimizers/theory.md#5-adagrad-per-parameter-learning-rates).

---

### 4.4 The Learning Rate

The learning rate $\eta$ is the single most important hyperparameter in all of ML.

| Regime                 | $\eta$ value                                             | Behaviour                                                                                             |
| ---------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Too small**          | $\eta \ll \frac{2}{\lambda_{\max} + \lambda_{\min}}$     | Convergence is guaranteed but extremely slow; may not reach a good solution within the compute budget |
| **Just right**         | $\eta \approx \frac{2}{\lambda_{\max} + \lambda_{\min}}$ | Fast convergence; the loss decreases rapidly and smoothly                                             |
| **Too large**          | $\eta > \frac{2}{\lambda_{\max}}$                        | Divergence — the loss increases without bound                                                         |
| **Slightly too large** | $\eta$ close to $\frac{2}{\lambda_{\max}}$               | Convergence but with oscillation along the direction of highest curvature                             |

> **Intuition.** If you are walking downhill with your eyes closed and taking large steps, you risk overshooting the valley bottom and ending up *higher* on the opposite slope. The steeper the valley walls (large $\lambda_{\max}$), the smaller your steps must be.

**The learning rate dilemma.** In a well-conditioned problem ($\kappa \approx 1$), a single learning rate works well for all directions. In an ill-conditioned problem ($\kappa \gg 1$), the learning rate must be small enough for the steep direction, which makes it painfully slow for the shallow direction. This dilemma is resolved by:
1. **Momentum** (Section 6): accumulates velocity in persistent directions.
2. **Adaptive methods** ([Week 02](../week02_advanced_optimizers/theory.md#7-adam-combining-momentum-and-adaptivity)): maintain per-parameter effective learning rates (Adam, RMSProp).
3. **Preconditioning** (advanced): transform the parameter space to improve conditioning.

---

### 4.5 Full-Batch Gradient Descent in Matrix Form

For a linear model $\hat{\mathbf{y}} = X\mathbf{w}$ with MSE loss:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2 = \frac{1}{n}(\mathbf{y} - X\mathbf{w})^\top(\mathbf{y} - X\mathbf{w})$$

**Gradient derivation.** Expanding:

$$\mathcal{L} = \frac{1}{n}\left(\mathbf{y}^\top \mathbf{y} - 2\mathbf{y}^\top X\mathbf{w} + \mathbf{w}^\top X^\top X \mathbf{w}\right)$$

Using the vector calculus identities from [Week 00b](../../01_intro/week00b_math_and_data/theory.md#37-vector-calculus-identities-for-ml) (Section 3.7):
- $\nabla_{\mathbf{w}}(\mathbf{y}^\top X\mathbf{w}) = X^\top \mathbf{y}$
- $\nabla_{\mathbf{w}}(\mathbf{w}^\top X^\top X \mathbf{w}) = 2X^\top X \mathbf{w}$

Therefore:

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{n}\left(-2X^\top \mathbf{y} + 2X^\top X \mathbf{w}\right) = \frac{2}{n}X^\top(X\mathbf{w} - \mathbf{y}) = -\frac{2}{n}X^\top \mathbf{r}$$

| Symbol                                  | Shape    | Meaning                               |
| --------------------------------------- | -------- | ------------------------------------- |
| $X$                                     | $(n, d)$ | Data matrix                           |
| $\mathbf{w}$                            | $(d,)$   | Weight vector                         |
| $\mathbf{r} = \mathbf{y} - X\mathbf{w}$ | $(n,)$   | Residual vector                       |
| $X^\top \mathbf{r}$                     | $(d,)$   | Feature-weighted residuals            |
| $\nabla_\mathbf{w} \mathcal{L}$         | $(d,)$   | Gradient — same shape as $\mathbf{w}$ |

The full-batch gradient descent update becomes:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \frac{2\eta}{n}X^\top (\mathbf{y} - X\mathbf{w}_t)$$

```python
# Full-batch gradient descent for linear regression
def train_linear_gd(X, y, lr=0.01, n_steps=1000):
    n, d = X.shape
    w = np.zeros(d)                          # initialise at origin
    losses = []
    for t in range(n_steps):
        r = y - X @ w                        # residuals:    (n,)
        grad = -2/n * X.T @ r               # gradient:     (d,)
        w = w - lr * grad                    # update:       (d,)
        loss = np.mean(r**2)                 # MSE loss:     scalar
        losses.append(loss)
    return w, losses
```

> **Shape check.** $X$ is $(n, d)$, $X^\top$ is $(d, n)$, $\mathbf{r}$ is $(n,)$, so $X^\top \mathbf{r}$ is $(d,)$. The gradient has the same shape $(d,)$ as $\mathbf{w}$. ✓

---

## 5. Stochastic Gradient Descent (SGD)

### 5.1 Motivation

Full-batch gradient descent computes the gradient over **all** $n$ samples before each update:

$$\nabla_\theta \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\nabla_\theta \ell(y_i, f_\theta(\mathbf{x}_i))$$

For large $n$ (e.g., $n = 10^6$), this is expensive. The per-sample gradient $\nabla_\theta \ell_i$ is an unbiased estimator of the full gradient:

$$\mathbb{E}_i\left[\nabla_\theta \ell_i\right] = \frac{1}{n}\sum_{i=1}^{n}\nabla_\theta \ell_i = \nabla_\theta \mathcal{L}$$

| Symbol                                       | Meaning                                      |
| -------------------------------------------- | -------------------------------------------- |
| $\ell_i = \ell(y_i, f_\theta(\mathbf{x}_i))$ | Per-sample loss for sample $i$               |
| $\nabla_\theta \ell_i$                       | Gradient of the loss for sample $i$ alone    |
| $\mathbb{E}_i[\cdot]$                        | Expectation over uniformly random sample $i$ |

Therefore, using a **single random sample** (or a small batch) to estimate the gradient introduces noise but does not introduce systematic bias. Provided the learning rate is appropriately controlled, the algorithm still converges.

> **Analogy.** Imagine polling voters before an election. Asking every citizen (full-batch) gives an exact result but is expensive. Asking a random subset (mini-batch) gives a noisy but unbiased estimate, at a fraction of the cost. You can take many more polls in the same time, and the noise averages out.

---

### 5.2 Mini-Batch SGD

In practice, one uses a **mini-batch** $\mathcal{B} \subset \{1, \ldots, n\}$ of size $B$:

$$\nabla_\theta \mathcal{L}_\mathcal{B} = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_\theta \ell_i$$

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}_\mathcal{B}$$

| Symbol                                  | Meaning                                                    |
| --------------------------------------- | ---------------------------------------------------------- |
| $\mathcal{B}$                           | Mini-batch — a randomly sampled subset of training indices |
| $B =                                    | \mathcal{B}                                                | $ | Batch size |
| $\nabla_\theta \mathcal{L}_\mathcal{B}$ | Mini-batch gradient estimate                               |

**Algorithm: one epoch of mini-batch SGD.**

```
Shuffle the training indices {1, ..., n}
Partition into mini-batches B₁, B₂, ..., B_{n/B}
For each mini-batch Bₖ:
    Compute mini-batch gradient: g = (1/B) Σ_{i ∈ Bₖ} ∇ℓ_i
    Update parameters: θ ← θ - η·g
```

One full pass through all mini-batches is one **epoch**.

**Batch size spectrum:**

| Batch size $B$       | Name           | Update frequency    | Gradient noise | Memory       |
| -------------------- | -------------- | ------------------- | -------------- | ------------ |
| $1$                  | Stochastic GD  | $n$ updates/epoch   | Very high      | Minimal      |
| $32$–$256$ (typical) | Mini-batch SGD | $n/B$ updates/epoch | Moderate       | Moderate     |
| $n$                  | Full-batch GD  | 1 update/epoch      | Zero           | Full dataset |

> **Notebook reference.** Exercise 1 in `starter.ipynb` (Cell 10) asks you to implement SGD on a small regression dataset and compare its noisy trajectory to full-batch GD. Observe that SGD trajectories are jagged but converge to the same region.
>
> **Suggested experiment.** Run SGD with batch sizes $B \in \{1, 4, 16, 64, n\}$ on the same dataset. For each, plot the loss curve over iterations. You will observe:
> - $B = 1$: very noisy; the loss fluctuates wildly but trends downward.
> - $B = n$: smooth but slow per-epoch improvement.
> - Intermediate $B$: the "sweet spot" — enough progress per epoch with tolerable noise.

---

### 5.3 Variance and the Noise–Speed Trade-off

The variance of the mini-batch gradient estimate is:

$$\text{Var}\left[\nabla_\theta \mathcal{L}_\mathcal{B}\right] = \frac{\sigma_g^2}{B}$$

where $\sigma_g^2$ is the variance of the per-sample gradients. Increasing $B$ reduces the noise by $1/\sqrt{B}$ (standard deviation scales as $\sigma_g / \sqrt{B}$).

> **The linear scaling rule.** When increasing $B$ by a factor $k$, the gradient noise decreases by $\sqrt{k}$. To maintain the same effective noise-to-signal ratio, the learning rate should be increased by a factor $k$: $\eta \to k\eta$. This "linear scaling rule" (Goyal et al., 2017) is widely used in large-scale training.

**Why noise can be beneficial:**
1. **Escaping local minima and saddle points.** The noise in SGD acts as implicit exploration, pushing the optimiser out of shallow traps.
2. **Implicit regularisation.** SGD noise biases the optimiser toward flatter minima, which tend to generalise better (Keskar et al., 2017). Flat minima are robust to perturbation in parameters, which means they are robust to the distributional shift between training and test data.
3. **Computational efficiency.** Many noisy steps are cheaper and often faster (wall-clock) than few exact steps.

---

## 6. Momentum

### 6.1 The Physics Analogy

Consider a ball rolling on the loss surface under gravity. Unlike gradient descent (which is more like a hiker taking steps), a ball has **inertia**: it accelerates in the direction of the slope and maintains velocity even on flat regions or slight uphill slopes. This physical intuition motivates the **momentum** method.

| Physical concept    | Optimisation analogue                                                    |
| ------------------- | ------------------------------------------------------------------------ |
| Position            | Parameters $\theta$                                                      |
| Velocity            | Accumulated gradient history $\mathbf{v}$                                |
| Gravitational force | Negative gradient $-\nabla \mathcal{L}$                                  |
| Friction            | The decay factor $1 - \beta$ (where $\beta$ is the momentum coefficient) |
| Mass                | Implicitly 1; can be absorbed into the learning rate                     |

---

### 6.2 The Momentum Update Rule

The classical (Polyak) momentum update maintains a **velocity** vector $\mathbf{v}$ that accumulates past gradients with exponential decay:

$$\mathbf{v}_{t+1} = \beta \, \mathbf{v}_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)$$

$$\theta_{t+1} = \theta_t + \mathbf{v}_{t+1}$$

| Symbol             | Meaning                                                      | Typical value               |
| ------------------ | ------------------------------------------------------------ | --------------------------- |
| $\mathbf{v}_t$     | Velocity vector at iteration $t$                             | Initialised to $\mathbf{0}$ |
| $\beta \in [0, 1)$ | Momentum coefficient — controls how much history is retained | $0.9$                       |
| $\eta$             | Learning rate                                                | Problem-dependent           |

**Unrolling the recursion.** The velocity at step $t$ is:

$$\mathbf{v}_t = -\eta \sum_{s=0}^{t-1} \beta^{t-1-s} \nabla_\theta \mathcal{L}(\theta_s)$$

This is an exponentially weighted moving average of past gradients. Recent gradients have weight $\approx 1$; a gradient from $k$ steps ago has weight $\beta^k$. The **effective window** is approximately $1/(1 - \beta)$: for $\beta = 0.9$, the last $\sim 10$ gradients dominate.

**Notebook reference.** Cell 7 of `starter.ipynb` implements momentum:

```python
def gradient_descent_momentum(x0, y0, grad_fn, lr=0.1, momentum=0.9, n_steps=50):
    trajectory = [(x0, y0)]
    x, y = x0, y0
    vx, vy = 0.0, 0.0                       # initialise velocity to 0
    for _ in range(n_steps):
        gx, gy = grad_fn(x, y)
        vx = momentum * vx - lr * gx        # update velocity
        vy = momentum * vy - lr * gy
        x += vx; y += vy                     # update position
        trajectory.append((x, y))
    return trajectory
```

---

### 6.3 Why Momentum Helps

**Problem 1: Ravines (ill-conditioned loss).** In a narrow valley where the curvature is very different along the two axes, gradient descent oscillates across the valley while making slow progress along its floor. Momentum dampens the oscillation (the transverse gradient components cancel over steps) and amplifies the consistent downhill component.

Formally, for the quadratic $\mathcal{L}(w_1, w_2) = \frac{1}{2}(\lambda_1 w_1^2 + \lambda_2 w_2^2)$ with $\lambda_1 \ll \lambda_2$:
- The gradient along $w_2$ is large but oscillating (it changes sign each step).
- The gradient along $w_1$ is small but consistent (it always points toward 0).
- Momentum accumulates the consistent $w_1$ component while the oscillating $w_2$ component cancels, yielding faster net progress.

> **Suggested experiment.** In the notebook, compare vanilla GD and momentum on $\mathcal{L}(x, y) = x^2 + 50y^2$ (high condition number). Use the same initial point and learning rate. Plot both trajectories on a contour plot. GD should zigzag wildly along $y$; momentum should dampen the oscillation and converge faster. Then vary $\beta \in \{0.0, 0.5, 0.9, 0.99\}$ and observe how higher momentum smooths the trajectory but can overshoot if too aggressive.

**Problem 2: Flat regions and saddle points.** In a flat region, the gradient is near zero and GD stalls. But a ball with momentum coasts through flat regions using its accumulated velocity.

**Problem 3: Gradient noise (SGD).** When using mini-batch gradients, the noise causes random fluctuations. Momentum averages these out (the noise is zero-mean), effectively reducing the variance of the update direction.

---

### 6.4 Nesterov Accelerated Gradient

Nesterov momentum (NAG) is a subtle improvement: instead of computing the gradient at the current position $\theta_t$, it computes the gradient at the **lookahead** position $\theta_t + \beta \mathbf{v}_t$ — where momentum would take us if the gradient were zero:

$$\mathbf{v}_{t+1} = \beta \, \mathbf{v}_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t + \beta \mathbf{v}_t)$$

$$\theta_{t+1} = \theta_t + \mathbf{v}_{t+1}$$

> **Intuition.** Classical momentum is like throwing a ball and then checking where it lands. Nesterov momentum is like looking ahead to where the ball *will* land and correcting course before it arrives. This "lookahead correction" reduces overshooting and yields provably better convergence rates for convex optimisation.

**Convergence rates (convex, $L$-smooth functions):**

| Algorithm         | Convergence rate                     | Note                                             |
| ----------------- | ------------------------------------ | ------------------------------------------------ |
| Gradient descent  | $\mathcal{O}(1/t)$                   | Linear convergence for strongly convex           |
| Momentum          | $\mathcal{O}(1/t)$ (same asymptotic) | Practical improvement via damping oscillation    |
| Nesterov momentum | $\mathcal{O}(1/t^2)$                 | Optimal for first-order methods (Nesterov, 1983) |

Nesterov's $\mathcal{O}(1/t^2)$ convergence rate is provably optimal among all methods that only use gradient information — no first-order method can do better in the worst case.

---

## 7. The Geometry of Loss Landscapes

### 7.1 Convex Functions

A function $f : \mathbb{R}^d \to \mathbb{R}$ is **convex** if, for all $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0, 1]$:

$$f(\lambda \mathbf{x} + (1 - \lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1 - \lambda) f(\mathbf{y})$$

Geometrically: the line segment connecting any two points on the graph of $f$ lies **above** the graph. Equivalently, $f$ is convex if and only if its Hessian $H(\mathbf{x})$ is positive semi-definite everywhere.

**Consequences for optimisation:**
- Every local minimum is a **global** minimum.
- Gradient descent with appropriate $\eta$ converges to the global minimum.
- No saddle points exist (for strictly convex functions).

**Examples of convex losses in ML:**
- MSE with a linear model: $\mathcal{L}(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2$. The Hessian is $\frac{2}{n}X^\top X$, which is PSD. ✓
- Cross-entropy with a logistic model (convex in $\mathbf{w}$, proven in [Week 03](../week03_linear_models/theory.md#7-logistic-regression)).

**Non-convex losses:**
- Any neural network with hidden layers. The composition of layers with nonlinear activations destroys convexity.

> **Why convexity matters.** For convex problems, optimisation theory provides strong guarantees: gradient descent will find the best solution if given enough time. For non-convex problems (neural networks), there are no such guarantees — but in practice, SGD with momentum finds solutions that generalise well. Understanding the convex case first provides the baseline intuition.

---

### 7.2 Non-Convex Landscapes, Local Minima, and Saddle Points

A **critical point** is a point where $\nabla \mathcal{L} = \mathbf{0}$. The Hessian $H$ at a critical point classifies it:

| Hessian at critical point | All eigenvalues              | Classification    |
| ------------------------- | ---------------------------- | ----------------- |
| Positive definite         | $\lambda_i > 0 \; \forall i$ | **Local minimum** |
| Negative definite         | $\lambda_i < 0 \; \forall i$ | **Local maximum** |
| Indefinite                | Mixed signs                  | **Saddle point**  |

> **Definition recap.** The Hessian $H = \nabla^2 \mathcal{L}$ is the $p \times p$ matrix of second derivatives ([Week 00b](../../01_intro/week00b_math_and_data/theory.md#34-the-jacobian-and-the-hessian), Section 3.4). Its eigenvalues are the curvatures along the principal directions. A positive eigenvalue means the loss curves upward in that direction (a valley); a negative eigenvalue means it curves downward (a hill).

> **Notebook reference.** Exercise 2 in `starter.ipynb` (Cell 11) asks you to define the classic saddle function $f(x, y) = x^2 - y^2$ and observe GD behaviour. The Hessian is $\text{diag}(2, -2)$: one positive eigenvalue (minimum in $x$), one negative (maximum in $y$). At the origin, $\nabla f = \mathbf{0}$ but it is not a minimum — it is a saddle.
>
> **Suggested experiment.** Start GD at $(0.01, 0.01)$ on $f(x, y) = x^2 - y^2$. Observe that $x$ converges to 0 but $y$ diverges — the optimiser is repelled from the saddle along $y$. Now add small Gaussian noise to the gradients (simulating SGD). The noise provides the perturbation that helps escape saddle points. Try different noise magnitudes and observe the effect.

---

### 7.3 Conditioning and the Hessian

The **condition number** of the Hessian at a minimum determines how difficult the optimisation is:

$$\kappa = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$$

| Condition number   | Landscape shape                         | GD behaviour                 |
| ------------------ | --------------------------------------- | ---------------------------- |
| $\kappa \approx 1$ | Circular/spherical contours             | Fast, direct convergence     |
| $\kappa \gg 1$     | Elongated elliptical contours (ravines) | Slow, zigzagging convergence |

For the notebook's quadratic $\mathcal{L}(x, y) = x^2 + 3y^2$:

$$H = \begin{bmatrix} 2 & 0 \\ 0 & 6 \end{bmatrix}, \quad \kappa = \frac{6}{2} = 3$$

This is well-conditioned. The contours are moderately elongated ellipses.

For the modified $\mathcal{L}(x, y) = x^2 + 100y^2$:

$$H = \begin{bmatrix} 2 & 0 \\ 0 & 200 \end{bmatrix}, \quad \kappa = \frac{200}{2} = 100$$

This is ill-conditioned. The contours are extremely elongated, and GD requires $\eta < 2/200 = 0.01$ to converge — while the shallow $x$-direction would benefit from $\eta$ up to $2/2 = 1$.

> **The fundamental tension.** In an ill-conditioned problem, no single learning rate is good for all directions. The steep direction demands a small $\eta$ (to avoid divergence), while the shallow direction wants a large $\eta$ (to make progress). This is the problem that adaptive methods like Adam ([Week 02](../week02_advanced_optimizers/theory.md#7-adam-combining-momentum-and-adaptivity)) solve: they maintain a separate effective learning rate for each parameter.

> **Physical analogy.** An ill-conditioned loss is like a long, narrow bowling lane. The gutter walls are steep (you get penalised heavily for going sideways), but the lane is nearly flat along its length (you need large steps to make progress toward the pins). Gradient descent keeps bouncing off the gutter walls instead of rolling straight.

---

### 7.4 Saddle Points in High Dimensions

A critical insight for neural network training: in high-dimensional spaces ($p \gg 1$), **saddle points are exponentially more common than local minima**.

**Heuristic argument.** At a critical point ($\nabla \mathcal{L} = \mathbf{0}$), each of the $p$ Hessian eigenvalues is independently positive (valley) or negative (hill). For a local minimum, **all** $p$ eigenvalues must be positive — this requires $p$ independent events to all go one way. The probability of this decreases exponentially with $p$. Saddle points (mixed signs) are the generic case.

**Formal result (Bray & Dean, 2007; Dauphin et al., 2014).** For random high-dimensional loss functions (and neural networks empirically), the fraction of negative eigenvalues at a critical point correlates with the loss value. Low-loss critical points tend to have few negative eigenvalues (they are "almost minima"), while high-loss critical points have many negative eigenvalues (they are saddle points). This means:

1. Getting trapped at a high-loss local minimum is unlikely — high-loss critical points are almost always saddle points with escape routes.
2. The real risk is not local minima but **slow traversal through saddle-point regions** where the gradient is small.
3. SGD noise helps escape saddle points by providing perturbations along the directions of negative curvature.

> **Practical implication.** When training a neural network and the loss plateaus, the likely cause is a saddle point, not a local minimum. Increasing the learning rate, adding momentum, or using a larger batch size (to reduce noise for more deterministic exploration) can help escape.

---

## 8. Learning Rate Selection

### 8.1 The Divergence Threshold

From Section 4.3, for a quadratic loss with maximum Hessian eigenvalue $\lambda_{\max}$:

$$\eta_{\max} = \frac{2}{\lambda_{\max}}$$

For general smooth functions, the condition is:

$$\eta < \frac{2}{L}$$

where $L$ is the **Lipschitz constant of the gradient** (the maximum curvature): $\|\nabla \mathcal{L}(\theta_1) - \nabla \mathcal{L}(\theta_2)\| \leq L \|\theta_1 - \theta_2\|$ for all $\theta_1, \theta_2$.

| Symbol              | Meaning                                                       |
| ------------------- | ------------------------------------------------------------- |
| $L$                 | Lipschitz constant of the gradient (upper bound on curvature) |
| $\eta_{\max} = 2/L$ | Maximum learning rate for convergence                         |

In practice, $L$ is unknown and one determines $\eta$ empirically (Section 8.3).

> **Notebook reference.** Exercise 5 in `starter.ipynb` (Cell 14) asks you to empirically find the divergence threshold for the quadratic loss. The theoretical answer is $\eta_{\max} = 2/6 \approx 0.333$, so you should observe: $\eta = 0.3$ converges (barely), $\eta = 0.4$ diverges.

---

### 8.2 Learning Rate Schedules

A fixed learning rate is suboptimal: one wants a large $\eta$ early (to make rapid progress) and a small $\eta$ later (to fine-tune near the minimum without oscillation). **Learning rate schedules** formalise this idea.

| Schedule              | Formula                                                                         | Character                                                  |
| --------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Constant**          | $\eta_t = \eta_0$                                                               | Baseline; simple but not optimal                           |
| **Step decay**        | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$                          | Discrete drops every $s$ steps by factor $\gamma$          |
| **Exponential decay** | $\eta_t = \eta_0 \cdot e^{-\alpha t}$                                           | Smooth continuous decay                                    |
| **Cosine annealing**  | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t / T))$ | Smooth decay from $\eta_0$ to $\eta_{\min}$ over $T$ steps |
| **Warmup + decay**    | Linear ramp from 0 to $\eta_0$ over $T_w$ steps, then cosine decay              | Standard for transformers ([Week 18](../../06_sequence_models/week18_transformers/theory.md#91-learning-rate-warm-up))                        |
| **Cyclical**          | Oscillate between $\eta_{\min}$ and $\eta_{\max}$                               | Escapes local basins (Smith, 2017)                         |

| Symbol        | Meaning                          |
| ------------- | -------------------------------- |
| $\eta_0$      | Initial learning rate            |
| $\eta_{\min}$ | Minimum learning rate            |
| $\gamma$      | Decay factor (e.g., 0.1)         |
| $s$           | Step size (steps between decays) |
| $T$           | Total training steps             |
| $T_w$         | Warmup duration (steps)          |

> **Why warmup?** At initialisation, the model's parameters are random and the loss surface is poorly approximated by the local gradient. A large initial $\eta$ can cause instability. Warming up the learning rate gradually allows the model to reach a region where the gradient is informative before taking large steps. This is especially important for large models ([Week 14](../../05_deep_learning/week14_training_at_scale/theory.md#46-warmup), 18).

> **Notebook reference.** Exercise 3 in `starter.ipynb` (Cell 12) asks you to implement Leslie Smith's cyclical LR policy and compare it to constant LR on the quadratic loss. Implement the triangular schedule and observe whether the cyclical LR escapes shallow basins that a constant LR gets trapped in (this is more visible on the multimodal loss from Cell 3).

---

### 8.3 Leslie Smith's LR Range Test

The **LR range test** (Smith, 2017) is an empirical method for finding a good learning rate:

1. Start with a very small $\eta$ (e.g., $10^{-7}$).
2. Train for one epoch, increasing $\eta$ exponentially after each mini-batch.
3. Plot the loss vs. $\log(\eta)$.

**Interpretation:**  
- The loss first decreases (the learning rate is now large enough to make progress).
- The loss reaches a minimum (the "sweet spot").
- The loss begins to increase or diverge (the learning rate is too large).

Choose $\eta$ at the point where the loss is **still decreasing sharply** — typically one order of magnitude below the $\eta$ that minimises the loss.

```python
def lr_range_test(X, y, model_fn, grad_fn, lr_start=1e-7, lr_end=10, n_steps=200):
    """
    Exponentially increase LR from lr_start to lr_end over n_steps updates.
    Returns (lr_history, loss_history).
    """
    lr_factor = (lr_end / lr_start) ** (1 / n_steps)
    lr = lr_start
    theta = np.zeros(X.shape[1])
    lrs, losses = [], []
    for _ in range(n_steps):
        loss, grad = model_fn(theta, X, y), grad_fn(theta, X, y)
        lrs.append(lr)
        losses.append(loss)
        theta = theta - lr * grad
        lr *= lr_factor
    return lrs, losses
```

> **Suggested experiment.** Implement the LR range test on a synthetic regression dataset (e.g., $y = 3x + \epsilon$) with a linear model. Plot loss vs. log(LR). Identify the "cliff" where the loss starts rising — this is your empirical $\eta_{\max}$. Compare it to the theoretical $2/\lambda_{\max}$ from the Hessian.

---

## 9. Convergence Diagnostics

Monitoring the **training loss curve** ($\mathcal{L}$ vs. iteration $t$) is the primary diagnostic tool. The curve reveals the optimiser's health:

| Curve shape                                   | Diagnosis                                              | Action                                                    |
| --------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------------- |
| Smooth, rapid decrease levelling to a plateau | Healthy convergence                                    | ✓ Good                                                    |
| Very slow decrease (nearly flat)              | Learning rate too small                                | Increase $\eta$                                           |
| Wild oscillation (loss swings up and down)    | Learning rate too large                                | Decrease $\eta$ or add momentum                           |
| Loss increases monotonically                  | Severe divergence                                      | Drastically decrease $\eta$                               |
| Loss decreases then rises                     | Overfitting (if validation loss rises) or LR too large | Add regularisation ([Week 06](../week06_regularization/theory.md#2-why-regularisation-the-overfitting-problem-revisited)) or decrease $\eta$           |
| Loss plateaus at a high value                 | Stuck at saddle / poor initialisation                  | Add noise (SGD), increase capacity, or try different init |

**Additional diagnostics:**

1. **Gradient norm** $\|\nabla_\theta \mathcal{L}\|$ over time. Should decrease as the model approaches a minimum. If it explodes, the learning rate is too large. If it goes to zero prematurely, the model may be stuck at a saddle point.

2. **Parameter norm** $\|\theta\|$ over time. If it grows without bound, the model may be diverging. If it stays near the initial value, the learning rate may be too small to make meaningful updates.

3. **Validation loss** alongside training loss. A gap opening between training loss (decreasing) and validation loss (increasing) signals overfitting — the model is memorising training data rather than learning generalisable patterns.

```python
# Diagnostic: plot training loss curve
plt.semilogy(losses)   # log-scale y-axis reveals convergence rate
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.show()
```

> **Suggested experiment.** In `starter.ipynb`, add a `grad_norm` tracker to the gradient descent function (compute `np.linalg.norm(grad)` at each step). Plot the gradient norm alongside the loss. For the quadratic loss with $\eta = 0.1$, both should decay exponentially. For the divergent case ($\eta = 0.5$), the gradient norm should grow, confirming instability.

---

## 10. The Physical Analogy: Loss as Energy

This section develops the central metaphor of this week: **loss is energy** and **optimisation is the process of a physical system finding its lowest energy state**.

### The Gradient Descent Analogy

| Physics (ball on a surface)               | Optimisation                                                                 |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| Position of the ball                      | Parameters $\theta$                                                          |
| Height / potential energy                 | Loss $\mathcal{L}(\theta)$                                                   |
| Gravitational force ($-\nabla \text{PE}$) | Negative gradient $-\nabla_\theta \mathcal{L}$                               |
| Friction / viscous damping                | In GD: the absence of velocity memory. In momentum: the factor $(1 - \beta)$ |
| Kinetic energy                            | Momentum term $\frac{1}{2}\|\mathbf{v}\|^2$                                  |
| Terminal velocity                         | Effective step size when momentum saturates                                  |
| Temperature                               | Noise magnitude in SGD (related to learning rate and batch size)             |

**Gradient descent without momentum** is an **overdamped** system — the ball moves in the direction of the force but has no inertia. It stops immediately when the surface is flat.

**Gradient descent with momentum** is an **underdamped** system — the ball has inertia. It can overshoot the minimum (oscillate around it) but it can also coast through flat regions and narrow saddle points.

**SGD (with noise)** is analogous to a system at **non-zero temperature** — the ball jiggles randomly due to thermal fluctuations. This helps it escape shallow energy wells (local minima / saddle points) but prevents it from settling exactly at the bottom. Reducing the temperature (annealing the learning rate) allows it to eventually settle.

### Simulated Annealing Connection

The analogy to temperature is more than metaphorical. In **simulated annealing** (a classical optimisation method), the system is initialised at high temperature (large random perturbations) and the temperature is gradually decreased. The probability of accepting a worse solution decreases with temperature: $P(\text{accept}) = e^{-\Delta E / T}$, where $\Delta E$ is the increase in energy and $T$ is the temperature.

In SGD with learning rate decay, the noise magnitude (proportional to $\eta / B$) decreases over training. Early in training, the high noise explores broadly; late in training, the low noise allows fine convergence. LR schedules are, in this sense, **annealing schedules** for stochastic optimisation.

> **Suggested experiment.** Create a multimodal loss function with one deep minimum and several shallow ones (e.g., add wells of different depths to a quadratic base). Run gradient descent with different starting points — observe that it gets trapped in whichever basin it starts in. Then run SGD with large noise (small batch, large LR) and observe that it can hop between basins. Finally, apply a cosine annealing schedule and observe that the ball explores broadly at first and settles into the deep minimum at the end.

---

## 11. Notebook Reference Guide

This section maps each cell of `starter.ipynb` to the theoretical concepts it illustrates.

| Cell                      | Section                                                    | What to observe                                                                                               | Theory reference |
| ------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------- |
| 3 (Loss landscapes)       | Contour plots of quadratic and multimodal functions        | Elliptical contours from $\lambda_1 \neq \lambda_2$; multiple minima in multimodal                            | Section 3.4, 7.1 |
| 5 (GD dynamics)           | GD trajectory on quadratic contours                        | The path curves toward the minimum; faster along the steep direction ($y$) than the shallow direction ($x$)   | Section 4.2, 4.3 |
| 7 (Momentum)              | Momentum vs. vanilla GD trajectories                       | Momentum smooths the path and converges in fewer steps                                                        | Section 6.2, 6.3 |
| 8 (LR sweep)              | Trajectories for $\eta \in \{0.01, 0.05, 0.1, 0.2, 0.25\}$ | Small $\eta$ converges slowly; $\eta = 0.25$ is near-optimal; the final-loss-vs-LR plot should show a U-shape | Section 4.4, 8.1 |
| 10 (Exercise: SGD)        | SGD with noisy gradients                                   | Noisy trajectory but same asymptotic convergence                                                              | Section 5        |
| 11 (Exercise: Saddle)     | $f(x,y) = x^2 - y^2$ with GD                               | GD stalls at the saddle; adding noise helps escape                                                            | Section 7.2, 7.4 |
| 12 (Exercise: CLR)        | Cyclical learning rate                                     | Oscillating LR can escape shallow basins                                                                      | Section 8.2      |
| 13 (Exercise: 3D plots)   | Surface plot with trajectory overlay                       | 3D perspective reveals structure invisible in 2D contours                                                     | Section 3.4      |
| 14 (Exercise: Divergence) | LR values above the divergence threshold                   | Loss grows exponentially; threshold at $\eta = 2/\lambda_{\max}$                                              | Section 4.3, 8.1 |

---

## 12. Symbol Reference

| Symbol                      | Name                   | Meaning                                                     |
| --------------------------- | ---------------------- | ----------------------------------------------------------- |
| $\theta \in \mathbb{R}^p$   | Parameters             | All learnable weights and biases                            |
| $\theta^*$                  | Optimal parameters     | $\arg\min_\theta \mathcal{L}(\theta)$                       |
| $\theta_t$                  | Parameters at step $t$ | Current iterate                                             |
| $p$                         | Parameter count        | Dimension of the parameter space                            |
| $\mathcal{L}(\theta)$       | Loss function          | Scalar measuring prediction error                           |
| $\ell_i$                    | Per-sample loss        | $\ell(y_i, f_\theta(\mathbf{x}_i))$                         |
| $\nabla_\theta \mathcal{L}$ | Gradient               | Vector of partial derivatives; direction of steepest ascent |
| $H = \nabla^2 \mathcal{L}$  | Hessian                | Matrix of second derivatives; encodes curvature             |
| $\lambda_i$                 | Eigenvalue             | Curvature along the $i$-th principal direction              |
| $\lambda_{\max}$            | Largest eigenvalue     | Steepest curvature direction                                |
| $\lambda_{\min}$            | Smallest eigenvalue    | Shallowest curvature direction                              |
| $\kappa$                    | Condition number       | $\lambda_{\max}/\lambda_{\min}$; measures difficulty        |
| $\eta$                      | Learning rate          | Step size per iteration                                     |
| $\eta_{\max}$               | Max stable LR          | $2/\lambda_{\max}$; divergence threshold                    |
| $\mathbf{v}_t$              | Velocity               | Accumulated gradient history (momentum)                     |
| $\beta$                     | Momentum coefficient   | Decay rate for velocity averaging; typical: 0.9             |
| $\mathbf{r}$                | Residual vector        | $\mathbf{y} - \hat{\mathbf{y}}$                             |
| $X$                         | Data matrix            | $(n, d)$: $n$ samples, $d$ features                         |
| $\mathbf{w}$                | Weight vector          | $(d,)$: one weight per feature                              |
| $B$                         | Batch size             | Number of samples per gradient estimate                     |
| $\mathcal{B}$               | Mini-batch             | Random subset of training indices                           |
| $\sigma_g^2$                | Gradient variance      | Variance of per-sample gradient estimates                   |
| $L$                         | Lipschitz constant     | Upper bound on gradient change rate                         |
| $\rho$                      | Convergence factor     | Per-step contraction ratio; smaller is faster               |
| $T$                         | Total steps            | Number of training iterations                               |
| $t$                         | Iteration index        | Current step number                                         |

---

## 13. References

1. Cauchy, A. (1847). "Méthode générale pour la résolution des systèmes d'équations simultanées." *Comptes Rendus de l'Académie des Sciences*, 25, 536–538. — The original gradient descent paper.
2. Polyak, B. T. (1964). "Some methods of speeding up the convergence of iteration methods." *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1–17. — Introduced the momentum method.
3. Nesterov, Y. (1983). "A method of solving a convex programming problem with convergence rate $O(1/k^2)$." *Soviet Mathematics Doklady*, 27, 372–376. — Nesterov accelerated gradient.
4. Robbins, H. & Monro, S. (1951). "A Stochastic Approximation Method." *The Annals of Mathematical Statistics*, 22(3), 400–407. — Foundational work on stochastic approximation (SGD).
5. Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks." *IEEE WACV*, 464–472. — LR range test and cyclical schedules.
6. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." *arXiv:1706.02677*. — Linear scaling rule for learning rate with batch size.
7. Keskar, N. S., et al. (2017). "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." *ICLR*. — Flat vs. sharp minima and generalisation.
8. Dauphin, Y. N., et al. (2014). "Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization." *NeurIPS*. — Saddle points dominate local minima in high dimensions.
9. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 8 (Optimization). MIT Press. Available at [deeplearningbook.org](https://www.deeplearningbook.org).
10. Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*, Chapter 9 (Unconstrained minimization). Cambridge University Press. Available at [stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/).
11. Ruder, S. (2016). "An overview of gradient descent optimization algorithms." *arXiv:1609.04747*. — Excellent survey covering GD, momentum, Adam, and variants.
