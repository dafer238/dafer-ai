# Advanced Optimisation: From SGD to Adam

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [The Problem with Vanilla Gradient Descent](#2-the-problem-with-vanilla-gradient-descent)
3. [Exponential Moving Averages](#3-exponential-moving-averages)
4. [Momentum (Revisited and Formalised)](#4-momentum-revisited-and-formalised)
5. [AdaGrad: Per-Parameter Learning Rates](#5-adagrad-per-parameter-learning-rates)
6. [RMSProp: Fixing AdaGrad's Decay](#6-rmsprop-fixing-adagrads-decay)
7. [Adam: Combining Momentum and Adaptivity](#7-adam-combining-momentum-and-adaptivity)
    - 7.1 [The Adam Algorithm](#71-the-adam-algorithm)
    - 7.2 [Bias Correction](#72-bias-correction)
    - 7.3 [Effective Step Size Analysis](#73-effective-step-size-analysis)
    - 7.4 [Adam's Hyperparameters](#74-adams-hyperparameters)
8. [Variants of Adam](#8-variants-of-adam)
    - 8.1 [AdamW: Decoupled Weight Decay](#81-adamw-decoupled-weight-decay)
    - 8.2 [AMSGrad](#82-amsgrad)
    - 8.3 [RAdam: Rectified Adam](#83-radam-rectified-adam)
9. [Learning Rate Schedules](#9-learning-rate-schedules)
    - 9.1 [Why Schedules Matter](#91-why-schedules-matter)
    - 9.2 [Step Decay](#92-step-decay)
    - 9.3 [Cosine Annealing](#93-cosine-annealing)
    - 9.4 [Warmup](#94-warmup)
    - 9.5 [Warmup + Cosine Annealing](#95-warmup--cosine-annealing)
    - 9.6 [The LR Range Test](#96-the-lr-range-test)
10. [Choosing an Optimiser](#10-choosing-an-optimiser)
11. [Convergence Guarantees and Limitations](#11-convergence-guarantees-and-limitations)
12. [Notebook Reference Guide](#12-notebook-reference-guide)
13. [Symbol Reference](#13-symbol-reference)
14. [References](#14-references)

---

## 1. Scope and Purpose

[Week 01](../week01_optimization/theory.md#4-gradient-descent) introduced vanilla gradient descent and its fundamental limitations: sensitivity to the learning rate, slow convergence on ill-conditioned problems, and inability to adapt to the geometry of the loss surface. This week develops the family of **adaptive optimisers** that address these limitations, culminating in **Adam** — the default optimiser in modern deep learning.

The treatment follows the historical and conceptual progression:

$$\text{SGD} \;\to\; \text{Momentum} \;\to\; \text{AdaGrad} \;\to\; \text{RMSProp} \;\to\; \text{Adam}$$

Each algorithm fixes a specific shortcoming of its predecessor. Understanding this progression — not just memorising update rules — is the goal.

**Prerequisites.** [Week 01](../week01_optimization/theory.md#4-gradient-descent): gradient descent, learning rate, convergence analysis for quadratic losses, the condition number $\kappa$, and the momentum update rule.

---

## 2. The Problem with Vanilla Gradient Descent

Recall the gradient descent update from [Week 01](../week01_optimization/theory.md#42-the-update-rule):

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)$$

Three pathologies motivate the algorithms in this week:

### Pathology 1: Ill-Conditioning

When the loss surface has very different curvatures along different directions (condition number $\kappa = \lambda_{\max}/\lambda_{\min} \gg 1$), gradient descent **zigzags** across narrow valleys while making slow progress along them.

For the quadratic $\mathcal{L}(x, y) = x^2 + 5y^2$ used in the notebook:

$$H = \begin{bmatrix} 2 & 0 \\ 0 & 10 \end{bmatrix}, \quad \kappa = \frac{10}{2} = 5$$

The maximum stable learning rate is $\eta < 2/\lambda_{\max} = 2/10 = 0.2$. At this rate, the $x$-direction (curvature 2) converges slowly — the per-step contraction is $|1 - 0.2 \times 2| = 0.6$, while the $y$-direction contracts by $|1 - 0.2 \times 10| = 1$ (marginal stability).

> **The core issue.** A single scalar $\eta$ cannot simultaneously be large enough for shallow directions and small enough for steep directions.

### Pathology 2: Gradient Noise (SGD)

When using mini-batches, the gradient is noisy. The noise variance is $\sigma_g^2 / B$ ([Week 01](../week01_optimization/theory.md#53-variance-and-the-noisespeed-trade-off), Section 5.3). This noise causes the optimiser to jitter around the minimum rather than converging exactly.

### Pathology 3: Saddle Points and Plateaus

At saddle points, the gradient magnitude vanishes. Vanilla GD stalls because it has no memory of past gradients — each step is computed independently. Momentum ([Week 01](../week01_optimization/theory.md#6-momentum)) partially addresses this, but does not adapt to the local curvature.

> **Summary of remedies:**
>
> | Pathology | Root cause | Remedy | Algorithm |
> |---|---|---|---|
> | Zigzagging in ravines | Single global $\eta$ | Accumulate gradient history | Momentum |
> | Sensitivity to curvature | Uniform step size across parameters | Per-parameter adaptive $\eta$ | AdaGrad, RMSProp |
> | Both simultaneously | Need both direction memory and scale adaptation | Combine momentum + adaptivity | Adam |
> | Stale learning rates | Accumulated history never forgets | Use exponential moving average | RMSProp (fixes AdaGrad) |

---

## 3. Exponential Moving Averages

The **Exponential Moving Average (EMA)** is the mathematical building block of every modern optimiser. Understanding it is essential before proceeding.

Given a sequence of values $g_1, g_2, \ldots$, the EMA with decay parameter $\beta \in [0, 1)$ is:

$$s_t = \beta \, s_{t-1} + (1 - \beta) \, g_t, \qquad s_0 = 0$$

| Symbol  | Meaning                                                                    |
| ------- | -------------------------------------------------------------------------- |
| $g_t$   | The new observation at step $t$ (e.g., a gradient or squared gradient)     |
| $s_t$   | The EMA at step $t$ — a smoothed estimate of the recent average            |
| $\beta$ | Decay factor — how much weight is given to history vs. the new observation |

**Unrolling the recursion:**

$$s_t = (1 - \beta)\sum_{k=1}^{t} \beta^{t-k} g_k$$

This is a weighted average of all past observations, with exponentially decaying weights. The weight of $g_k$ is $(1 - \beta)\beta^{t-k}$.

**Effective window.** The contributions older than $\sim 1/(1 - \beta)$ steps are negligible. For $\beta = 0.9$: effective window $\approx 10$ steps. For $\beta = 0.999$: effective window $\approx 1000$ steps.

| $\beta$ | Effective window      | Character                   |
| ------- | --------------------- | --------------------------- |
| $0.0$   | 1 step (no smoothing) | Reacts instantly, no memory |
| $0.9$   | ~10 steps             | Moderate smoothing          |
| $0.99$  | ~100 steps            | Heavy smoothing             |
| $0.999$ | ~1000 steps           | Very heavy smoothing        |

> **Intuition.** An EMA is like a leaky integrator: at each step, it "leaks" a fraction $(1 - \beta)$ of the old estimate and replaces it with the new observation. High $\beta$ means slow leaking (long memory); low $\beta$ means fast leaking (short memory).

### Bias at Initialisation

Since $s_0 = 0$, the EMA is **biased toward zero** during the first few steps. At step $t = 1$: $s_1 = (1 - \beta)g_1$, which underestimates $g_1$ by a factor of $(1 - \beta)$. The **bias-corrected** estimate is:

$$\hat{s}_t = \frac{s_t}{1 - \beta^t}$$

As $t \to \infty$, $\beta^t \to 0$ and the correction vanishes. At $t = 1$: $\hat{s}_1 = g_1$ (exactly correct). This bias correction is a key component of Adam (Section 7.2).

```python
# EMA demonstration
beta = 0.9
signal = np.random.randn(100)  # noisy signal
ema = np.zeros(100)
for t in range(1, 100):
    ema[t] = beta * ema[t-1] + (1 - beta) * signal[t]

# Bias-corrected
ema_corrected = ema / (1 - beta ** np.arange(1, 101))
```

> **Suggested experiment.** Generate a noisy signal (e.g., a sine wave plus Gaussian noise) and compute EMAs with $\beta \in \{0.5, 0.9, 0.99\}$. Plot all three alongside the raw signal. Observe how higher $\beta$ produces a smoother estimate but introduces more lag (delayed response to changes). This trade-off is identical in optimisers: high $\beta_1$ in Adam smooths the gradient estimates but responds slowly to curvature changes.

---

## 4. Momentum (Revisited and Formalised)

Momentum was introduced in [Week 01](../week01_optimization/theory.md#6-momentum) as a physical analogy. Here, it is presented as a specific instance of the EMA framework.

### The Update Rule

$$\mathbf{v}_{t} = \beta \, \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1})$$

$$\theta_t = \theta_{t-1} - \eta \, \mathbf{v}_t$$

| Symbol         | Meaning                                            | Typical value               |
| -------------- | -------------------------------------------------- | --------------------------- |
| $\mathbf{v}_t$ | Velocity — exponential moving average of gradients | Initialised to $\mathbf{0}$ |
| $\beta$        | Momentum coefficient (decay factor)                | $0.9$                       |
| $\eta$         | Learning rate                                      | Problem-dependent           |

> **Note on conventions.** The formulation above follows the "gradient accumulation" convention (used in the notebook). The alternative "velocity" convention from [Week 01](../week01_optimization/theory.md#62-the-momentum-update-rule) writes $\mathbf{v}_t = \beta \mathbf{v}_{t-1} - \eta \nabla \mathcal{L}$ and updates $\theta_t = \theta_{t-1} + \mathbf{v}_t$. Both are mathematically equivalent up to how $\eta$ is factored.

**Unrolling** shows that $\mathbf{v}_t$ is an EMA of past gradients (without the $(1 - \beta)$ normalisation factor):

$$\mathbf{v}_t = \sum_{k=1}^{t} \beta^{t-k} \nabla \mathcal{L}(\theta_{k-1})$$

### Why Momentum Works (Formal Argument)

Consider a 2D quadratic loss $\mathcal{L}(w_1, w_2) = \frac{1}{2}(\lambda_1 w_1^2 + \lambda_2 w_2^2)$ with $\lambda_1 \ll \lambda_2$.

Along the steep direction ($w_2$): the gradient alternates sign each step (zigzagging). When summed in the EMA, these oscillating components partially **cancel**, reducing the effective step size in $w_2$.

Along the shallow direction ($w_1$): the gradient consistently points toward 0. These consistent components **accumulate** in the EMA, increasing the effective step size in $w_1$.

The net effect: momentum **amplifies consistent gradient directions and dampens oscillating ones**. In the limit of large $\beta$, the effective step size in a consistent direction scales as $\eta / (1 - \beta)$. For $\beta = 0.9$: the effective step size is $10\eta$ — a 10× speedup along consistent directions.

| Direction behaviour              | GD effective step   | Momentum effective step                                         |
| -------------------------------- | ------------------- | --------------------------------------------------------------- |
| Consistent (same sign each step) | $\eta \|\nabla_j\|$ | $\frac{\eta}{1 - \beta} \|\nabla_j\|$ (amplified)               |
| Oscillating (alternating sign)   | $\eta \|\nabla_j\|$ | $\approx \frac{\eta(1-\beta)}{1+\beta} \|\nabla_j\|$ (dampened) |

> **Notebook reference.** In `starter.ipynb` Cell 5, the `MomentumOptimizer` class implements this update. Cell 7 compares Momentum, RMSProp, and Adam on the quadratic $\mathcal{L}(x,y) = x^2 + 5y^2$. The Momentum trajectory should be smoother than vanilla GD (if it were shown), curving directly toward the minimum instead of zigzagging.

---

## 5. AdaGrad: Per-Parameter Learning Rates

### Motivation

The fundamental insight of **AdaGrad** (Duchi et al., 2011) is that different parameters may need different learning rates. A parameter whose gradient is consistently large (steep direction) should have its learning rate **reduced**; a parameter whose gradient is consistently small (shallow direction) should have its learning rate **increased**.

### The Algorithm

AdaGrad accumulates the sum of squared gradients for each parameter:

$$G_t = G_{t-1} + \nabla_\theta \mathcal{L}(\theta_t) \odot \nabla_\theta \mathcal{L}(\theta_t)$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot \nabla_\theta \mathcal{L}(\theta_t)$$

| Symbol                               | Meaning                                                      |
| ------------------------------------ | ------------------------------------------------------------ |
| $G_t \in \mathbb{R}^p$               | Accumulated sum of squared gradients (per parameter)         |
| $\odot$                              | Element-wise (Hadamard) product                              |
| $\sqrt{G_t}$                         | Element-wise square root                                     |
| $\epsilon$                           | Small constant to prevent division by zero (e.g., $10^{-8}$) |
| $\eta / (\sqrt{G_{t,j}} + \epsilon)$ | Effective learning rate for parameter $j$                    |

**The key mechanism.** Each parameter $j$ has its own denominator $\sqrt{G_{t,j}}$:
- If gradients for parameter $j$ have been large → $G_{t,j}$ is large → effective LR is small.
- If gradients for parameter $j$ have been small → $G_{t,j}$ is small → effective LR is large.

This automatically adapts the step size to the geometry — no manual tuning per direction.

### The Fatal Flaw

$G_t$ is a **monotonically increasing** sum. It never decreases. Over time, $\sqrt{G_t} \to \infty$ and the effective learning rate $\eta / \sqrt{G_t} \to 0$ for every parameter. The optimiser **stops learning**.

For short training (convex problems with a fixed number of steps), this is acceptable — the decaying LR mirrors the optimal schedule. For deep learning (long training, non-convex), it is catastrophic.

> **When AdaGrad is appropriate.** Sparse problems where some parameters receive gradients infrequently (e.g., natural language processing with large vocabularies). For frequent features, the LR decays normally; for rare features, $G_{t,j}$ stays small and the LR remains high. This is AdaGrad's original motivation.

> **Notebook reference.** Exercise 5 in `starter.ipynb` (Cell 14) asks you to implement AdaGrad and compare it to RMSProp. The key observation: on a long optimisation run, AdaGrad's loss curve plateaus while RMSProp continues to converge.
>
> **Suggested experiment.** Implement AdaGrad and run it for 500 steps on the quadratic loss. Plot the effective learning rate $\eta / (\sqrt{G_{t,j}} + \epsilon)$ for each parameter over time. You will see both rates decay monotonically toward zero, confirming the fundamental issue.

---

## 6. RMSProp: Fixing AdaGrad's Decay

### Motivation

Geoffrey Hinton proposed **RMSProp** (Root Mean Square Propagation) in his 2012 Coursera lectures (never formally published) as a simple fix: replace AdaGrad's cumulative sum with an **exponential moving average** of squared gradients. This allows the denominator to forget old gradients, preventing the learning rate from decaying to zero.

### The Algorithm

$$v_t = \beta \, v_{t-1} + (1 - \beta) \, (\nabla_\theta \mathcal{L})^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot \nabla_\theta \mathcal{L}(\theta_t)$$

| Symbol                 | Meaning                                       | Typical value               |
| ---------------------- | --------------------------------------------- | --------------------------- |
| $v_t \in \mathbb{R}^p$ | EMA of squared gradients (per parameter)      | Initialised to $\mathbf{0}$ |
| $\beta$                | Decay factor for the squared gradient EMA     | $0.9$ or $0.99$             |
| $\epsilon$             | Numerical stability constant                  | $10^{-8}$                   |
| $\sqrt{v_{t,j}}$       | **RMS** of recent gradients for parameter $j$ | Adapts per parameter        |

**What $\sqrt{v_t}$ estimates.** The EMA $v_t$ approximates $\mathbb{E}[g_t^2]$ — the second moment (mean of squares) of the gradient over a recent window. Therefore $\sqrt{v_t} \approx \text{RMS}(g)$ — the root-mean-square of recent gradients. The effective learning rate is:

$$\eta_{\text{eff},j} = \frac{\eta}{\sqrt{v_{t,j}} + \epsilon} \approx \frac{\eta}{\text{RMS}(g_j)}$$

**Interpretation.** Each parameter's step is normalised by the magnitude of its recent gradients:
- Large recent gradients $\Rightarrow$ large RMS $\Rightarrow$ smaller effective LR (cautious steps).
- Small recent gradients $\Rightarrow$ small RMS $\Rightarrow$ larger effective LR (bold steps).

This achieves the same adaptive effect as AdaGrad but without the monotonic decay problem: old, large gradients are forgotten through the exponential decay.

### Connection to Natural Gradient and Diagonal Preconditioning

The RMSProp update can be written as:

$$\theta_{t+1} = \theta_t - \eta \, D_t^{-1} \nabla_\theta \mathcal{L}$$

where $D_t = \text{diag}(\sqrt{v_t} + \epsilon)$. This is a **preconditioned gradient descent** step with a diagonal approximation to the curvature matrix. In the ideal case, $D_t$ would approximate $\sqrt{\text{diag}(H)}$ (the square root of the diagonal of the Hessian), which makes the preconditioned update equivalent to Newton's method restricted to diagonal scaling.

> **Intuition.** RMSProp is like a hiker who remembers how steep the terrain has been in each direction over the last few steps. In directions where the terrain has been steep (large gradient), the hiker takes cautious steps. In directions where the terrain has been gentle (small gradient), the hiker strides confidently. Unlike AdaGrad's hiker, who remembers the steepness from the *entire* journey and becomes increasingly cautious, RMSProp's hiker only remembers the recent past.

> **Notebook reference.** In `starter.ipynb` Cell 5, the `RMSPropOptimizer` class implements this algorithm. In Cell 7, compare RMSProp's trajectory to Momentum's: RMSProp should navigate the elongated contours more directly because it automatically scales each parameter's step.
>
> **Suggested experiment.** Change the loss to a more ill-conditioned quadratic: $\mathcal{L}(x,y) = x^2 + 100y^2$ ($\kappa = 100$). Run both Momentum and RMSProp with the same initial $\eta$. Momentum will struggle (zigzag along $y$ or require tiny $\eta$), while RMSProp should converge smoothly because it automatically shrinks the $y$-step and amplifies the $x$-step.

---

## 7. Adam: Combining Momentum and Adaptivity

### 7.1 The Adam Algorithm

**Adam** (Adaptive Moment Estimation; Kingma & Ba, 2015) combines the two ideas:
- **First moment** EMA (momentum): smooth, directional gradient estimate.
- **Second moment** EMA (RMSProp-like): adaptive per-parameter scaling.

The full algorithm:

**Initialise:** $m_0 = \mathbf{0}$, $v_0 = \mathbf{0}$, $t = 0$

**For each step:**

1. $t \leftarrow t + 1$
2. $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ — compute gradient
3. $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ — update first moment (EMA of gradients)
4. $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ — update second moment (EMA of squared gradients)
5. $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ — bias-corrected first moment
6. $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ — bias-corrected second moment
7. $\theta_t = \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ — parameter update

| Symbol                 | Meaning                                                          | Default value |
| ---------------------- | ---------------------------------------------------------------- | ------------- |
| $m_t \in \mathbb{R}^p$ | First moment estimate (EMA of gradients) — like momentum         | —             |
| $v_t \in \mathbb{R}^p$ | Second moment estimate (EMA of squared gradients) — like RMSProp | —             |
| $\hat{m}_t$            | Bias-corrected first moment                                      | —             |
| $\hat{v}_t$            | Bias-corrected second moment                                     | —             |
| $\beta_1$              | Decay rate for first moment                                      | $0.9$         |
| $\beta_2$              | Decay rate for second moment                                     | $0.999$       |
| $\eta$                 | Base learning rate                                               | $0.001$       |
| $\epsilon$             | Numerical stability constant                                     | $10^{-8}$     |
| $g_t^2$                | Element-wise squared gradient: $g_t \odot g_t$                   | —             |

**The genealogy of Adam:**

$$\text{Adam} = \underbrace{\text{Momentum}}_{\text{first moment}} + \underbrace{\text{RMSProp}}_{\text{second moment}} + \underbrace{\text{Bias correction}}_{\text{initialisation fix}}$$

> **Notebook reference.** The `AdamOptimizer` class in `starter.ipynb` Cell 5 implements exactly this algorithm. Study the code and match each line to the numbered steps above.

---

### 7.2 Bias Correction

Since $m_0 = \mathbf{0}$ and $v_0 = \mathbf{0}$, both moment estimates are biased toward zero during the first few steps.

**First moment bias.** At step $t$:

$$\mathbb{E}[m_t] = (1 - \beta_1^t)\,\mathbb{E}[g_t]$$

(This follows from unrolling the EMA with geometric series, assuming the gradient distribution is stationary.)

Therefore:

$$\mathbb{E}\left[\frac{m_t}{1 - \beta_1^t}\right] = \mathbb{E}[g_t]$$

The denominator $1 - \beta_1^t$ corrects the bias. At $t = 1$: $m_1 = (1 - \beta_1)g_1$, and $\hat{m}_1 = m_1 / (1 - \beta_1) = g_1$. ✓

**Second moment bias.** Analogously:

$$\mathbb{E}[\hat{v}_t] = \frac{v_t}{1 - \beta_2^t} \approx \mathbb{E}[g_t^2]$$

**Practical impact.** For $\beta_2 = 0.999$, the bias correction factor $1/(1 - 0.999^t)$ is very large at small $t$: at $t = 1$, the correction multiplies by 1000. Without bias correction, the second moment underestimates $\mathbb{E}[g^2]$ by a factor of 1000, causing the effective step to be $\sim \sqrt{1000} \approx 32\times$ too large. This can cause divergence in the first few steps.

> **Suggested experiment.** Implement Adam with and without bias correction. Start from a point where the gradient is non-trivial (e.g., $\theta_0 = (3, -2)$ on the quadratic). Plot the loss for the first 50 steps. The uncorrected version should overshoot or diverge in the first few steps, while the corrected version should be smooth.

---

### 7.3 Effective Step Size Analysis

The Adam update for parameter $j$ at step $t$ (ignoring bias correction for simplicity) is:

$$\Delta \theta_j = -\eta \frac{\hat{m}_{t,j}}{\sqrt{\hat{v}_{t,j}} + \epsilon}$$

**Case 1: Deterministic gradients (full batch).** If the gradient for parameter $j$ is constant at $g_j$ every step:
- $\hat{m}_{t,j} \to g_j$
- $\hat{v}_{t,j} \to g_j^2$
- $\Delta \theta_j \to -\eta \frac{g_j}{\sqrt{g_j^2} + \epsilon} = -\eta \frac{g_j}{|g_j| + \epsilon} \approx -\eta \cdot \text{sign}(g_j)$

The step size is approximately **constant** ($\eta$), regardless of the gradient magnitude! Adam effectively becomes a **sign-based** optimser: each parameter moves by $\pm \eta$ per step.

> **Insight.** This explains why Adam's $\eta$ has a very different meaning from SGD's $\eta$. In SGD, $\eta$ scales the gradient magnitude. In Adam, $\eta$ is approximately the step size itself. This is why Adam's default $\eta = 0.001$ is typically good — it means each parameter moves by at most $\sim 0.001$ per step.

**Case 2: Noisy gradients (SGD).** If the gradient for parameter $j$ has mean $\mu_j$ and variance $\sigma_j^2$:
- $\hat{m}_{t,j} \approx \mu_j$
- $\hat{v}_{t,j} \approx \mu_j^2 + \sigma_j^2$

The effective step:

$$|\Delta \theta_j| \approx \eta \frac{|\mu_j|}{\sqrt{\mu_j^2 + \sigma_j^2} + \epsilon}$$

The denominator includes the noise variance, so noisy gradients produce smaller steps — a natural **signal-to-noise ratio** adaptation.

| Gradient condition        | Effective step | Behaviour                             |
| ------------------------- | -------------- | ------------------------------------- |
| Large $                   | \mu_j          | $, small $\sigma_j$ (strong signal)   | $\approx \eta$ | Full step |
| Small $                   | \mu_j          | $, large $\sigma_j$ (noise-dominated) | $\approx \eta  | \mu_j     | /\sigma_j$ (reduced) | Cautious |
| Zero $\mu_j$ (pure noise) | $\approx 0$    | No systematic progress                |

---

### 7.4 Adam's Hyperparameters

| Hyperparameter | Default   | Role                                                                 | Sensitivity                                                                                 |
| -------------- | --------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| $\eta$         | $0.001$   | Base learning rate / approximate step size                           | **Most sensitive.** Start with $10^{-3}$; try $\{3\times10^{-4}, 10^{-3}, 3\times10^{-3}\}$ |
| $\beta_1$      | $0.9$     | First moment decay (momentum) — how many past gradients are averaged | Moderate. $0.9$ works for most problems. Decrease for very noisy gradients.                 |
| $\beta_2$      | $0.999$   | Second moment decay — how many past squared gradients are averaged   | Low. $0.999$ works almost universally. Decrease for rapidly changing curvature.             |
| $\epsilon$     | $10^{-8}$ | Numerical stability                                                  | Very low. Rarely needs changing.                                                            |

**Practical rules of thumb:**
1. Start with the defaults: $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
2. If training is unstable → reduce $\eta$.
3. If convergence is slow → increase $\eta$ or try a warmup schedule.
4. Almost never change $\beta_1$, $\beta_2$, or $\epsilon$ — the defaults are robust.

> **Notebook reference.** Exercise 2 in `starter.ipynb` (Cell 12) asks you to grid-search over $\beta_1 \in [0.5, 0.99]$ and $\beta_2 \in [0.9, 0.9999]$. The heatmap should show that Adam is robust across a wide range, with degradation only at extreme values (e.g., $\beta_1 = 0.5$ reduces momentum too much; $\beta_2 = 0.9$ forgets curvature too quickly).

---

## 8. Variants of Adam

### 8.1 AdamW: Decoupled Weight Decay

**The problem with L2 regularisation in Adam.** The standard approach adds a weight decay term to the loss:

$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

The gradient becomes $g_t + \lambda \theta_t$. In SGD, this is equivalent to the weight decay update $\theta \leftarrow (1 - \eta\lambda)\theta - \eta g_t$. But in Adam, the adaptive denominator $\sqrt{v_t}$ scales the regularisation term alongside the gradient, altering its effect. Large parameters do not get penalised more than small ones, which defeats the purpose of weight decay.

**AdamW** (Loshchilov & Hutter, 2019) decouples the weight decay from the adaptive update:

$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \, \theta_{t-1}\right)$$

The weight decay term $\lambda \theta_{t-1}$ is applied directly to the parameters, **not** passed through the adaptive scaling. This ensures that weight decay behaves consistently regardless of the gradient history.

| Variant           | Weight decay behaviour                        | Recommended for                   |
| ----------------- | --------------------------------------------- | --------------------------------- |
| Adam + L2 loss    | Decay scaled by $1/\sqrt{v_t}$ — inconsistent | Generally avoided                 |
| AdamW (decoupled) | Decay applied uniformly                       | Standard choice for deep learning |

> **Practical impact.** AdamW is the default optimiser in PyTorch (`torch.optim.AdamW`) and in the Hugging Face Transformers library. For most modern deep learning experiments, use AdamW rather than Adam with L2 regularisation.

> **When you reach [Week 06](../week06_regularization/theory.md#3-ridge-regression-l2-regularisation) (Regularisation):** the distinction between L2 regularisation and weight decay becomes precise. For now, the key takeaway is that **AdamW is preferred over Adam when using weight decay**, and this is what PyTorch's default implementations assume.

---

### 8.2 AMSGrad

Reddi et al. (2018) showed that Adam can **fail to converge** in certain adversarial settings because the second moment EMA can decrease, allowing the effective learning rate to increase at the worst possible time.

**AMSGrad** fixes this by maintaining a running maximum of $v_t$:

$$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$

This ensures the effective learning rate never increases — it can only stay the same or decrease. The fix provides convergence guarantees in the adversarial cases where Adam fails.

> **In practice.** The adversarial cases are rare, and AMSGrad often performs similarly to (or slightly worse than) Adam on real problems. It is mentioned for theoretical completeness.

---

### 8.3 RAdam: Rectified Adam

Liu et al. (2020) observed that Adam's bias correction is not sufficient early in training when the second-moment estimate $v_t$ has high variance (because very few gradient samples have been observed). This early instability is the reason warmup is needed (Section 9.4).

**RAdam** dynamically computes the variance of $v_t$ and switches between:
- **SGD with momentum** (when $v_t$'s variance is high — early steps)
- **Full Adam** (when $v_t$'s variance is low — after enough steps)

This eliminates the need for a manually tuned warmup schedule.

> **Practical impact.** RAdam is a reasonable "set-and-forget" alternative to Adam + warmup. It requires no warmup hyperparameter. In PyTorch: `torch.optim.RAdam`.

---

## 9. Learning Rate Schedules

### 9.1 Why Schedules Matter

Even with adaptive optimisers, the learning rate $\eta$ plays a critical role:
- **Too large:** Adam's $\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ ratio is $\mathcal{O}(1)$, so the step is $\mathcal{O}(\eta)$. A large $\eta$ causes oscillation near the minimum.
- **Too small early:** slow initial progress; may get stuck in a poor basin.
- **Too large late:** prevents fine convergence near the optimum.

A **schedule** $\eta(t)$ varies the learning rate over training to address different needs at different stages.

---

### 9.2 Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

| Symbol   | Meaning                     | Typical value         |
| -------- | --------------------------- | --------------------- |
| $\eta_0$ | Initial learning rate       | Problem-dependent     |
| $\gamma$ | Multiplicative decay factor | $0.1$ or $0.5$        |
| $s$      | Steps between decays        | E.g., every 30 epochs |

The learning rate drops by a constant factor every $s$ steps. This produces a staircase pattern.

**Character.** Simple, interpretable, manually tuned. Commonly used in vision tasks (e.g., multiply by 0.1 at epochs 30 and 60 of a 90-epoch run).

> **Notebook reference.** The `step_decay` function in `starter.ipynb` Cell 9 implements this schedule with `drop_rate=0.5` and `epochs_drop=10`.

---

### 9.3 Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

| Symbol        | Meaning                                 |
| ------------- | --------------------------------------- |
| $\eta_0$      | Initial (maximum) learning rate         |
| $\eta_{\min}$ | Final (minimum) learning rate (often 0) |
| $T$           | Total number of training steps          |

The learning rate follows a half-cosine curve: it starts at $\eta_0$, slowly decreases, then drops more rapidly toward $\eta_{\min}$.

**Why cosine rather than linear?** The cosine schedule spends more time at intermediate learning rates (the derivative is small near $t = 0$ and $t = T$) and decays most rapidly in the middle. Empirically, this produces better final performance than linear decay.

**Cosine annealing with warm restarts (SGDR).** Loshchilov & Hutter (2017) proposed periodically resetting $\eta$ back to $\eta_0$, creating multiple cosine cycles. The re-warming helps escape local minima and explore different basins.

> **Notebook reference.** The `cosine_annealing` function in Cell 9 plots this schedule. Observe the smooth S-shaped decay.

---

### 9.4 Warmup

$$\eta_t = \eta_0 \cdot \frac{t}{T_w}, \qquad t = 1, 2, \ldots, T_w$$

For the first $T_w$ steps, the learning rate increases linearly from $\sim 0$ to $\eta_0$.

| Symbol | Meaning         | Typical value                 |
| ------ | --------------- | ----------------------------- |
| $T_w$  | Warmup duration | 1–10% of total training steps |

**Why warmup is needed.** At initialisation ($t = 0$):
1. The parameters are random → the gradients point in essentially random directions.
2. Adam's second moment $v_t$ is near zero → the effective step $\hat{m}_t / \sqrt{\hat{v}_t}$ can be enormous.
3. The loss surface geometry near a random initialisation is poorly modelled by the local gradient.

A large $\eta$ at this point can send the parameters far from the initialisation region, possibly into a region of the loss surface that is numerically unstable (e.g., saturated activations). Warmup gives the moment estimates time to calibrate before taking large steps.

> **Formal justification.** During warmup, the second moment $v_t$ accumulates enough gradient samples to reliably estimate $\mathbb{E}[g^2]$. After $\sim 1/(1 - \beta_2)$ steps (e.g., 1000 steps for $\beta_2 = 0.999$), the bias correction factor is near 1 and the estimate is stable. This is the theoretical minimum warmup duration.
>
> RAdam (Section 8.3) automates this reasoning, effectively doing warmup without a fixed $T_w$.

---

### 9.5 Warmup + Cosine Annealing

The "standard" modern schedule combines linear warmup with cosine decay:

$$\eta_t = \begin{cases} \eta_0 \cdot \frac{t}{T_w} & t \leq T_w \qquad \text{(warmup phase)} \\ \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi(t - T_w)}{T - T_w}\right)\right) & t > T_w \qquad \text{(cosine phase)} \end{cases}$$

This schedule is used by default in most transformer training recipes ([Week 18](../../06_sequence_models/week18_transformers/theory.md#91-learning-rate-warm-up)) and large-scale deep learning.

> **Notebook reference.** The `warmup_cosine` function in Cell 9 implements this composite schedule. The plot shows the characteristic ramp followed by a smooth decay.
>
> **Suggested experiment.** Run Adam with (a) constant LR, (b) cosine only, and (c) warmup + cosine on the same problem. Compare the final loss after 100 steps. Particularly for a non-convex loss (e.g., the multimodal function from [Week 01](../week01_optimization/theory.md#72-non-convex-landscapes-local-minima-and-saddle-points)), the warmup + cosine variant should reach a lower final loss because it explores more broadly early and settles more precisely late.

---

### 9.6 The LR Range Test

The LR range test (Smith, 2017) provides an empirical method to find a good $\eta_0$:

1. Start with $\eta = \eta_{\text{low}}$ (e.g., $10^{-7}$).
2. After each mini-batch, multiply $\eta$ by a constant factor so that $\eta$ increases exponentially over one epoch.
3. Record the loss at each step.
4. Plot loss vs. $\log(\eta)$.

**Reading the plot:**
- **Stable / decreasing loss:** the LR is in the useful range.
- **Loss starts increasing:** the LR has crossed the instability threshold.
- **Best $\eta_0$:** approximately 1/10th of the LR where the loss is minimised (i.e., well within the stable regime).

$$\eta_{\text{schedule}} = \eta_{\text{low}} \cdot \left(\frac{\eta_{\text{high}}}{\eta_{\text{low}}}\right)^{t/T}$$

where $T$ is the total number of steps in the test.

> **Notebook reference.** Exercise 1 in `starter.ipynb` (Cell 11) asks you to implement this. Use Adam as the optimiser.

---

## 10. Choosing an Optimiser

### Decision Flowchart

| Situation                             | Recommended optimiser             | Rationale                                                          |
| ------------------------------------- | --------------------------------- | ------------------------------------------------------------------ |
| **Default deep learning**             | AdamW                             | Robust default; handles diverse curvature well                     |
| **Transformer training**              | AdamW + warmup + cosine           | Industry standard (GPT, BERT, ViT)                                 |
| **Computer vision (CNNs)**            | SGD + momentum + step decay       | Often outperforms Adam on vision tasks (Loshchilov & Hutter, 2019) |
| **Limited hyperparameter tuning**     | Adam ($\eta = 10^{-3}$, defaults) | Least sensitive to hyperparameters                                 |
| **Sparse gradients (NLP embeddings)** | Adam or AdaGrad                   | Adaptive LR crucial for rare features                              |
| **Need best possible generalisation** | SGD + momentum + careful schedule | SGD finds wider minima (but requires more tuning)                  |

### SGD vs. Adam: The Generalisation Debate

A persistent finding in the literature (Wilson et al., 2017) is that **SGD + momentum often generalises better than Adam** on certain vision benchmarks, even though Adam converges faster. Proposed explanations:

1. **SGD's noise is implicitly regularising.** The gradient noise in SGD biases the optimiser toward flat minima (Keskar et al., 2017). Adam's adaptive rates reduce this effect.
2. **Adam's effective step size is nearly constant** (Section 7.3). This can cause it to overshoot or oscillate in the final fine-convergence phase.
3. **Weight decay behaves differently.** In Adam (but not AdamW), L2 regularisation is distorted by the adaptive scaling (Section 8.1).

> **Practical advice.** Use **AdamW** as the default. If validation accuracy plateaus and you have the compute budget for hyperparameter search, try **SGD + momentum + cosine annealing** and compare.

### The Unified View

All the optimisers in this week can be expressed as:

$$\theta_{t+1} = \theta_t - \eta_t \, P_t^{-1} \, \hat{g}_t$$

| Component                       | SGD   | Momentum           | RMSProp                              | Adam                                       |
| ------------------------------- | ----- | ------------------ | ------------------------------------ | ------------------------------------------ |
| $\hat{g}_t$ (gradient estimate) | $g_t$ | $m_t$ (EMA of $g$) | $g_t$                                | $\hat{m}_t$ (bias-corrected EMA of $g$)    |
| $P_t$ (preconditioning)         | $I$   | $I$                | $\text{diag}(\sqrt{v_t} + \epsilon)$ | $\text{diag}(\sqrt{\hat{v}_t} + \epsilon)$ |

> **Interpretation.** SGD and momentum use a "direction" strategy (raw gradient or smoothed gradient) but treat all parameters equally. RMSProp and Adam add a "scaling" strategy that adjusts each parameter independently. Adam combines both.

---

## 11. Convergence Guarantees and Limitations

### Convex Case

For convex, $L$-smooth functions:

| Algorithm            | Convergence rate                                              | Note                                    |
| -------------------- | ------------------------------------------------------------- | --------------------------------------- |
| GD                   | $\mathcal{O}(1/T)$                                            | Sublinear                               |
| GD (strongly convex) | $\mathcal{O}(\rho^T)$, $\rho = \frac{\kappa - 1}{\kappa + 1}$ | Linear (exponential)                    |
| Nesterov momentum    | $\mathcal{O}(1/T^2)$                                          | Optimal for first-order methods         |
| Adam                 | $\mathcal{O}(1/\sqrt{T})$                                     | Proven by Kingma & Ba (2015) for convex |
| SGD                  | $\mathcal{O}(1/\sqrt{T})$                                     | With $\eta_t \propto 1/\sqrt{t}$        |

> **Adam is slower in theory?** Yes, for convex problems, Adam's $\mathcal{O}(1/\sqrt{T})$ is worse than GD's $\mathcal{O}(1/T)$. Adam's strength is **per-parameter adaptation**, which pays off in practice for high-dimensional, ill-conditioned, non-convex problems where the theoretical rates are not tight.

### Non-Convex Case (Deep Learning)

For non-convex problems, the goal is weaker: find a point $\theta$ where $\|\nabla \mathcal{L}(\theta)\| \leq \epsilon$ (an approximate stationary point — which could be a minimum, saddle, or anything with small gradient).

| Algorithm | Steps to reach $\|\nabla \mathcal{L}\| \leq \epsilon$                                |
| --------- | ------------------------------------------------------------------------------------ |
| GD        | $\mathcal{O}(1/\epsilon^2)$                                                          |
| SGD       | $\mathcal{O}(1/\epsilon^4)$ (or $\mathcal{O}(1/\epsilon^2)$ with variance reduction) |
| Adam      | $\mathcal{O}(1/\epsilon^2)$ (under bounded gradient assumptions)                     |

These are worst-case rates. In practice, the structure of deep learning loss surfaces (which are not adversarial) means all algorithms converge much faster.

### Known Failure Cases of Adam

1. **Non-convergence (Reddi et al., 2018).** In specific adversarial constructions, Adam can oscillate indefinitely. AMSGrad (Section 8.2) fixes this.
2. **Worse generalisation than SGD on some vision tasks** (Wilson et al., 2017). Mitigated by AdamW and careful scheduling.
3. **Sensitivity to $\epsilon$ in half-precision training.** When using FP16, $\epsilon = 10^{-8}$ can be below the precision floor. Use $\epsilon = 10^{-4}$ or higher. This becomes relevant in [Week 14](../../05_deep_learning/week14_training_at_scale/theory.md#6-mixed-precision-training) (Training at Scale).

---

## 12. Notebook Reference Guide

| Cell                      | Section                                      | What to observe                                                                          | Theory reference             |
| ------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------- |
| 3 (Optimiser classes)     | Momentum, RMSProp, Adam implementations      | Match code to math: `self.m = beta1 * self.m + (1-beta1) * grads` ↔ Step 3 of Adam       | Sections 4, 6, 7             |
| 5 (Trajectory comparison) | Three optimisers on $x^2 + 5y^2$             | Momentum overshoots then settles; RMSProp navigates directly; Adam combines both         | Section 7.3                  |
| 5 (Loss curves)           | Log-scale loss vs. step                      | Adam typically converges fastest; note the early-step behaviour (bias correction effect) | Section 7.2                  |
| 7 (LR schedules)          | Four schedules plotted                       | Step decay staircase; cosine smooth curve; warmup ramp; warmup+cosine composite          | Section 9                    |
| Ex.1 (LR range test)      | Loss vs. log(LR)                             | Find the "cliff" — the LR where training becomes unstable                                | Section 9.6                  |
| Ex.2 (Beta grid search)   | Heatmap of final loss vs. $\beta_1, \beta_2$ | Robustness plateau with degradation at extremes                                          | Section 7.4                  |
| Ex.3 (Rosenbrock)         | $f(x,y) = (1-x)^2 + 100(y-x^2)^2$            | A narrow curved valley — all optimisers struggle; LR tuning is critical                  | Section 2 (ill-conditioning) |
| Ex.5 (AdaGrad)            | AdaGrad vs. RMSProp over 500+ steps          | AdaGrad's LR decays to ~0; RMSProp's LR stabilises                                       | Sections 5, 6                |

**Suggested modifications across exercises:**

| Modification                                                | What it reveals                                                          |
| ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| Change $\mathcal{L}(x,y) = x^2 + 5y^2$ to $x^2 + 500y^2$    | Adam's advantage over Momentum increases with $\kappa$                   |
| Remove bias correction from Adam (`m_hat = m`, `v_hat = v`) | Early-step instability — overshooting or divergence                      |
| Set $\beta_2 = 0.9$ (instead of 0.999)                      | Second moment forgets too fast → noisy effective LR                      |
| Set $\epsilon = 0.1$ (instead of $10^{-8}$)                 | Cuts the adaptive effect — Adam behaves like SGD + momentum              |
| Run Adam + constant LR vs. Adam + cosine on Rosenbrock      | Schedule can make the difference between finding and missing the minimum |

---

## 13. Symbol Reference

| Symbol                | Name                          | Meaning                                              |
| --------------------- | ----------------------------- | ---------------------------------------------------- |
| $\theta_t$            | Parameters at step $t$        | All learnable weights and biases                     |
| $g_t$                 | Gradient                      | $\nabla_\theta \mathcal{L}(\theta_{t-1})$            |
| $g_t^2$               | Squared gradient              | Element-wise: $g_t \odot g_t$                        |
| $m_t$                 | First moment                  | EMA of gradients (momentum direction)                |
| $v_t$                 | Second moment                 | EMA of squared gradients (curvature estimate)        |
| $\hat{m}_t$           | Bias-corrected first moment   | $m_t / (1 - \beta_1^t)$                              |
| $\hat{v}_t$           | Bias-corrected second moment  | $v_t / (1 - \beta_2^t)$                              |
| $G_t$                 | Accumulated squared gradients | AdaGrad's accumulator (monotonically increasing)     |
| $\eta$                | Learning rate                 | Base step size                                       |
| $\eta_{\text{eff},j}$ | Effective learning rate       | $\eta / (\sqrt{v_{t,j}} + \epsilon)$ — per-parameter |
| $\beta_1$             | First moment decay            | Momentum coefficient; default $0.9$                  |
| $\beta_2$             | Second moment decay           | Curvature averaging coefficient; default $0.999$     |
| $\beta$               | Generic decay                 | Used for Momentum ($\beta$) and RMSProp ($\beta$)    |
| $\epsilon$            | Stability constant            | Prevents division by zero; default $10^{-8}$         |
| $\lambda$             | Weight decay coefficient      | Strength of parameter regularisation                 |
| $\gamma$              | Step decay factor             | Multiplicative LR reduction (e.g., $0.1$)            |
| $s$                   | Step decay period             | Steps between LR reductions                          |
| $T$                   | Total training steps          | Budget for the optimisation process                  |
| $T_w$                 | Warmup duration               | Steps of linear LR ramp                              |
| $\kappa$              | Condition number              | $\lambda_{\max}/\lambda_{\min}$ of the Hessian       |
| $\rho$                | Convergence factor            | Per-step contraction ratio                           |
| $P_t$                 | Preconditioner                | Diagonal matrix that scales the gradient             |
| $D_t$                 | Diagonal scaling              | $\text{diag}(\sqrt{v_t} + \epsilon)$                 |
| $\odot$               | Hadamard product              | Element-wise multiplication                          |
| $\text{sign}(\cdot)$  | Sign function                 | $+1$ if positive, $-1$ if negative, $0$ if zero      |
| $\text{RMS}(\cdot)$   | Root mean square              | $\sqrt{\mathbb{E}[g^2]}$                             |

---

## 14. References

1. Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980). — The original Adam paper.
2. Duchi, J., Hazan, E., & Singer, Y. (2011). "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization." *JMLR*, 12, 2121–2159. — AdaGrad.
3. Hinton, G. (2012). "Neural networks for machine learning." Coursera Lecture 6e. — RMSProp (unpublished; first described in lecture slides).
4. Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101). — AdamW.
5. Reddi, S. J., Kale, S., & Kumar, S. (2018). "On the Convergence of Adam and Beyond." *ICLR*. — AMSGrad; proves Adam can fail to converge in adversarial settings.
6. Liu, L., et al. (2020). "On the Variance of the Adaptive Learning Rate and Beyond." *ICLR*. [arXiv:1908.03265](https://arxiv.org/abs/1908.03265). — RAdam.
7. Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*. — Cosine annealing with warm restarts.
8. Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks." *IEEE WACV*. — LR range test.
9. Smith, L. N. (2018). "A Disciplined Approach to Neural Network Hyper-Parameters." *arXiv:1803.09820*. — 1cycle policy and practical hyperparameter tuning.
10. Wilson, A. C., et al. (2017). "The Marginal Value of Adaptive Gradient Methods in Machine Learning." *NeurIPS*. — SGD generalises better than Adam on some vision benchmarks.
11. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." *arXiv:1706.02677*. — Linear scaling rule.
12. Keskar, N. S., et al. (2017). "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." *ICLR*. — Flat vs. sharp minima.
13. Ruder, S. (2016). "An overview of gradient descent optimization algorithms." *arXiv:1609.04747*. — Excellent survey.
14. Polyak, B. T. (1964). "Some methods of speeding up the convergence of iteration methods." *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1–17. — Momentum.
15. Nesterov, Y. (1983). "A method for solving a convex programming problem with convergence rate $O(1/k^2)$." *Soviet Mathematics Doklady*, 27, 372–376. — Nesterov accelerated gradient.
