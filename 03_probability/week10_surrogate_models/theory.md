# Surrogate Models & Gaussian Processes

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Surrogate Modelling](#2-surrogate-modelling)
   - 2.1 [The Problem Setting](#21-the-problem-setting)
   - 2.2 [Requirements for a Good Surrogate](#22-requirements-for-a-good-surrogate)
3. [Gaussian Processes — Intuition](#3-gaussian-processes--intuition)
   - 3.1 [From Bayesian Linear Regression to GPs](#31-from-bayesian-linear-regression-to-gps)
   - 3.2 [The Weight-Space View](#32-the-weight-space-view)
   - 3.3 [The Function-Space View](#33-the-function-space-view)
4. [GP Prior](#4-gp-prior)
   - 4.1 [Mean and Covariance Functions](#41-mean-and-covariance-functions)
   - 4.2 [Sampling from a GP Prior](#42-sampling-from-a-gp-prior)
5. [GP Posterior (Regression)](#5-gp-posterior-regression)
   - 5.1 [The Conditioning Formula](#51-the-conditioning-formula)
   - 5.2 [Derivation](#52-derivation)
   - 5.3 [Cholesky Implementation](#53-cholesky-implementation)
   - 5.4 [Noise-Free vs. Noisy Observations](#54-noise-free-vs-noisy-observations)
   - 5.5 [Computational Cost](#55-computational-cost)
6. [Kernel Functions](#6-kernel-functions)
   - 6.1 [What Makes a Valid Kernel](#61-what-makes-a-valid-kernel)
   - 6.2 [RBF (Squared Exponential)](#62-rbf-squared-exponential)
   - 6.3 [Matérn Family](#63-matrn-family)
   - 6.4 [Periodic Kernel](#64-periodic-kernel)
   - 6.5 [Linear Kernel](#65-linear-kernel)
   - 6.6 [Composing Kernels](#66-composing-kernels)
   - 6.7 [The Length-Scale Parameter](#67-the-length-scale-parameter)
7. [Hyperparameter Optimisation via Marginal Likelihood](#7-hyperparameter-optimisation-via-marginal-likelihood)
   - 7.1 [The Log-Marginal-Likelihood](#71-the-log-marginal-likelihood)
   - 7.2 [Automatic Relevance Determination (ARD)](#72-automatic-relevance-determination-ard)
8. [Calibration and Coverage](#8-calibration-and-coverage)
9. [Bayesian Optimisation](#9-bayesian-optimisation)
   - 9.1 [The Optimisation Loop](#91-the-optimisation-loop)
   - 9.2 [Expected Improvement (EI)](#92-expected-improvement-ei)
   - 9.3 [Upper Confidence Bound (UCB)](#93-upper-confidence-bound-ucb)
   - 9.4 [Exploration–Exploitation Trade-Off](#94-explorationexploitation-trade-off)
   - 9.5 [Practical Considerations](#95-practical-considerations)
10. [Connections to the Rest of the Course](#10-connections-to-the-rest-of-the-course)
11. [Notebook Reference Guide](#11-notebook-reference-guide)
12. [Symbol Reference](#12-symbol-reference)
13. [References](#13-references)

---

## 1. Scope and Purpose

[[Weeks 07](../week07_likelihood/theory.md)](../week07_likelihood/theory.md)–[09](../week09_time_series/theory.md) developed the probabilistic and statistical toolkit — likelihood, uncertainty quantification, time series. This week brings it all together in **Gaussian Process (GP) regression**, the method that simultaneously:

- Gives a **prediction** at every point (like regression).
- Gives a **calibrated uncertainty estimate** (like Bayesian inference).
- Works in a **non-parametric** setting (the model complexity grows with the data).

GPs are the foundation for **surrogate-based optimisation** (also called Bayesian optimisation, or BO), which is the dominant approach for tuning expensive functions — hyperparameter search for deep networks, physical experiments, simulations.

**Goals for this week:**
1. Understand the GP as a distribution over functions, defined by a mean and kernel.
2. Derive and implement the GP posterior (conditioning on observed data).
3. Explore how the choice of kernel controls the properties of the modelled function.
4. Implement Bayesian optimisation using the Expected Improvement acquisition function.

**Prerequisites.** [Week 07](../week07_likelihood/theory.md) (likelihood, MLE — used for kernel hyperparameter optimisation), [Week 08](../week08_uncertainty/theory.md) (Bayesian linear regression — the GP is its infinite-dimensional extension), [Week 03](../../02_fundamentals/week03_linear_models/theory.md) (linear regression, polynomial features — the weight-space to function-space transition).

---

## 2. Surrogate Modelling

### 2.1 The Problem Setting

A **black-box function** $f: \mathcal{X} \to \mathbb{R}$ is:
- **Expensive** to evaluate (e.g., a 3-hour CFD simulation, a week-long lab experiment, or training a neural network to convergence).
- **Has no known closed form** — we can query $f(\mathbf{x})$ but cannot compute $\nabla f$ or exploit structural properties.
- Possibly **noisy**: we observe $y = f(\mathbf{x}) + \epsilon$.

A **surrogate model** $\hat{f}$ is a cheap-to-evaluate approximation of $f$ built from a limited budget of evaluations $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$.

| Application           | Expensive function $f$              | Input $\mathbf{x}$                      |
| --------------------- | ----------------------------------- | --------------------------------------- |
| Hyperparameter tuning | Validation loss of a neural network | Learning rate, batch size, architecture |
| Engineering design    | Structural simulation (FEM)         | Material properties, geometry           |
| Drug discovery        | Molecular binding affinity          | Molecular descriptors                   |
| Environmental science | Climate model output                | Emission scenarios, forcings            |

---

### 2.2 Requirements for a Good Surrogate

1. **Accuracy** — $\hat{f}(\mathbf{x}) \approx f(\mathbf{x})$ where data exists.
2. **Uncertainty quantification** — the surrogate should know what it doesn't know (high uncertainty far from data).
3. **Data efficiency** — must work well from a small number of evaluations (tens, not thousands).
4. **Smoothness control** — the surrogate's assumptions about smoothness should match the true function.

GPs satisfy all four by design.

---

## 3. Gaussian Processes — Intuition

### 3.1 From Bayesian Linear Regression to GPs

Recall BLR ([Week 08](../week08_uncertainty/theory.md)):

$$y = \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}) + \epsilon, \qquad \mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1}I)$$

where $\boldsymbol{\phi}(\mathbf{x})$ is a feature map (e.g., polynomial features). The prior on $\mathbf{w}$ induces a prior on functions:

$$f(\mathbf{x}) = \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x})$$

**Key question:** what happens as the number of basis functions goes to infinity ($\dim(\boldsymbol{\phi}) \to \infty$)?

The prior on $f$ becomes a **Gaussian process**: instead of a finite-dimensional Gaussian over weights, we get an infinite-dimensional Gaussian over functions. The kernel arises naturally:

$$k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^\top(\alpha^{-1}I)\boldsymbol{\phi}(\mathbf{x}') = \frac{1}{\alpha}\boldsymbol{\phi}(\mathbf{x})^\top\boldsymbol{\phi}(\mathbf{x}')$$

---

### 3.2 The Weight-Space View

In the **weight-space view**, the GP is BLR with (possibly infinite-dimensional) features:

$$f(\mathbf{x}) = \sum_{j=1}^{J}w_j\phi_j(\mathbf{x}), \qquad \mathbf{w} \sim \mathcal{N}(\mathbf{0}, \Sigma_w)$$

The covariance between function values at two inputs is:

$$\text{Cov}[f(\mathbf{x}), f(\mathbf{x}')] = \boldsymbol{\phi}(\mathbf{x})^\top\Sigma_w\boldsymbol{\phi}(\mathbf{x}') = k(\mathbf{x}, \mathbf{x}')$$

This is the **kernel trick**: we never need to explicitly compute the (possibly infinite-dimensional) features — only their inner product (the kernel).

---

### 3.3 The Function-Space View

In the **function-space view**, we define the GP directly:

> **Definition.** A **Gaussian process** is a collection of random variables, any finite number of which have a joint Gaussian distribution.

A GP is fully specified by:
- A **mean function** $m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$
- A **covariance (kernel) function** $k(\mathbf{x}, \mathbf{x}') = \text{Cov}[f(\mathbf{x}), f(\mathbf{x}')]$

$$f \sim \mathcal{GP}\!\left(m(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}')\right)$$

For any finite set of inputs $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$, the function values are jointly Gaussian:

$$\mathbf{f} = [f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)]^\top \sim \mathcal{N}(\mathbf{m}, K)$$

where $\mathbf{m}_i = m(\mathbf{x}_i)$ and $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$.

> **The GP is a "distribution over functions."** Each sample from the GP is an entire function $f: \mathcal{X} \to \mathbb{R}$. The kernel controls the properties of these sampled functions — their smoothness, periodicity, amplitude, length scale.

---

## 4. GP Prior

### 4.1 Mean and Covariance Functions

**Mean function.** By convention, $m(\mathbf{x}) = 0$. This is not restrictive because:
1. The data can be centred (subtract the sample mean of $\mathbf{y}$).
2. The posterior mean will be non-zero after conditioning on data.
3. `sklearn`'s `normalize_y=True` handles this automatically.

**Covariance function (kernel).** The kernel encodes all prior assumptions about $f$:

| Kernel property                                                          | Encodes                                                      |
| ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| $k(\mathbf{x}, \mathbf{x})$                                              | Prior variance (amplitude) at $\mathbf{x}$                   |
| $k(\mathbf{x}, \mathbf{x}')$ large when $\mathbf{x} \approx \mathbf{x}'$ | Nearby inputs have similar outputs (smoothness)              |
| Length-scale $\ell$                                                      | How far apart inputs must be before they become uncorrelated |
| Differentiability of $k$                                                 | Smoothness of sampled functions                              |

---

### 4.2 Sampling from a GP Prior

To draw a function sample from the prior $f \sim \mathcal{GP}(0, k)$ at a grid $\mathbf{X}_*$:

1. Compute the $n_* \times n_*$ covariance matrix $K_{**} = k(\mathbf{X}_*, \mathbf{X}_*)$.
2. Add jitter: $K_{**} \leftarrow K_{**} + \delta I$ with $\delta \approx 10^{-6}$ (for numerical stability).
3. Cholesky decomposition: $K_{**} = LL^\top$.
4. Sample $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I)$.
5. The sample is $\mathbf{f}_* = L\mathbf{u}$.

```python
K_prior = rbf_kernel(X_grid, X_grid, length_scale=1.5)
K_prior += 1e-6 * np.eye(len(X_grid))
L = np.linalg.cholesky(K_prior)
sample = L @ np.random.randn(len(X_grid))
```

> **Notebook reference.** Cell 5 draws 5 prior samples from an RBF kernel with $\ell = 1.5$ on $[-5, 5]$. Note how the samples are smooth (infinitely differentiable) and vary on a length scale controlled by $\ell$.

---

## 5. GP Posterior (Regression)

### 5.1 The Conditioning Formula

Given training data $(\mathbf{X}, \mathbf{y})$ with $y_i = f(\mathbf{x}_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, and test inputs $\mathbf{X}_*$, the joint prior is:

$$\begin{bmatrix}\mathbf{y}\\\mathbf{f}_*\end{bmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\,\begin{bmatrix}K + \sigma_n^2 I & K_* \\ K_*^\top & K_{**}\end{bmatrix}\right)$$

where:
- $K = k(\mathbf{X}, \mathbf{X})$ — $n \times n$ training covariance.
- $K_* = k(\mathbf{X}, \mathbf{X}_*)$ — $n \times n_*$ cross-covariance.
- $K_{**} = k(\mathbf{X}_*, \mathbf{X}_*)$ — $n_* \times n_*$ test covariance.

Conditioning on $\mathbf{y}$ gives the **GP posterior**:

$$\boxed{\mathbf{f}_* \mid \mathbf{X}, \mathbf{y}, \mathbf{X}_* \sim \mathcal{N}(\boldsymbol{\mu}_*, \Sigma_*)}$$

$$\boxed{\boldsymbol{\mu}_* = K_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{y}}$$

$$\boxed{\Sigma_* = K_{**} - K_*^\top(K + \sigma_n^2 I)^{-1}K_*}$$

---

### 5.2 Derivation

The conditioning formula follows from the standard Gaussian conditioning identity. For a joint Gaussian:

$$\begin{bmatrix}\mathbf{a}\\\mathbf{b}\end{bmatrix} \sim \mathcal{N}\!\left(\begin{bmatrix}\boldsymbol{\mu}_a\\\boldsymbol{\mu}_b\end{bmatrix},\begin{bmatrix}\Sigma_{aa} & \Sigma_{ab}\\\Sigma_{ba} & \Sigma_{bb}\end{bmatrix}\right)$$

the conditional $\mathbf{b} \mid \mathbf{a}$ is:

$$\mathbf{b} \mid \mathbf{a} \sim \mathcal{N}\!\left(\boldsymbol{\mu}_b + \Sigma_{ba}\Sigma_{aa}^{-1}(\mathbf{a} - \boldsymbol{\mu}_a),\, \Sigma_{bb} - \Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}\right)$$

Substituting $\mathbf{a} = \mathbf{y}$, $\mathbf{b} = \mathbf{f}_*$, $\Sigma_{aa} = K + \sigma_n^2 I$, $\Sigma_{ab} = K_*$, $\Sigma_{ba} = K_*^\top$, $\Sigma_{bb} = K_{**}$, and $\boldsymbol{\mu}_a = \boldsymbol{\mu}_b = \mathbf{0}$ yields the result.

---

### 5.3 Cholesky Implementation

Direct matrix inversion is numerically unstable and costs $O(n^3)$. Instead, use the **Cholesky decomposition** $K + \sigma_n^2 I = LL^\top$:

$$\boldsymbol{\alpha} = L^{-\top}L^{-1}\mathbf{y}, \quad \boldsymbol{\mu}_* = K_*^\top\boldsymbol{\alpha}$$

$$V = L^{-1}K_*, \quad \Sigma_* = K_{**} - V^\top V$$

```python
def gp_posterior(X_train, y_train, X_test, kernel_fn, sigma_n=0.1):
    K    = kernel_fn(X_train, X_train) + sigma_n**2 * np.eye(len(X_train))
    K_s  = kernel_fn(X_train, X_test)
    K_ss = kernel_fn(X_test,  X_test)
    L     = np.linalg.cholesky(K + 1e-10 * np.eye(len(K)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu    = K_s.T @ alpha
    v     = np.linalg.solve(L, K_s)
    var   = np.diag(K_ss) - np.einsum('ij,ij->j', v, v)
    return mu, np.sqrt(np.maximum(var, 0))
```

> **Notebook reference.** Cell 7 implements this function and applies it to noisy sine observations. The posterior mean interpolates the data and the $\pm 2\sigma$ bands contract near observations and widen where data is sparse.

---

### 5.4 Noise-Free vs. Noisy Observations

| Setting                         | $K$ in posterior                                         | Behaviour                                                                   |
| ------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Noise-free** ($\sigma_n = 0$) | $K = k(\mathbf{X}, \mathbf{X}) + \delta I$ (jitter only) | Posterior mean **interpolates** data exactly; $\sigma_* = 0$ at data points |
| **Noisy** ($\sigma_n > 0$)      | $K = k(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I$           | Posterior mean **smooths** through data; $\sigma_* > 0$ even at data points |

> The noise parameter $\sigma_n^2$ is analogous to $1/\beta$ (noise variance) in BLR. It controls the **aleatoric uncertainty** — the irreducible noise floor.

---

### 5.5 Computational Cost

| Operation                         | Cost         |
| --------------------------------- | ------------ |
| Cholesky decomposition of $K$     | $O(n^3)$     |
| Solving for $\boldsymbol{\alpha}$ | $O(n^2)$     |
| Prediction at $n_*$ test points   | $O(n^2 n_*)$ |
| Storage                           | $O(n^2)$     |

**For $n > 10{,}000$**, exact GP regression becomes impractical. Approximations:
- **Sparse / inducing-point GPs** — select $m \ll n$ inducing points; cost $O(nm^2)$.
- **Random Fourier features** — approximate the kernel via random projections; cost $O(n)$.
- **KISS-GP / GPU acceleration** — exploit kernel structure for scalability.

For the surrogate-modelling use case (typically $n < 500$ evaluations), exact GPs are fast enough.

---

## 6. Kernel Functions

The kernel is the **design choice** in GP regression. It encodes all prior knowledge about the function.

### 6.1 What Makes a Valid Kernel

A function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a valid covariance function if and only if the matrix $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is **positive semi-definite** for any finite set of inputs. This is Mercer's condition.

**Building valid kernels.** If $k_1$ and $k_2$ are valid kernels, then so are:
- $k_1 + k_2$ (sum)
- $k_1 \cdot k_2$ (product)
- $c \cdot k_1$ for $c > 0$ (scaling)
- $k(\mathbf{x}, \mathbf{x}') = g(\mathbf{x})\,k_1(\mathbf{x}, \mathbf{x}')\,g(\mathbf{x}')$ for any function $g$

---

### 6.2 RBF (Squared Exponential)

$$\boxed{k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2\exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)}$$

| Hyperparameter | Name                        | Effect                                               |
| -------------- | --------------------------- | ---------------------------------------------------- |
| $\sigma_f^2$   | Signal variance (amplitude) | Vertical scale of function                           |
| $\ell$         | Length-scale                | Horizontal scale; larger $\ell$ → smoother functions |

**Properties:**
- **Infinitely differentiable** — samples are very smooth. This can be unrealistically smooth for some physical processes.
- **Stationary** — depends only on $\|\mathbf{x} - \mathbf{x}'\|$, not on absolute position.
- As $\|\mathbf{x} - \mathbf{x}'\| \gg \ell$, $k \to 0$ — distant points are uncorrelated.

> **Notebook reference.** Cell 5 uses `rbf_kernel` with $\ell = 1.5$ and Cell 7 uses $\ell = 1.0$ for the posterior.

---

### 6.3 Matérn Family

$$k_{\text{Matérn}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\,r}{\ell}\right)^{\!\nu}\!K_\nu\!\left(\frac{\sqrt{2\nu}\,r}{\ell}\right)$$

where $r = \|\mathbf{x} - \mathbf{x}'\|$ and $K_\nu$ is the modified Bessel function of the second kind.

**Special cases (closed-form):**

$$k_{\text{Matérn 1/2}}(r) = \sigma_f^2\exp\!\left(-\frac{r}{\ell}\right) \qquad \text{(Ornstein–Uhlenbeck; rough, non-differentiable)}$$

$$k_{\text{Matérn 3/2}}(r) = \sigma_f^2\left(1 + \frac{\sqrt{3}\,r}{\ell}\right)\exp\!\left(-\frac{\sqrt{3}\,r}{\ell}\right) \qquad \text{(once differentiable)}$$

$$k_{\text{Matérn 5/2}}(r) = \sigma_f^2\left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5r^2}{3\ell^2}\right)\exp\!\left(-\frac{\sqrt{5}\,r}{\ell}\right) \qquad \text{(twice differentiable)}$$

| $\nu$         | Smoothness                        | When to use                       |
| ------------- | --------------------------------- | --------------------------------- |
| $\frac{1}{2}$ | Continuous but not differentiable | Very rough processes              |
| $\frac{3}{2}$ | Once differentiable               | Default for many physical systems |
| $\frac{5}{2}$ | Twice differentiable              | Default in Bayesian optimisation  |
| $\to \infty$  | Infinitely differentiable         | Equivalent to RBF                 |

> **Matérn 5/2 is the standard choice for Bayesian optimisation** because it is smooth enough to be well-behaved but does not impose the unrealistic infinite differentiability of the RBF.

---

### 6.4 Periodic Kernel

$$\boxed{k_{\text{Per}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2\exp\!\left(-\frac{2\sin^2\!\left(\pi\frac{\|\mathbf{x} - \mathbf{x}'\|}{p}\right)}{\ell^2}\right)}$$

| Hyperparameter | Meaning                       |
| -------------- | ----------------------------- |
| $p$            | Period of the function        |
| $\ell$         | Smoothness within each period |
| $\sigma_f^2$   | Amplitude                     |

Useful for modelling **seasonal** or **cyclic** data (connects to [Week 09](../week09_time_series/theory.md)'s seasonality). Can be combined with an RBF to allow slowly varying periodicity: $k_{\text{RBF}} \cdot k_{\text{Per}}$.

---

### 6.5 Linear Kernel

$$k_{\text{Lin}}(\mathbf{x}, \mathbf{x}') = \sigma_b^2 + \sigma_v^2(\mathbf{x} - c)^\top(\mathbf{x}' - c)$$

This gives a **GP whose posterior mean is a linear function** — equivalent to Bayesian linear regression. Useful as a building block (e.g., $k_{\text{Lin}} \cdot k_{\text{Per}}$ models linear trends with periodic fluctuations).

---

### 6.6 Composing Kernels

| Operation                              | Effect                    | Example                                  |
| -------------------------------------- | ------------------------- | ---------------------------------------- |
| $k_1 + k_2$                            | Superposition of patterns | RBF + Periodic → smooth + oscillatory    |
| $k_1 \times k_2$                       | Interaction / modulation  | Linear × Periodic → growing oscillations |
| $k_1 + \text{WhiteKernel}(\sigma_n^2)$ | Adds observation noise    | Standard practice in sklearn             |

> **Notebook reference.** Cell 11 compares RBF, Matérn(1.5), and Periodic kernels on a seasonal 1D function. The log-marginal-likelihood ranks them — higher LML means the kernel better explains the data.

---

### 6.7 The Length-Scale Parameter

The length-scale $\ell$ is the most intuitive hyperparameter:

| $\ell$                         | Effect on GP                                                         |
| ------------------------------ | -------------------------------------------------------------------- |
| **Small $\ell$**               | Functions vary rapidly; GP fits local details; risk of overfitting   |
| **Large $\ell$**               | Functions vary slowly; GP smooths aggressively; risk of underfitting |
| **$\ell$ per dimension** (ARD) | Each input dimension has its own relevance; see Section 7.2          |

**Rule of thumb:** $\ell$ should be roughly the distance over which the function changes appreciably.

> **Suggested experiment.** In Cell 7, change `length_scale` from 0.3 to 5.0 and observe how the posterior mean and uncertainty bands change. With $\ell = 0.3$, the GP wiggles through each point; with $\ell = 5.0$, it draws a near-flat line with wide bands.

---

## 7. Hyperparameter Optimisation via Marginal Likelihood

### 7.1 The Log-Marginal-Likelihood

The hyperparameters $\boldsymbol{\theta} = \{\sigma_f, \ell, \sigma_n, \ldots\}$ are **not** fit by looking at training error. Instead, we maximise the **marginal likelihood** (evidence):

$$p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}) = \int p(\mathbf{y} \mid \mathbf{f})\,p(\mathbf{f} \mid \mathbf{X}, \boldsymbol{\theta})\,d\mathbf{f}$$

For a GP with Gaussian noise, this integral is tractable:

$$\boxed{\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y} - \frac{1}{2}\log|K_y| - \frac{n}{2}\log 2\pi}$$

where $K_y = K + \sigma_n^2 I$.

**The three terms:**

| Term                                             | Role                                                        | Analogy                     |
| ------------------------------------------------ | ----------------------------------------------------------- | --------------------------- |
| $-\frac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y}$ | **Data fit** — how well the model explains the observations | Training loss               |
| $-\frac{1}{2}\log                                | K_y                                                         | $                           | **Complexity penalty** — penalises overly flexible models | Regularisation |
| $-\frac{n}{2}\log 2\pi$                          | Normalisation constant                                      | Irrelevant for optimisation |

This embodies an **automatic Occam's razor**: the marginal likelihood naturally balances fit and complexity without a separate validation set.

> **Comparison with AIC/BIC ([Week 09](../week09_time_series/theory.md)).** AIC and BIC approximate the marginal likelihood. The GP marginal likelihood is the exact (Gaussian) integral — no approximation needed.

**In practice:**

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

gpr = GaussianProcessRegressor(
    kernel=1.0 * RBF(1.0) + WhiteKernel(1e-4),
    n_restarts_optimizer=5   # multiple random restarts for the LML maximiser
)
gpr.fit(X_train, y_train)
print(f'Optimised kernel: {gpr.kernel_}')
print(f'Log-ML: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.2f}')
```

---

### 7.2 Automatic Relevance Determination (ARD)

Use separate length-scales per input dimension:

$$k_{\text{ARD}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2\exp\!\left(-\frac{1}{2}\sum_{d=1}^{D}\frac{(x_d - x_d')^2}{\ell_d^2}\right)$$

If the marginal likelihood drives $\ell_d \to \infty$ for dimension $d$, then input $d$ is irrelevant (the function doesn't vary along it). This is automatic feature selection — the GP equivalent of Lasso ([Week 06](../../02_fundamentals/week06_regularization/theory.md)).

---

## 8. Calibration and Coverage

A properly specified GP should produce **well-calibrated** uncertainty estimates ([Week 08](../week08_uncertainty/theory.md)). Specifically:

$$P\!\left(y_* \in [\mu_* - z_{1-\alpha/2}\sigma_*, \, \mu_* + z_{1-\alpha/2}\sigma_*]\right) \approx 1 - \alpha$$

**Coverage check procedure:**
1. Generate test data $(X_{\text{test}}, y_{\text{test}})$ from the true function (with noise).
2. Predict $\mu_*$ and $\sigma_*$ at each test point.
3. Check the fraction of test points falling inside the $\pm 2\sigma$ band.
4. This fraction should be approximately 95%.

**Potential miscalibration sources:**
| Cause                         | Effect                           | Fix                                                     |
| ----------------------------- | -------------------------------- | ------------------------------------------------------- |
| Wrong kernel family           | Uncertainty over/under-estimated | Try alternative kernels; compare LML                    |
| Fixed (wrong) hyperparameters | Overconfident or underconfident  | Optimise hyperparameters via marginal likelihood        |
| Non-Gaussian noise            | Coverage may deviate             | Use Student-$t$ likelihood or robust GP                 |
| Too few data points           | Hyperparameters poorly estimated | Use prior on hyperparameters (fully Bayesian treatment) |

> **Notebook reference.** Exercise 2 generates 50 training + 200 test points and checks that ~95% of test points fall inside the 95% credible interval.

---

## 9. Bayesian Optimisation

### 9.1 The Optimisation Loop

**Goal:** find $\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ using as few evaluations of $f$ as possible.

**Bayesian Optimisation (BO) algorithm:**

1. **Initialise:** evaluate $f$ at $n_0$ random points.
2. **Loop** for $t = n_0 + 1, \ldots, T$:
   a. Fit a GP surrogate to all observations $\{(\mathbf{x}_i, y_i)\}_{i=1}^{t-1}$.
   b. Compute the **acquisition function** $a(\mathbf{x})$ over the search space.
   c. Select the next query point: $\mathbf{x}_t = \arg\max_{\mathbf{x}} a(\mathbf{x})$.
   d. Evaluate $y_t = f(\mathbf{x}_t) + \epsilon$.
   e. Augment the dataset.
3. **Return** $\mathbf{x}^* = \arg\min_i y_i$.

```
Iteration 1:  GP ──► acquisition ──► [x_new] ──► f(x_new) ──► update GP
Iteration 2:  GP ──► acquisition ──► [x_new] ──► f(x_new) ──► update GP
     ...
Iteration T:  return best
```

---

### 9.2 Expected Improvement (EI)

The most widely used acquisition function. For **minimisation**, let $f^* = \min_i y_i$ (best value so far):

$$\boxed{\text{EI}(\mathbf{x}) = \mathbb{E}\!\left[\max(f^* - f(\mathbf{x}), 0)\right]}$$

Under the GP posterior $f(\mathbf{x}) \sim \mathcal{N}(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))$, this has a **closed-form** solution:

$$\text{EI}(\mathbf{x}) = (f^* - \mu(\mathbf{x}) - \xi)\,\Phi(Z) + \sigma(\mathbf{x})\,\phi(Z)$$

$$Z = \frac{f^* - \mu(\mathbf{x}) - \xi}{\sigma(\mathbf{x})}$$

where $\Phi$ and $\phi$ are the standard normal CDF and PDF, and $\xi \geq 0$ is a small **exploration parameter** (typically $\xi = 0.01$).

**Derivation sketch.** Let $I(\mathbf{x}) = \max(f^* - f(\mathbf{x}), 0)$. Then:

$$\text{EI} = \int_{-\infty}^{f^*}(f^* - f)\,\frac{1}{\sigma}\phi\!\left(\frac{f - \mu}{\sigma}\right)df$$

Substituting $z = (f - \mu)/\sigma$ and splitting the integral:

$$= (f^* - \mu)\Phi(Z) + \sigma\phi(Z)$$

| Term                 | Role                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------ |
| $(f^* - \mu)\Phi(Z)$ | **Exploitation** — large when $\mu$ is well below $f^*$ (model predicts improvement) |
| $\sigma\phi(Z)$      | **Exploration** — large when $\sigma$ is high (uncertain region)                     |

> **Notebook reference.** Cell 13 implements `expected_improvement` and plots it below the GP posterior. The EI peak balances where the mean is low (exploitation) and where uncertainty is high (exploration). Cell 15 runs the full BO loop on the Branin function.

---

### 9.3 Upper Confidence Bound (UCB)

A simpler acquisition function:

$$\text{UCB}(\mathbf{x}) = \mu(\mathbf{x}) - \kappa\,\sigma(\mathbf{x}) \qquad \text{(for minimisation; select } \arg\min)$$

or equivalently for maximisation:

$$\text{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + \kappa\,\sigma(\mathbf{x}) \qquad \text{(select } \arg\max)$$

The parameter $\kappa$ directly controls the exploration–exploitation trade-off:

| $\kappa$                 | Behaviour                                    |
| ------------------------ | -------------------------------------------- |
| $\kappa = 0$             | Pure exploitation (greedy)                   |
| $\kappa$ large (e.g., 3) | More exploration (prefers uncertain regions) |
| $\kappa = 2$             | Common default; roughly matches EI           |

> **Exercise 4 in the notebook** asks you to implement UCB and compare it to EI.

---

### 9.4 Exploration–Exploitation Trade-Off

The fundamental tension in sequential decision-making:

| Strategy                                          | Pros                                          | Cons                                              |
| ------------------------------------------------- | --------------------------------------------- | ------------------------------------------------- |
| **Exploitation** (query where $\mu$ is best)      | Fast convergence if the region is correct     | Gets stuck in local optima; misses better regions |
| **Exploration** (query where $\sigma$ is largest) | Discovers the global structure                | Wastes budget on irrelevant regions               |
| **Balanced (EI, UCB)**                            | Systematically covers the space and converges | Requires GP to be well-calibrated                 |

> **Key insight.** BO outperforms random search precisely because the GP's **uncertainty estimate guides the search**. If the GP were only a point predictor (like a neural network without uncertainty), there would be no principled way to balance exploration and exploitation.

---

### 9.5 Practical Considerations

| Issue                        | Recommendation                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Acquisition optimisation** | Maximise EI over a dense random grid or use L-BFGS-B with multiple restarts                                                     |
| **Kernel choice**            | Matérn 5/2 + WhiteKernel is the standard starting point for BO                                                                  |
| **Normalisation**            | Set `normalize_y=True` in sklearn; rescale inputs to $[0, 1]^D$                                                                 |
| **Initialisation**           | Use Latin Hypercube Sampling (LHS) for initial points (better coverage than uniform random)                                     |
| **Budget**                   | BO excels for budgets of 20–200 evaluations; for larger budgets, random search catches up                                       |
| **Dimensionality**           | Standard BO works well for $D \leq 20$; high-dimensional BO requires special techniques (random embeddings, additive structure) |
| **Batch BO**                 | When you can run multiple evaluations in parallel, use batch acquisition functions (e.g., q-EI, Thompson sampling)              |

**Libraries for production BO:**
- `BoTorch` (PyTorch-based; supports batch BO, multi-objective BO).
- `Optuna` (practical HPO; uses Tree-structured Parzen Estimator by default, but supports GP).
- `scikit-optimize` (`skopt`; simple sklearn-compatible interface).
- `GPyOpt` (built on GPy; research-oriented).

> **Notebook reference.** Cell 15 runs GP + EI on the 2D Branin function (30 evaluations) and compares to random search. BO finds a value close to the global minimum ($\approx 0.397$) much faster.

---

## 10. Connections to the Rest of the Course

| Week                            | Connection                                                                                         |
| ------------------------------- | -------------------------------------------------------------------------------------------------- |
| **[Week 03](../../02_fundamentals/week03_linear_models/theory.md) (Linear Models)**     | GP with a linear kernel = Bayesian linear regression                                               |
| **[Week 06](../../02_fundamentals/week06_regularization/theory.md) (Regularisation)**    | Marginal likelihood provides automatic Occam's razor (no separate validation set)                  |
| **[Week 07](../week07_likelihood/theory.md) (Likelihood)**        | GP hypers are optimised by maximising the marginal likelihood = type-II MLE                        |
| **[Week 08](../week08_uncertainty/theory.md) (Uncertainty)**       | GP posterior is the gold standard for calibrated uncertainty; posterior = aleatoric + epistemic    |
| **[Week 09](../week09_time_series/theory.md) (Time Series)**       | Periodic/Matérn kernels model temporal correlation; GP-based time-series = prior over trajectories |
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md) (Neural Networks)**   | Infinite-width neural networks converge to GPs (Neal, 1996; Neural Tangent Kernel)                 |
| **[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md) (Training at Scale)** | BO is used for hyperparameter tuning of deep networks                                              |
| **[Week 17](../../06_sequence_models/week17_attention/theory.md) (Attention)**         | The attention matrix $QK^\top/\sqrt{d}$ is a kernel Gram matrix; attention ≈ kernel smoothing      |
| **[Week 20](../../08_deployment/week20_deployment/theory.md) (Deployment)**        | BO for hyperparameter optimisation in production ML pipelines                                      |

---

## 11. Notebook Reference Guide

| Cell                   | Section                 | What it demonstrates                                                  | Theory reference |
| ---------------------- | ----------------------- | --------------------------------------------------------------------- | ---------------- |
| 5 (GP Prior)           | Prior samples           | 5 function draws from RBF prior ($\ell = 1.5$)                        | Section 4        |
| 7 (GP Posterior)       | From-scratch posterior  | Cholesky-based posterior on noisy sine; $\pm 2\sigma$ bands           | Section 5        |
| 9 (sklearn)            | GP comparison           | From-scratch vs. sklearn `GaussianProcessRegressor`; optimised hypers | Section 7        |
| 11 (Kernel comparison) | Kernel selection        | RBF vs. Matérn(1.5) vs. Periodic; log-marginal-likelihood ranking     | Section 6        |
| 13 (EI)                | Acquisition function    | 1D EI plot; exploration vs. exploitation; next query point            | Section 9.2      |
| 15 (BO)                | Bayesian optimisation   | GP + EI on 2D Branin (30 evals) vs. random search; convergence trace  | Section 9.1      |
| Ex. 1                  | From-scratch validation | `gp_posterior` with fixed hypers matches sklearn (`optimizer=None`)   | Section 5.3      |
| Ex. 2                  | Coverage check          | 95% credible interval covers ~95% of test points                      | Section 8        |
| Ex. 3                  | Kernel exploration      | Add `RBF + ExpSineSquared` composite kernel                           | Section 6.6      |
| Ex. 4                  | UCB implementation      | Compare UCB to EI on 1D sin                                           | Section 9.3      |
| Ex. 5                  | BO visualisation        | 50-eval Branin contour coloured by iteration order                    | Section 9.4      |

**Suggested modifications:**

| Modification                                           | What it reveals                                                           |
| ------------------------------------------------------ | ------------------------------------------------------------------------- |
| Change `length_scale` from 0.3 to 5.0 in `rbf_kernel`  | Short $\ell$ → wiggly interpolation; long $\ell$ → smooth with wide bands |
| Set `sigma_n=0` (noise-free)                           | Posterior passes exactly through data; zero variance at observations      |
| Replace RBF with Matérn 1/2 in prior sampling          | Samples become rough (non-differentiable; looks like Brownian motion)     |
| Increase training data from 5 to 50 points             | Epistemic uncertainty shrinks everywhere; bands become very narrow        |
| Run BO with $\kappa = 0$ (pure exploitation) on Branin | Gets stuck near the first good point; misses better global optima         |
| Run BO with $\kappa = 10$ (heavy exploration)          | Wastes budget exploring globally; slow convergence but good coverage      |
| Try BO on a 5D function                                | Performance degrades; requires more initial points and budget             |

---

## 12. Symbol Reference

| Symbol                                                    | Name                          | Meaning                                                                   |
| --------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------- |
| $f$                                                       | Latent function               | The unknown true function                                                 |
| $\mathbf{x}$                                              | Input                         | Point in $\mathcal{X} \subseteq \mathbb{R}^D$                             |
| $y$                                                       | Observation                   | $f(\mathbf{x}) + \epsilon$                                                |
| $\mathcal{GP}(m, k)$                                      | Gaussian process              | Distribution over functions with mean $m$, kernel $k$                     |
| $m(\mathbf{x})$                                           | Mean function                 | $\mathbb{E}[f(\mathbf{x})]$; typically $0$                                |
| $k(\mathbf{x}, \mathbf{x}')$                              | Kernel / covariance function  | $\text{Cov}[f(\mathbf{x}), f(\mathbf{x}')]$                               |
| $K$                                                       | Training covariance matrix    | $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$; $n \times n$                    |
| $K_*$                                                     | Cross-covariance              | $k(\mathbf{X}_{\text{train}}, \mathbf{X}_{\text{test}})$; $n \times n_*$  |
| $K_{**}$                                                  | Test covariance               | $k(\mathbf{X}_{\text{test}}, \mathbf{X}_{\text{test}})$; $n_* \times n_*$ |
| $\boldsymbol{\mu}_*$                                      | Posterior mean                | $K_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{y}$                               |
| $\Sigma_*$                                                | Posterior covariance          | $K_{**} - K_*^\top(K + \sigma_n^2 I)^{-1}K_*$                             |
| $\sigma_n^2$                                              | Noise variance                | Observation noise; aleatoric component                                    |
| $\sigma_f^2$                                              | Signal variance               | Kernel amplitude hyperparameter                                           |
| $\ell$                                                    | Length-scale                  | Controls smoothness / correlation range                                   |
| $\nu$                                                     | Matérn smoothness             | Controls differentiability of samples                                     |
| $p$                                                       | Period                        | Periodic kernel period                                                    |
| $\boldsymbol{\theta}$                                     | Kernel hyperparameters        | $\{\sigma_f, \ell, \sigma_n, \nu, \ldots\}$                               |
| $\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta})$ | Log-marginal-likelihood (LML) | Objective for hyperparameter optimisation                                 |
| $L$                                                       | Cholesky factor               | $K + \sigma_n^2 I = LL^\top$                                              |
| $\boldsymbol{\alpha}$                                     | Dual coefficients             | $L^{-\top}L^{-1}\mathbf{y}$; used for predictions                         |
| $f^*$                                                     | Best observed value           | $\min_i y_i$ (minimisation) or $\max_i y_i$                               |
| $\text{EI}(\mathbf{x})$                                   | Expected Improvement          | Acquisition function: expected reduction below $f^*$                      |
| $\xi$                                                     | EI jitter parameter           | Controls exploration; typically 0.01                                      |
| $\kappa$                                                  | UCB parameter                 | Exploration weight in $\mu \pm \kappa\sigma$                              |
| $\Phi, \phi$                                              | Standard normal CDF, PDF      | Used in closed-form EI                                                    |
| $Z$                                                       | Standardised improvement      | $(f^* - \mu - \xi)/\sigma$                                                |
| $B$                                                       | Backshift operator            | From [Week 09](../week09_time_series/theory.md); not used here                                               |

---

## 13. References

1. Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. Available free online at [gaussianprocess.org/gpml](http://www.gaussianprocess.org/gpml/). — **The** reference for GP regression; covers priors, posteriors, kernels, hyperparameter optimisation, approximations.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 6. Springer. — Connects GPs to kernel methods and BLR; derives the weight-space and function-space views.
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapter 15. MIT Press. — GP regression and classification; Bayesian optimisation.
4. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization." *Proceedings of the IEEE*, 104(1), 148–175. — Comprehensive survey of Bayesian optimisation.
5. Snoek, J., Larochelle, H., & Adams, R. P. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." *NeurIPS*. — Introduced BO for hyperparameter tuning of neural networks (Spearmint).
6. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). "Efficient Global Optimization of Expensive Black-Box Functions." *Journal of Global Optimization*, 13(4), 455–492. — The original EGO paper; derives Expected Improvement.
7. Matérn, B. (1960). *Spatial Variation*. Springer (reprinted 1986). — Original Matérn covariance family.
8. Neal, R. M. (1996). *Bayesian Learning for Neural Networks*. Springer. — Shows that infinite-width neural networks converge to GPs; foundation for the Neural Tangent Kernel.
9. Garnett, R. (2023). *Bayesian Optimization*. Cambridge University Press. Available free at [bayesoptbook.com](https://bayesoptbook.com/). — Modern, comprehensive textbook covering theory and practice.
10. Balandat, M., et al. (2020). "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization." *NeurIPS*. — The BoTorch library for scalable, GPU-accelerated BO.
