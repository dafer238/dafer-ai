# Regularisation & Validation

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Why Regularisation? The Overfitting Problem Revisited](#2-why-regularisation-the-overfitting-problem-revisited)
    - 2.1 [Bias–Variance Recap](#21-biasvariance-recap)
    - 2.2 [Penalised Objectives: The General Idea](#22-penalised-objectives-the-general-idea)
3. [Ridge Regression (L2 Regularisation)](#3-ridge-regression-l2-regularisation)
    - 3.1 [The Objective](#31-the-objective)
    - 3.2 [Closed-Form Solution](#32-closed-form-solution)
    - 3.3 [Geometric Interpretation: Constrained Optimisation](#33-geometric-interpretation-constrained-optimisation)
    - 3.4 [Ridge and SVD — Shrinkage of Singular Values](#34-ridge-and-svd--shrinkage-of-singular-values)
    - 3.5 [Bayesian Interpretation](#35-bayesian-interpretation)
    - 3.6 [Effect on Bias and Variance](#36-effect-on-bias-and-variance)
4. [Lasso Regression (L1 Regularisation)](#4-lasso-regression-l1-regularisation)
    - 4.1 [The Objective](#41-the-objective)
    - 4.2 [Sparsity: Why Lasso Produces Zeros](#42-sparsity-why-lasso-produces-zeros)
    - 4.3 [Soft-Thresholding and Coordinate Descent](#43-soft-thresholding-and-coordinate-descent)
    - 4.4 [Bayesian Interpretation: Laplace Prior](#44-bayesian-interpretation-laplace-prior)
    - 4.5 [Ridge vs. Lasso: Summary](#45-ridge-vs-lasso-summary)
5. [Elastic Net](#5-elastic-net)
    - 5.1 [The Objective](#51-the-objective)
    - 5.2 [When to Use Elastic Net](#52-when-to-use-elastic-net)
6. [Cross-Validation](#6-cross-validation)
    - 6.1 [The Train/Validation/Test Protocol](#61-the-trainvalidationtest-protocol)
    - 6.2 [K-Fold Cross-Validation](#62-k-fold-cross-validation)
    - 6.3 [Stratified K-Fold](#63-stratified-k-fold)
    - 6.4 [Leave-One-Out Cross-Validation](#64-leave-one-out-cross-validation)
    - 6.5 [Validation Curves and Learning Curves](#65-validation-curves-and-learning-curves)
7. [Time-Series Cross-Validation](#7-time-series-cross-validation)
    - 7.1 [Why Standard K-Fold Fails for Time Series](#71-why-standard-k-fold-fails-for-time-series)
    - 7.2 [Walk-Forward Validation](#72-walk-forward-validation)
    - 7.3 [Lag Features](#73-lag-features)
8. [Early Stopping](#8-early-stopping)
    - 8.1 [Early Stopping as Implicit Regularisation](#81-early-stopping-as-implicit-regularisation)
    - 8.2 [Equivalence to L2 Regularisation (Linear Case)](#82-equivalence-to-l2-regularisation-linear-case)
9. [The Regularisation Landscape: A Unified View](#9-the-regularisation-landscape-a-unified-view)
10. [Connections to the Rest of the Course](#10-connections-to-the-rest-of-the-course)
11. [Notebook Reference Guide](#11-notebook-reference-guide)
12. [Symbol Reference](#12-symbol-reference)
13. [References](#13-references)

---

## 1. Scope and Purpose

[Week 03](../week03_linear_models/theory.md#3-linear-regression) introduced linear and logistic regression: powerful tools but susceptible to **overfitting**, especially when the number of features $d$ is large relative to the number of samples $n$. [Week 04](../week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view) showed that PCA can reduce dimensionality before fitting, but that is an **indirect** approach — it discards variance in an unsupervised manner, without looking at the labels.

This week introduces the **direct** approach: modify the objective function to penalise complexity. This is **regularisation** and it is arguably the single most important technique in all of machine learning. Every model from here forward — neural networks ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#9-regularisation-in-neural-networks)), CNNs ([Week 15](../../05_deep_learning/week15_cnn_representations/theory.md)), transformers ([Week 18](../../06_sequence_models/week18_transformers/theory.md)) — uses some form of regularisation.

Alongside regularisation, we cover **cross-validation**, the standard protocol for choosing the regularisation strength (and, more broadly, any hyperparameter) without overfitting to the validation data.

**Prerequisites.** [Week 03](../week03_linear_models/theory.md#3-linear-regression) (linear/logistic regression, MSE, normal equations, bias–variance tradeoff), [Week 04](../week04_dimensionality_reduction/theory.md#5-pca-via-the-singular-value-decomposition) (SVD, eigendecomposition).

---

## 2. Why Regularisation? The Overfitting Problem Revisited

### 2.1 Bias–Variance Recap

Recall from [Week 03](../week03_linear_models/theory.md#4-statistical-interpretation-of-linear-regression) (Section 4) that the expected prediction error decomposes as:

$$\mathbb{E}\left[(y - \hat{f}(\mathbf{x}))^2\right] = \underbrace{\text{Bias}[\hat{f}(\mathbf{x})]^2}_{\text{systematic error}} + \underbrace{\text{Var}[\hat{f}(\mathbf{x})]}_{\text{sensitivity to training set}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

| Regime                          | Problem                            | Symptom                                   |
| ------------------------------- | ---------------------------------- | ----------------------------------------- |
| **Underfitting** (high bias)    | Model too simple; misses structure | Training and validation error both high   |
| **Overfitting** (high variance) | Model too complex; memorises noise | Training error low, validation error high |

The **goal of regularisation** is to reduce variance at a small cost in bias, yielding a net decrease in total error. This is the essence of the bias–variance tradeoff.

---

### 2.2 Penalised Objectives: The General Idea

All regularised models share the same structure:

$$\boxed{\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \left[\underbrace{\mathcal{L}(\boldsymbol{\theta})}_{\text{data fit}} + \underbrace{\lambda \, \Omega(\boldsymbol{\theta})}_{\text{complexity penalty}}\right]}$$

| Component                          | Role                                          | Examples                                                   |
| ---------------------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| $\mathcal{L}(\boldsymbol{\theta})$ | Loss function (measures fit to training data) | MSE, cross-entropy                                         |
| $\Omega(\boldsymbol{\theta})$      | Penalty (measures model complexity)           | $\|\boldsymbol{\theta}\|_2^2$, $\|\boldsymbol{\theta}\|_1$ |
| $\lambda \geq 0$                   | Regularisation strength (hyperparameter)      | Trades off fit vs. simplicity                              |

- $\lambda = 0$: no regularisation; reduces to the ordinary (unpenalised) estimator.
- $\lambda \to \infty$: the penalty dominates; parameters are forced toward zero, producing an underfit model.
- Optimal $\lambda$: found by cross-validation.

---

## 3. Ridge Regression (L2 Regularisation)

### 3.1 The Objective

Ridge regression (Hoerl & Kennard, 1970) adds the squared L2-norm of the parameter vector to the MSE:

$$\boxed{J_{\text{Ridge}}(\mathbf{w}) = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|_2^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top\mathbf{w})^2 + \lambda\sum_{j=1}^{d}w_j^2}$$

| Symbol                          | Meaning                                                                    |
| ------------------------------- | -------------------------------------------------------------------------- |
| $X \in \mathbb{R}^{n \times d}$ | Design matrix (rows are samples, columns are features)                     |
| $\mathbf{w} \in \mathbb{R}^d$   | Weight vector (we do **not** regularise the intercept $b$; see note below) |
| $\mathbf{y} \in \mathbb{R}^n$   | Target vector                                                              |
| $\lambda > 0$                   | Regularisation strength (called `alpha` in sklearn)                        |

> **Note on the intercept.** The penalty applies only to $\mathbf{w}$, not to the intercept $b$. Penalising $b$ would push the predictions toward zero, which is wrong — we want to penalise the _complexity_ of the model (the magnitude of the weights), not its baseline output. Sklearn handles this automatically: `Ridge(fit_intercept=True)` centres the data internally before fitting.

> **Note on scaling.** Because the penalty sums $w_j^2$, a feature measured in millimetres will receive a much smaller weight (and thus less penalty) than one measured in metres. **Always standardise features before Ridge/Lasso.** The notebook does this: `StandardScaler().fit_transform(X)`.

---

### 3.2 Closed-Form Solution

Setting the gradient to zero:

$$\nabla_\mathbf{w} J = -\frac{2}{n}X^\top(\mathbf{y} - X\mathbf{w}) + 2\lambda\mathbf{w} = \mathbf{0}$$

Solving:

$$X^\top X\mathbf{w} + n\lambda\mathbf{w} = X^\top\mathbf{y}$$

$$\boxed{\hat{\mathbf{w}}_{\text{Ridge}} = (X^\top X + n\lambda I)^{-1}X^\top\mathbf{y}}$$

> Some references absorb the $1/n$ into the loss and write $\hat{\mathbf{w}} = (X^\top X + \lambda I)^{-1}X^\top\mathbf{y}$. The two forms are equivalent up to a rescaling of $\lambda$. Sklearn uses $\hat{\mathbf{w}} = (X^\top X + \alpha I)^{-1}X^\top\mathbf{y}$, where `alpha` plays the role of $n\lambda$.

**Key observation.** Compare with the OLS solution from [Week 03](../week03_linear_models/theory.md#33-the-normal-equations-closed-form-solution):

| Estimator | Formula                                        | Invertible?                                                                               |
| --------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| OLS       | $(X^\top X)^{-1}X^\top\mathbf{y}$              | Only if $X^\top X$ is full rank ($n \geq d$ and no collinearity)                          |
| Ridge     | $(X^\top X + n\lambda I)^{-1}X^\top\mathbf{y}$ | **Always** (adding $n\lambda I$ makes the matrix positive definite for any $\lambda > 0$) |

This is one of Ridge's most important practical benefits: it makes the normal equations **unconditionally invertible**, even when $d > n$ or the features are highly correlated.

---

### 3.3 Geometric Interpretation: Constrained Optimisation

The penalised objective is equivalent (by Lagrange duality) to the _constrained_ problem:

$$\min_\mathbf{w} \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2 \quad \text{subject to} \quad \|\mathbf{w}\|_2^2 \leq t$$

for some budget $t$ that depends on $\lambda$ (there is a one-to-one mapping between $\lambda$ and $t$).

**Visualising in 2D ($d = 2$):**
- The MSE contours are **ellipses** centred at the OLS estimate $\hat{\mathbf{w}}_{\text{OLS}}$.
- The constraint region $\|\mathbf{w}\|_2^2 \leq t$ is a **circle** (a ball) centred at the origin.
- The Ridge solution is the point on the circle boundary closest to $\hat{\mathbf{w}}_{\text{OLS}}$ (the first ellipse that touches the circle).

This is the exact same picture as in the "constraint ball" illustration in most textbook treatments. The circle is smooth, so the tangent point is almost never on an axis — meaning **Ridge almost never produces exactly zero weights**.

---

### 3.4 Ridge and SVD — Shrinkage of Singular Values

To understand Ridge's effect precisely, express the solution via the SVD of $X$ ([Week 04](../week04_dimensionality_reduction/theory.md#5-pca-via-the-singular-value-decomposition)). Let $X = U S V^\top$ where $S = \text{diag}(s_1, \ldots, s_p)$ with $p = \min(n, d)$.

The OLS fitted values are:

$$\hat{\mathbf{y}}_{\text{OLS}} = X\hat{\mathbf{w}}_{\text{OLS}} = X(X^\top X)^{-1}X^\top\mathbf{y} = \sum_{j=1}^{p}\mathbf{u}_j\mathbf{u}_j^\top\mathbf{y}$$

The Ridge fitted values become:

$$\hat{\mathbf{y}}_{\text{Ridge}} = X\hat{\mathbf{w}}_{\text{Ridge}} = \sum_{j=1}^{p}\frac{s_j^2}{s_j^2 + n\lambda}\,\mathbf{u}_j\mathbf{u}_j^\top\mathbf{y}$$

Each component is multiplied by a **shrinkage factor**:

$$\boxed{f_j = \frac{s_j^2}{s_j^2 + n\lambda} \in [0, 1)}$$

| Singular value                    | Shrinkage $f_j$ | Effect                                                       |
| --------------------------------- | --------------- | ------------------------------------------------------------ |
| $s_j \gg \sqrt{n\lambda}$ (large) | $f_j \approx 1$ | Almost no shrinkage (high-variance directions are preserved) |
| $s_j \ll \sqrt{n\lambda}$ (small) | $f_j \approx 0$ | Nearly eliminated (noisy directions are suppressed)          |

**Interpretation.** Ridge continuously shrinks all coefficients toward zero, with the strongest shrinkage applied to directions of low variance in $X$. This is exactly where overfitting is most dangerous (small singular values correspond to poorly estimated directions — see [Week 04](../week04_dimensionality_reduction/theory.md#52-connecting-svd-to-pca), Section 4.4).

**Effective degrees of freedom** of Ridge regression:

$$\text{df}(\lambda) = \sum_{j=1}^{p}\frac{s_j^2}{s_j^2 + n\lambda}$$

When $\lambda = 0$, $\text{df} = p$ (unregularised). As $\lambda \to \infty$, $\text{df} \to 0$.

---

### 3.5 Bayesian Interpretation

If the data are generated by $y = \mathbf{x}^\top\mathbf{w}^* + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$ and we place a **Gaussian (normal) prior** on the weights:

$$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$$

then the **Maximum A Posteriori (MAP)** estimate is:

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_\mathbf{w} \underbrace{p(\mathbf{y} \mid X, \mathbf{w})}_{\text{likelihood}} \cdot \underbrace{p(\mathbf{w})}_{\text{prior}} = \arg\min_\mathbf{w} \left[\frac{1}{2\sigma^2}\|\mathbf{y} - X\mathbf{w}\|^2 + \frac{1}{2\tau^2}\|\mathbf{w}\|_2^2\right]$$

This is exactly Ridge regression with $\lambda = \sigma^2 / (n\tau^2)$.

| Prior width $\tau^2$              | Implied $\lambda$ | Interpretation                                      |
| --------------------------------- | ----------------- | --------------------------------------------------- |
| $\tau^2 \to \infty$ (vague prior) | $\lambda \to 0$   | Weights are unconstrained → OLS                     |
| $\tau^2$ small (tight prior)      | $\lambda$ large   | Weights are forced near zero → heavy regularisation |

> **Intuition.** Regularisation strength encodes your prior belief about the scale of the true weights. A strong penalty means you believe a priori that the weights are small.
>
> **Forward pointer.** Bayesian inference goes beyond MAP estimation — it computes the full posterior $p(\mathbf{w} \mid \mathbf{y}, X)$. This is the topic of [Week 08](../../03_probability/week08_uncertainty/theory.md#7-bayesian-inference) (Uncertainty).

---

### 3.6 Effect on Bias and Variance

The OLS estimator is **unbiased** (Gauss–Markov, [Week 03](../week03_linear_models/theory.md#43-properties-of-the-ols-estimator)) but can have **high variance** (especially when $d \approx n$ or features are correlated).

Ridge introduces **bias** (the estimator is pulled toward zero) but **reduces variance** (by shrinking unstable coefficients). The net effect on MSE:

$$\text{MSE}(\hat{\mathbf{w}}_{\text{Ridge}}) = \text{Bias}^2 + \text{Variance} = \underbrace{\lambda^2\|\mathbf{w}^*\|^2 \cdot C_1}_{\text{increases with } \lambda} + \underbrace{\sigma^2 \cdot C_2 / \lambda}_{\text{decreases with } \lambda}$$

(where $C_1, C_2$ absorb constants depending on $X$). The trade-off yields an interior optimum $\lambda^*$ that minimises the sum.

**Concrete example.** Suppose $d = 50$ features but only 10 are informative (as in the notebook). OLS gives 50 non-zero weights, many of which are fitting noise. Ridge shrinks all 50 weights, damping the noisy ones. The notebook's Cell 5 shows this: Ridge has all 50 weights non-zero but shrunk; Lasso sets ~40 to exactly zero.

---

## 4. Lasso Regression (L1 Regularisation)

### 4.1 The Objective

Lasso (Least Absolute Shrinkage and Selection Operator; Tibshirani, 1996) replaces the squared L2 penalty with the L1 norm:

$$\boxed{J_{\text{Lasso}}(\mathbf{w}) = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|_1 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top\mathbf{w})^2 + \lambda\sum_{j=1}^{d}|w_j|}$$

The L1 norm is $\|\mathbf{w}\|_1 = \sum_j |w_j|$: the sum of absolute values.

**Key difference from Ridge:** the $|\cdot|$ function has a **kink** at zero. This causes Lasso to produce **exact zeros** in the weight vector — it performs **automatic feature selection**.

---

### 4.2 Sparsity: Why Lasso Produces Zeros

#### The geometric argument

The constraint region for L1 is $\|\mathbf{w}\|_1 \leq t$, which in 2D is a **diamond** (rotated square). The MSE ellipses are most likely to touch the diamond at a **corner**, where one coordinate equals zero.

In contrast, Ridge's circle is smooth — the tangent point is (generically) not on an axis.

In higher dimensions, the L1 ball $\|\mathbf{w}\|_1 \leq t$ has $2d$ vertices (at $\pm t \mathbf{e}_j$), many edges, and many faces — all of which correspond to **sparse** solutions (one or more $w_j = 0$). The OLS ellipsoid generically touches these sparse facets rather than the smooth interior.

#### The subgradient argument

The L1 norm is not differentiable at $w_j = 0$. The **subgradient** of $|w_j|$ is:

$$\partial |w_j| = \begin{cases} \{+1\} & w_j > 0 \\ [-1, +1] & w_j = 0 \\ \{-1\} & w_j < 0 \end{cases}$$

The optimality condition for coordinate $j$ is:

$$\frac{\partial \mathcal{L}}{\partial w_j}\bigg|_{w_j = 0} + \lambda \cdot g = 0, \quad g \in [-1, 1]$$

If the data-fit gradient at $w_j = 0$ is small enough (i.e., $\left|\frac{\partial \mathcal{L}}{\partial w_j}\right| \leq \lambda$), the subgradient condition can be satisfied with $w_j = 0$ exactly. This is the mathematical reason for sparsity.

---

### 4.3 Soft-Thresholding and Coordinate Descent

Lasso has **no closed-form solution** (the L1 penalty is non-smooth). It is typically solved by **coordinate descent**: iterate over coordinates $j = 1, \ldots, d$, optimising each $w_j$ while holding the others fixed.

The single-coordinate update has a closed-form (**soft-thresholding operator**):

$$\hat{w}_j = \mathcal{S}\!\left(\tilde{w}_j, \, \lambda\right) = \text{sign}(\tilde{w}_j)\max\!\left(|\tilde{w}_j| - \lambda, 0\right)$$

where $\tilde{w}_j$ is the OLS solution for coordinate $j$ with all other coordinates fixed:

$$\tilde{w}_j = \frac{1}{\|\mathbf{x}_j\|^2}\mathbf{x}_j^\top(\mathbf{y} - X_{-j}\mathbf{w}_{-j})$$

**Interpretation of soft-thresholding:**
- If $|\tilde{w}_j| \leq \lambda$: the signal on feature $j$ is too weak to survive the penalty → set $\hat{w}_j = 0$.
- If $|\tilde{w}_j| > \lambda$: shrink toward zero by $\lambda$, but keep the sign.

```
Soft-thresholding function S(w, λ):

       ↗ w - λ        (w > λ)
S(w) = 0              (-λ ≤ w ≤ λ)
       ↘ w + λ        (w < -λ)
```

**Coordinate descent algorithm for Lasso:**
```
Initialise w = 0 (or OLS estimate)
Repeat until convergence:
    For j = 1, ..., d:
        Compute residual without feature j: r_j = y - X_{-j} w_{-j}
        Compute LS coefficient: w̃_j = x_j^T r_j / ||x_j||²
        Apply soft-threshold: w_j ← S(w̃_j, λ / ||x_j||²)
```

> **Notebook reference.** Exercise 1 (Cell 13) asks you to implement coordinate descent for Lasso. Compare your implementation to `Lasso` from sklearn.

---

### 4.4 Bayesian Interpretation: Laplace Prior

By the same MAP argument as Ridge:

$$\mathbf{w} \sim \text{Laplace}(\mathbf{0}, b) \implies p(w_j) = \frac{1}{2b}\exp\!\left(-\frac{|w_j|}{b}\right)$$

The negative log-prior is $\frac{1}{b}\sum_j |w_j| = \frac{1}{b}\|\mathbf{w}\|_1$, which is the L1 penalty. Thus, the Lasso MAP estimate corresponds to a **Laplace (double-exponential) prior** on the weights.

The Laplace distribution has **heavier tails** than the Gaussian and more mass at exactly zero. This makes the prior more "supportive" of sparse solutions — it concentrates probability around zero while still allowing large values for truly important features.

| Prior                             | Penalty                      | Effect on weights                         |
| --------------------------------- | ---------------------------- | ----------------------------------------- |
| Gaussian $\mathcal{N}(0, \tau^2)$ | $\|\mathbf{w}\|_2^2$ (Ridge) | Shrink all toward zero, none exactly zero |
| Laplace $\text{Lap}(0, b)$        | $\|\mathbf{w}\|_1$ (Lasso)   | Shrink and set some exactly to zero       |

---

### 4.5 Ridge vs. Lasso: Summary

| Property                | Ridge (L2)                                                   | Lasso (L1)                                       |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| **Penalty**             | $\lambda\sum_j w_j^2$                                        | $\lambda\sum_j \|w_j\|$                          |
| **Constraint shape**    | Ball (sphere)                                                | Diamond (cross-polytope)                         |
| **Sparsity**            | No (all weights non-zero)                                    | Yes (automatic feature selection)                |
| **Closed form**         | Yes: $(X^\top X + n\lambda I)^{-1}X^\top\mathbf{y}$          | No (coordinate descent, ADMM, etc.)              |
| **Bayesian prior**      | Gaussian                                                     | Laplace                                          |
| **Correlated features** | Distributes weight across correlated features                | Picks one, zeros the rest (unstable)             |
| **When $d > n$**        | Picks at most $n$ nonzero coefficients? No — all $d$ nonzero | At most $n$ non-zero coefficients                |
| **Best when**           | Many features contribute small amounts                       | Few features dominate; you want interpretability |

> **Notebook reference.** Cell 5 visualises the difference: Ridge keeps all 50 features non-zero; Lasso selects ~10 (matching the 10 truly informative features in the synthetic data).

---

## 5. Elastic Net

### 5.1 The Objective

Elastic Net (Zou & Hastie, 2005) combines L1 and L2 penalties:

$$\boxed{J_{\text{ElasticNet}}(\mathbf{w}) = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2 + \lambda\left[\rho\|\mathbf{w}\|_1 + \frac{1 - \rho}{2}\|\mathbf{w}\|_2^2\right]}$$

| Symbol    | Meaning                                            | Range             |
| --------- | -------------------------------------------------- | ----------------- |
| $\lambda$ | Overall regularisation strength                    | $\lambda > 0$     |
| $\rho$    | L1 ratio (mixing parameter, `l1_ratio` in sklearn) | $\rho \in [0, 1]$ |

- $\rho = 1$: pure Lasso.
- $\rho = 0$: pure Ridge.
- $0 < \rho < 1$: a blend of both.

**Constraint region geometry.** The Elastic Net constraint $\rho\|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2 / 2 \leq t$ is a **smoothed diamond** — it has the corners of the L1 ball (producing sparsity) but the curvature of the L2 ball (giving stable group behaviour). As $\rho$ decreases, the diamond becomes rounder; as $\rho$ increases, it becomes sharper.

---

### 5.2 When to Use Elastic Net

| Scenario                           | Best choice                   | Why                                                                                                      |
| ---------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| Few true features, many irrelevant | Lasso                         | Sparsity for feature selection                                                                           |
| Many correlated feature groups     | Elastic Net                   | L2 part shares weight across correlated features (grouping effect); L1 part still zeroes irrelevant ones |
| All features contribute            | Ridge                         | No feature selection needed                                                                              |
| You're unsure                      | Elastic Net with $\rho = 0.5$ | Safe default; tune $\rho$ by CV                                                                          |

> **The "grouping effect."** If features $j$ and $k$ are highly correlated ($\text{corr}(x_j, x_k) \approx 1$), Lasso arbitrarily selects one and zeros the other. This is unstable — the selected feature can flip between reruns. Elastic Net's L2 component ensures that correlated features receive similar weights (Zou & Hastie, 2005, Theorem 1).

> **Notebook reference.** Cell 11 fits Elastic Net at three L1 ratios (0.1, 0.5, 0.9) and shows the transition from Ridge-like behaviour (all non-zero) to Lasso-like behaviour (many zeros).
>
> **Suggested experiment.** Add two copies of an informative feature to the data (creating perfect correlation). Fit Lasso and Elastic Net. Lasso will select only one copy; Elastic Net will give both copies similar weights.

---

## 6. Cross-Validation

### 6.1 The Train/Validation/Test Protocol

Regularisation introduces a **hyperparameter** $\lambda$ that cannot be learned from the training data (setting $\lambda = 0$ always minimises training error). We need held-out data to choose $\lambda$.

| Split              | Purpose                                          | Proportion (rule of thumb) |
| ------------------ | ------------------------------------------------ | -------------------------- |
| **Training set**   | Fit model parameters ($\mathbf{w}$)              | 60–80%                     |
| **Validation set** | Select hyperparameters ($\lambda$, $K$, etc.)    | 10–20%                     |
| **Test set**       | Final, unbiased estimate of generalisation error | 10–20%                     |

> **The cardinal rule.** Never use the test set to make any decision. If you tune $\lambda$ by looking at test error, you are overfitting to the test set and your reported error is optimistic.

A single train/validation split wastes data and yields a high-variance estimate of validation error. **Cross-validation** fixes both problems.

---

### 6.2 K-Fold Cross-Validation

**Procedure:**

1. Randomly partition the $n$ training points into $K$ equal-sized **folds** $F_1, F_2, \ldots, F_K$.
2. For each fold $k = 1, \ldots, K$:
   - **Train** on all data except fold $k$: $\mathcal{D}_{\text{train}} = \bigcup_{j \neq k} F_j$.
   - **Evaluate** on fold $k$: compute error $e_k$.
3. The **CV score** is the average: $\text{CV}_K = \frac{1}{K}\sum_{k=1}^{K}e_k$.

**To select $\lambda$:** compute $\text{CV}_K$ for a grid of $\lambda$ values and choose $\lambda^* = \arg\min_\lambda \text{CV}_K(\lambda)$.

**Typical choices of $K$:**

| $K$ | Name                | Bias                                       | Variance                    | Cost     |
| --- | ------------------- | ------------------------------------------ | --------------------------- | -------- |
| 5   | 5-fold              | Small upward bias (trained on 80% of data) | Moderate                    | 5 fits   |
| 10  | 10-fold             | Slightly less bias (90%)                   | Slightly higher variance    | 10 fits  |
| $n$ | Leave-one-out (LOO) | Nearly unbiased (trained on $n-1$ points)  | High (folds are correlated) | $n$ fits |

$K = 5$ or $K = 10$ are the standard defaults; LOO is used mainly when $n$ is very small.

**Variance of CV estimate.** The standard error of the CV score is:

$$\text{SE} = \frac{1}{\sqrt{K}}\sqrt{\frac{1}{K-1}\sum_{k=1}^K(e_k - \bar{e})^2}$$

> **Notebook reference.** Cell 7 uses `RidgeCV(alphas=..., cv=5)` and `LassoCV(alphas=..., cv=5)` to find optimal $\lambda$ values, then plots validation curves (MSE vs. $\alpha$) for 50 alpha values.

---

### 6.3 Stratified K-Fold

For classification problems, standard K-fold can produce folds with very different class proportions (especially if classes are imbalanced). **Stratified K-fold** ensures each fold preserves the overall class distribution.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    # Each fold has approximately the same y distribution
    ...
```

> **Rule of thumb.** Always use `StratifiedKFold` for classification. Use regular `KFold` for regression.

---

### 6.4 Leave-One-Out Cross-Validation

LOO-CV sets $K = n$: each fold consists of a single data point. This is maximally unbiased (training on $n - 1$ points), but:
- **High variance:** folds differ by only one point, so the $n$ error estimates are highly correlated.
- **Expensive:** requires $n$ model fits (though for Ridge, there is a closed-form shortcut).

**Shortcut for Ridge (hat-matrix trick):**

$$\text{LOO-CV error} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2$$

where $h_{ii}$ is the $i$-th diagonal element of the hat matrix $H = X(X^\top X + n\lambda I)^{-1}X^\top$. This computes LOO-CV in $\mathcal{O}(nd^2)$ — the cost of a single fit — without actually refitting $n$ times.

---

### 6.5 Validation Curves and Learning Curves

Two diagnostic plots:

**Validation curve** (complexity curve):
- $x$-axis: hyperparameter (e.g., $\lambda$).
- $y$-axis: training score and validation score.
- **Interpretation:** large gap = overfitting. Both scores low = underfitting. Optimal $\lambda$ is where validation score is maximised (or validation error is minimised).

```python
from sklearn.model_selection import validation_curve
train_scores, val_scores = validation_curve(
    Ridge(), X, y, param_name='alpha', param_range=alphas,
    cv=5, scoring='neg_mean_squared_error')
```

**Learning curve:**
- $x$-axis: number of training samples.
- $y$-axis: training score and validation score.
- **Interpretation:** if the gap persists even at large $n$, the model is too complex (overfitting). If both scores plateau at a high error, the model is too simple (underfitting / more data won't help).

> **Notebook reference.** Cell 7 plots validation curves for both Ridge and Lasso, showing the U-shaped validation error (high at very low and very high $\alpha$, minimum at the optimal $\alpha$).
>
> **Suggested experiment.** Plot learning curves for Ridge at the optimal $\alpha$ and at $\alpha = 0$ (OLS). OLS will show a large train/val gap that does not close. Ridge will show a small gap that converges.

---

## 7. Time-Series Cross-Validation

### 7.1 Why Standard K-Fold Fails for Time Series

Standard K-fold randomly shuffles data across folds. For time-series data this is catastrophically wrong:

| Problem              | Consequence                                                                                                |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Temporal leakage** | Future data points appear in the training set when evaluating on past data                                 |
| **Autocorrelation**  | Nearby points are similar; random splits create artificially easy test sets                                |
| **Non-stationarity** | Data distribution shifts over time; a model trained on all time periods may not reflect future performance |

> **Example.** If you predict tomorrow's stock price using today's features, K-fold might put Wednesday's data in the training set while evaluating on Tuesday. The model has already "seen the future."

---

### 7.2 Walk-Forward Validation

**TimeSeriesSplit** (walk-forward validation) preserves temporal ordering:

For $K$ splits on $n$ data points:
- **Split 1:** train on $[1 : n_0]$, validate on $[n_0+1 : n_0 + s]$
- **Split 2:** train on $[1 : n_0 + s]$, validate on $[n_0 + s + 1 : n_0 + 2s]$
- **Split $k$:** train on $[1 : n_0 + (k-1)s]$, validate on $[n_0 + (k-1)s + 1 : n_0 + ks]$

Key properties:
- The training set **always precedes** the validation set.
- The training set **grows** with each split (expanding window).
- There is **no data leakage**.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=tscv,
                         scoring='neg_mean_squared_error')
```

> **Notebook reference.** Cell 9 implements walk-forward CV on synthetic time-series data and visualises the 5 splits. The training (blue) subset always comes before the test (red) subset.

**Variants:**
- **Expanding window:** training set grows (as above). Most common.
- **Sliding window:** training set has a fixed size and slides forward. Useful when the data is non-stationary and old data may be harmful.

---

### 7.3 Lag Features

Time-series regression typically uses **lag features**: predict $y_t$ from $y_{t-1}, y_{t-2}, \ldots, y_{t-L}$.

Given a time-series vector $\mathbf{y} = [y_1, y_2, \ldots, y_n]$ and $L$ lags, the feature matrix is:

$$X = \begin{bmatrix} y_L & y_{L-1} & \cdots & y_1 \\ y_{L+1} & y_L & \cdots & y_2 \\ \vdots & \vdots & \ddots & \vdots \\ y_{n-1} & y_{n-2} & \cdots & y_{n-L} \end{bmatrix}, \quad \mathbf{y}_{\text{target}} = \begin{bmatrix} y_{L+1} \\ y_{L+2} \\ \vdots \\ y_n \end{bmatrix}$$

The notebook implements this with `create_lag_features(y, n_lags=5)`, yielding a design matrix of shape $(n - L, L)$.

> **Forward pointer.** Lag-based models are the simplest form of sequence modelling. [Week 09](../../03_probability/week09_time_series/theory.md#6-autoregressive-models-arp) (Time Series) introduces autoregressive models, and [Weeks 17](../../06_sequence_models/week17_attention/theory.md#31-the-query-key-value-framework)–[18](../../06_sequence_models/week18_transformers/theory.md#41-sub-layer-1-multi-head-self-attention) cover attention and transformers — a fundamentally different (and far more powerful) approach to sequence data.

---

## 8. Early Stopping

### 8.1 Early Stopping as Implicit Regularisation

When training a model by gradient descent, the parameter vector traces a path from the initialisation $\mathbf{w}^{(0)}$ (typically $\mathbf{0}$ or random small values) toward the global minimum of the training loss. Along this path:

- **Early iterations:** the model fits the large-scale structure in the data (low complexity).
- **Later iterations:** the model fits the noise (high complexity, overfitting).

**Early stopping** halts training when the **validation loss** begins to increase, even though training loss continues to decrease. This is a form of regularisation because it limits how far $\mathbf{w}$ can travel from the initialisation.

**Procedure:**
1. Split data into train/validation/test.
2. Train the model by gradient descent.
3. After every $m$ iterations, evaluate on the validation set.
4. If validation loss has not improved for a **patience** period of $p$ evaluations, stop.
5. Return the model from the iteration with the lowest validation loss.

---

### 8.2 Equivalence to L2 Regularisation (Linear Case)

For linear regression with gradient descent (learning rate $\eta$, starting from $\mathbf{w}^{(0)} = \mathbf{0}$), early stopping after $T$ iterations is approximately equivalent to Ridge regression with:

$$\lambda_{\text{eff}} \approx \frac{1}{\eta T}$$

*Sketch of argument.* The GD update is $\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta\nabla\mathcal{L}$. For small $\eta$, this is an approximation to the ODE $\dot{\mathbf{w}} = -\nabla\mathcal{L}$. The solution at time $T$ in the eigenbasis of $X^\top X$ is:

$$w_j^{(T)} = \left(1 - (1 - \eta s_j^2/n)^T\right)\hat{w}_{j,\text{OLS}}$$

Compare with the Ridge solution: $w_j^{\text{Ridge}} = \frac{s_j^2}{s_j^2 + n\lambda}\hat{w}_{j,\text{OLS}}$. Both apply a shrinkage factor that suppresses small-$s_j$ directions, and the mapping $\lambda \leftrightarrow 1/(\eta T)$ makes the two approximately equal.

> **Practical implication.** Early stopping, learning rate, and L2 regularisation are all doing similar things: controlling how far the parameters move from zero. However, they are not _exactly_ interchangeable in practice (especially for non-linear models), so combining them is standard.

> **Notebook reference.** Exercise 3 (Cell 14) asks you to implement a GD loop with validation monitoring and early stopping. You should observe that stopping early yields test MSE comparable to Ridge at optimal $\lambda$.

---

## 9. The Regularisation Landscape: A Unified View

| Technique                          | What it constrains                        | Where it appears in the course                |
| ---------------------------------- | ----------------------------------------- | --------------------------------------------- |
| **L2 (Ridge)**                     | Weight magnitudes (squared)               | This week; all linear/logistic models         |
| **L1 (Lasso)**                     | Weight magnitudes (absolute)              | This week; sparse models                      |
| **Elastic Net**                    | Both L1 and L2                            | This week                                     |
| **Early stopping**                 | Number of GD iterations                   | This week; neural network training ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#9-regularisation-in-neural-networks)+) |
| **PCA (dimensionality reduction)** | Number of input dimensions                | [Week 04](../week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view)                                       |
| **Dropout**                        | Random neuron deactivation                | [Week 16](../../05_deep_learning/week16_regularization_dl/theory.md#3-dropout) (Regularisation for DL)               |
| **Batch Normalisation**            | Internal covariate shift                  | [Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#7-fix-2-batch-normalisation) (Training Pathologies)                |
| **Data augmentation**              | Effective training set size               | [Week 15](../../05_deep_learning/week15_cnn_representations/theory.md) (CNNs)                                |
| **Weight decay**                   | Same as L2, but for Adam/SGD              | [Week 02](../week02_advanced_optimizers/theory.md#81-adamw-decoupled-weight-decay), [Week 14](../../05_deep_learning/week14_training_at_scale/theory.md) (Training at Scale)          |
| **Layer freezing**                 | Prevents fine-tuning of pretrained layers | [Week 19](../../07_transfer_learning/week19_finetuning/theory.md#3-adaptation-strategies) (Fine-tuning)                         |

> **The fundamental principle.** All regularisation techniques reduce the effective complexity of the model, either explicitly (penalising the objective) or implicitly (early stopping, dropout, data augmentation). The right amount of regularisation makes the model capture the signal without memorising the noise.

---

## 10. Connections to the Rest of the Course

| Week                                | Connection                                                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **[Week 01](../week01_optimization/theory.md#73-conditioning-and-the-hessian) (Optimisation)**          | GD minimises the penalised objective; ridge adds $\lambda I$ to the Hessian (better conditioning)          |
| **[Week 02](../week02_advanced_optimizers/theory.md#81-adamw-decoupled-weight-decay) (Advanced Optimisers)**   | Weight decay in Adam is L2 regularisation; decoupled weight decay (AdamW) is subtly different              |
| **[Week 03](../week03_linear_models/theory.md#3-linear-regression) (Linear Models)**         | Ridge/Lasso are direct modifications of linear regression; bias–variance tradeoff motivates regularisation |
| **[Week 04](../week04_dimensionality_reduction/theory.md#34-ridge-and-svd-shrinkage-of-singular-values) (PCA)**                   | Ridge shrinks SVD components (Section 3.4); PCA truncation is an extreme form of shrinkage                 |
| **[Week 05](../week05_clustering/theory.md#6-gaussian-mixture-models-soft-clustering) (Clustering)**            | Regularised GMMs (adding $\epsilon I$ to covariance) prevent singular matrices — same idea as Ridge        |
| **[Week 07](../../03_probability/week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle) (Likelihood)**            | MAP estimation connects regularisation to Bayesian priors                                                  |
| **[Week 08](../../03_probability/week08_uncertainty/theory.md#7-bayesian-inference) (Uncertainty)**           | Full Bayesian inference goes beyond point estimation; regularisation is the MAP approximation              |
| **[Week 09](../../03_probability/week09_time_series/theory.md#10-forecast-evaluation) (Time Series)**           | Time-series CV is essential for any sequential prediction task                                             |
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#9-regularisation-in-neural-networks) (NNs)**                   | Weight decay (L2) is the standard regulariser for neural networks                                          |
| **[Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#7-fix-2-batch-normalisation) (Training Pathologies)**  | Batch normalisation has a regularising effect (mini-batch noise)                                           |
| **[Week 16](../../05_deep_learning/week16_regularization_dl/theory.md#3-dropout) (Regularisation for DL)** | Dropout, data augmentation, label smoothing — all regularisation techniques for deep learning              |
| **[Week 19](../../07_transfer_learning/week19_finetuning/theory.md#3-adaptation-strategies) (Fine-tuning)**           | Freezing pretrained layers = constraining parameters = regularisation                                      |

---

## 11. Notebook Reference Guide

| Cell                 | Section                           | What it demonstrates                                                                                          | Theory reference  |
| -------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------- |
| 5 (Ridge vs. Lasso)  | Ridge and Lasso on synthetic data | Ridge: all 50 weights non-zero. Lasso: ~10 non-zero (feature selection). Scatter plot shows the relationship. | Sections 3, 4     |
| 7 (Cross-Validation) | RidgeCV / LassoCV with 50 alphas  | Optimal alpha found by 5-fold CV. Validation curves show U-shaped MSE.                                        | Section 6.2, 6.5  |
| 9 (Time-Series CV)   | Walk-forward validation           | 5-split TimeSeriesSplit on lag features. Visualisation of expanding training windows.                         | Section 7         |
| 11 (Elastic Net)     | L1 ratios: 0.1, 0.5, 0.9          | Transition from Ridge-like (many non-zero) to Lasso-like (sparse).                                            | Section 5         |
| Exercise 1 (Cell 13) | Regularisation path               | Sweep alpha; plot coefficient paths. Ridge shrinks all; Lasso zeros them out progressively.                   | Sections 3.4, 4.2 |
| Exercise 3 (Cell 14) | Early stopping                    | GD with validation monitoring. Compare stopped model vs. Ridge.                                               | Section 8         |

**Suggested modifications:**

| Modification                                                  | What it reveals                                                            |
| ------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Set `alpha=0.0001` and `alpha=100.0` for Ridge                | Underfitting vs. overfitting extremes via the penalty                      |
| Increase `n_features` to 200 with `n_informative=10`          | Lasso becomes essential; OLS/Ridge overfit badly in the $d \gg n$ regime   |
| Use `make_regression(n_informative=50)` (all features useful) | Ridge outperforms Lasso (no sparsity to exploit)                           |
| Vary `cv` in `RidgeCV` (3, 5, 10, $n$)                        | See how CV fold count affects optimal alpha and its stability              |
| Add `noise=50` to increase irreducible noise                  | Optimal alpha increases (more regularisation needed)                       |
| Replace `Ridge` with `LinearRegression` in the time-series CV | OLS overfits on lag features; Ridge generalises better                     |
| Try `ElasticNetCV` with `l1_ratio=[0.1, 0.5, 0.9, 0.99]`      | Joint selection of alpha and l1_ratio; see which data regimes prefer which |

---

## 12. Symbol Reference

| Symbol                            | Name                         | Meaning                                                                     |
| --------------------------------- | ---------------------------- | --------------------------------------------------------------------------- |
| $X \in \mathbb{R}^{n \times d}$   | Design matrix                | $n$ samples, $d$ features                                                   |
| $\mathbf{w} \in \mathbb{R}^d$     | Weight vector                | Model parameters (excluding intercept)                                      |
| $b$                               | Intercept (bias)             | Not regularised                                                             |
| $\mathbf{y} \in \mathbb{R}^n$     | Target vector                | Response values                                                             |
| $\lambda$                         | Regularisation strength      | Hyperparameter (called `alpha` in sklearn)                                  |
| $\Omega(\boldsymbol{\theta})$     | Penalty function             | Measures model complexity                                                   |
| $\|\mathbf{w}\|_1$                | L1 norm                      | $\sum_j \|w_j\|$ (sum of absolute values)                                   |
| $\|\mathbf{w}\|_2^2$              | Squared L2 norm              | $\sum_j w_j^2$ (sum of squared values)                                      |
| $J_{\text{Ridge}}$                | Ridge objective              | MSE + $\lambda\|\mathbf{w}\|_2^2$                                           |
| $J_{\text{Lasso}}$                | Lasso objective              | MSE + $\lambda\|\mathbf{w}\|_1$                                             |
| $\rho$                            | L1 ratio (Elastic Net)       | Mixing parameter in $[0, 1]$; `l1_ratio` in sklearn                         |
| $s_j$                             | Singular value               | $j$-th singular value of $X$                                                |
| $f_j$                             | Shrinkage factor (Ridge)     | $s_j^2 / (s_j^2 + n\lambda)$                                                |
| $\text{df}(\lambda)$              | Effective degrees of freedom | $\sum_j f_j$ for Ridge                                                      |
| $\mathcal{S}(\tilde{w}, \lambda)$ | Soft-thresholding operator   | $\text{sign}(\tilde{w})\max(\|\tilde{w}\| - \lambda, 0)$                    |
| $K$                               | Number of CV folds           | Typically 5 or 10                                                           |
| $F_k$                             | Fold $k$                     | Subset of training data for cross-validation                                |
| $\text{CV}_K$                     | K-fold CV score              | Average validation error across $K$ folds                                   |
| $h_{ii}$                          | Leverage                     | Diagonal of hat matrix; used in LOO shortcut                                |
| $H$                               | Hat matrix                   | $X(X^\top X + n\lambda I)^{-1}X^\top$                                       |
| $\tau^2$                          | Prior variance (Bayesian)    | Width of the Gaussian prior on $\mathbf{w}$; $\lambda = \sigma^2/(n\tau^2)$ |
| $T$                               | Number of GD iterations      | Controls early stopping                                                     |
| $\lambda_{\text{eff}}$            | Effective regularisation     | $\approx 1/(\eta T)$ for early stopping                                     |
| $L$                               | Number of lags               | Time-series feature engineering                                             |
| $\eta$                            | Learning rate                | Step size in gradient descent                                               |

---

## 13. References

1. Hoerl, A. E. & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, 12(1), 55–67. — The original Ridge regression paper.
2. Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *JRSS Series B*, 58(1), 267–288. — The original Lasso paper.
3. Zou, H. & Hastie, T. (2005). "Regularization and Variable Selection via the Elastic Net." *JRSS Series B*, 67(2), 301–320. — Elastic Net; grouping effect theorem.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Chapters 3 and 7. Springer. — Comprehensive treatment of regularisation, shrinkage, and cross-validation.
5. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Chapter 6. Springer. — Accessible introduction to Ridge, Lasso, and cross-validation.
6. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 3. Springer. — Bayesian linear regression; MAP estimation as regularisation.
7. Friedman, J., Hastie, T., & Tibshirani, R. (2010). "Regularization Paths for Generalized Linear Models via Coordinate Descent." *Journal of Statistical Software*, 33(1), 1–22. — The `glmnet` algorithm for fast Lasso/Elastic Net.
8. Stone, M. (1974). "Cross-Validatory Choice and Assessment of Statistical Predictions." *JRSS Series B*, 36(2), 111–147. — Theoretical foundations of cross-validation.
9. Arlot, S. & Celisse, A. (2010). "A Survey of Cross-Validation Procedures for Model Selection." *Statistics Surveys*, 4, 40–79. — Comprehensive review of CV methods.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 7: Regularization. MIT Press. — Early stopping, weight decay, dropout, and other regularisation techniques for deep models.
11. Scikit-learn User Guide: Linear Models. [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html).
12. Scikit-learn User Guide: Cross-validation. [https://scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html).
