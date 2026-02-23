# Classical ML Foundations: Linear and Logistic Regression

## Table of Contents

- [Classical ML Foundations: Linear and Logistic Regression](#classical-ml-foundations-linear-and-logistic-regression)
  - [Table of Contents](#table-of-contents)
  - [1. Scope and Purpose](#1-scope-and-purpose)
  - [2. The Supervised Learning Setup](#2-the-supervised-learning-setup)
  - [3. Linear Regression](#3-linear-regression)
    - [3.1 The Model](#31-the-model)
    - [3.2 The Loss Function: Mean Squared Error](#32-the-loss-function-mean-squared-error)
    - [3.3 The Normal Equations (Closed-Form Solution)](#33-the-normal-equations-closed-form-solution)
    - [The Pseudoinverse](#the-pseudoinverse)
    - [3.4 Geometric Interpretation](#34-geometric-interpretation)
    - [3.5 Gradient Descent for Linear Regression](#35-gradient-descent-for-linear-regression)
    - [3.6 Closed-Form vs. Gradient Descent](#36-closed-form-vs-gradient-descent)
  - [4. Statistical Interpretation of Linear Regression](#4-statistical-interpretation-of-linear-regression)
    - [4.1 The Gaussian Noise Model](#41-the-gaussian-noise-model)
    - [4.2 Maximum Likelihood Estimation](#42-maximum-likelihood-estimation)
    - [4.3 Properties of the OLS Estimator](#43-properties-of-the-ols-estimator)
  - [5. Polynomial Regression and Feature Expansion](#5-polynomial-regression-and-feature-expansion)
  - [6. The Bias–Variance Tradeoff](#6-the-biasvariance-tradeoff)
    - [6.1 Decomposing Prediction Error](#61-decomposing-prediction-error)
    - [6.2 Formal Derivation](#62-formal-derivation)
    - [6.3 The Complexity Curve](#63-the-complexity-curve)
    - [6.4 Detecting Bias and Variance from Learning Curves](#64-detecting-bias-and-variance-from-learning-curves)
  - [7. Logistic Regression](#7-logistic-regression)
    - [7.1 From Regression to Classification](#71-from-regression-to-classification)
    - [7.2 The Sigmoid (Logistic) Function](#72-the-sigmoid-logistic-function)
    - [7.3 The Model](#73-the-model)
    - [7.4 The Loss Function: Binary Cross-Entropy](#74-the-loss-function-binary-cross-entropy)
    - [7.5 Gradient of the Cross-Entropy Loss](#75-gradient-of-the-cross-entropy-loss)
    - [7.6 Decision Boundary](#76-decision-boundary)
    - [7.7 Multi-Class Extension: Softmax Regression](#77-multi-class-extension-softmax-regression)
  - [8. Evaluation Metrics](#8-evaluation-metrics)
    - [8.1 Regression Metrics](#81-regression-metrics)
    - [8.2 Classification Metrics](#82-classification-metrics)
  - [9. Regularisation Preview](#9-regularisation-preview)
  - [10. Connections to the Rest of the Course](#10-connections-to-the-rest-of-the-course)
  - [11. Notebook Reference Guide](#11-notebook-reference-guide)
  - [12. Symbol Reference](#12-symbol-reference)
  - [13. References](#13-references)

---

## 1. Scope and Purpose

[Weeks 01](../week01_optimization/theory.md#4-gradient-descent)–[02](../week02_advanced_optimizers/theory.md) developed the **optimisation machinery** — gradient descent, momentum, Adam — as general-purpose tools for minimising any differentiable function. This week we apply those tools (and a powerful alternative: the closed-form normal equations) to the two most fundamental supervised learning models:

- **Linear regression** (continuous outputs): the foundation of all regression methods.
- **Logistic regression** (discrete outputs): the foundation of all classification methods.

These are not "toy" models. A neural network is a composition of linear transformations and nonlinearities; understanding the linear case deeply is prerequisite to understanding the nonlinear case. Furthermore, the **bias–variance tradeoff** introduced here is the central conceptual tool for diagnosing any model's performance.

**Prerequisites.** [Week 00b](../../01_intro/week00b_math_and_data/theory.md) (linear algebra, calculus, probability basics), [Weeks 01](../week01_optimization/theory.md#4-gradient-descent)–[02](../week02_advanced_optimizers/theory.md) (gradient descent, optimisers, learning rate schedules).

---

## 2. The Supervised Learning Setup

Recall from [Week 00a](../../01_intro/week00_ai_landscape/theory.md#4-the-supervised-learning-framework):

| Symbol                                            | Name                 | Meaning                                                |
| ------------------------------------------------- | -------------------- | ------------------------------------------------------ |
| $\mathbf{x} \in \mathbb{R}^d$                     | Feature vector       | $d$ measurements describing one example                |
| $y$                                               | Target (label)       | The quantity or class we want to predict               |
| $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ | Training set         | $n$ labelled examples                                  |
| $f(\mathbf{x}; \theta)$                           | Model (hypothesis)   | Parameterised function mapping features to predictions |
| $\theta$                                          | Parameters (weights) | The learnable quantities                               |
| $\mathcal{L}(\theta)$                             | Loss function        | Measures how badly $f$ fits the data                   |

The goal of supervised learning is to find $\theta^*$ that minimises the expected loss on unseen data (generalisation), not just on the training set. The gap between training performance and generalisation performance is the subject of the bias–variance tradeoff (Section 6).

---

## 3. Linear Regression

### 3.1 The Model

A linear regression model assumes the target $y$ is a linear function of the features:

$$f(\mathbf{x}; \mathbf{w}, b) = \mathbf{w}^\top \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

| Symbol                        | Meaning                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| $\mathbf{w} \in \mathbb{R}^d$ | Weight vector — one weight per feature                       |
| $b \in \mathbb{R}$            | Bias (intercept) — the prediction when all features are zero |
| $d$                           | Number of features (dimension of $\mathbf{x}$)               |

**Absorbing the bias.** It is conventional to prepend a constant 1 to every feature vector: $\tilde{\mathbf{x}} = [1, x_1, \ldots, x_d]^\top \in \mathbb{R}^{d+1}$, and define $\tilde{\mathbf{w}} = [b, w_1, \ldots, w_d]^\top$. Then:

$$f(\mathbf{x}) = \tilde{\mathbf{w}}^\top \tilde{\mathbf{x}}$$

This eliminates the need for a separate bias term. From here on, we write $\mathbf{w}$ and $\mathbf{x}$ to mean the augmented versions (bias absorbed), unless stated otherwise.

**Matrix form for the entire dataset.** Stacking all $n$ examples into a matrix $X \in \mathbb{R}^{n \times (d+1)}$ (one row per example, first column all ones):

$$X = \begin{bmatrix} 1 & x_{1,1} & \cdots & x_{1,d} \\ 1 & x_{2,1} & \cdots & x_{2,d} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n,1} & \cdots & x_{n,d} \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

The vector of all predictions is:

$$\hat{\mathbf{y}} = X\mathbf{w}$$

This single matrix multiplication replaces $n$ dot products. In NumPy: `y_hat = X @ w`.

> **Notebook reference.** In `starter.ipynb` Cell 5, $X$ is constructed as `X_with_bias = np.hstack([np.ones((n_samples, 1)), X])` and the prediction is `y_pred = X_line_bias @ w`.

---

### 3.2 The Loss Function: Mean Squared Error

The standard loss for regression is the **Mean Squared Error (MSE)**:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \mathbf{w}^\top \mathbf{x}_i\right)^2 = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2$$

where $\|\cdot\|$ is the Euclidean ($\ell_2$) norm.

**Terminology note.** The quantity $r_i = y_i - \hat{y}_i = y_i - \mathbf{w}^\top \mathbf{x}_i$ is the **residual** for example $i$. The MSE is the mean of squared residuals.

Some formulations use $\frac{1}{2n}$ instead of $\frac{1}{n}$ (to simplify the gradient). The factor only changes the effective learning rate and does not change the optimal $\mathbf{w}$. We use $\frac{1}{n}$ here and note when the factor differs.

> **Why squared error?** Three complementary justifications:
> 1. **Algebraic convenience.** It is smooth, differentiable, and convex — guaranteed to have a unique minimum.
> 2. **Statistical.** It corresponds to maximum likelihood estimation under the assumption of Gaussian noise (Section 4.2).
> 3. **Geometric.** Minimising MSE is equivalent to finding the orthogonal projection of $\mathbf{y}$ onto the column space of $X$ (Section 3.4).

---

### 3.3 The Normal Equations (Closed-Form Solution)

Unlike the general optimisation problems in [Weeks 01](../week01_optimization/theory.md#4-gradient-descent)–[02](../week02_advanced_optimizers/theory.md), linear regression has a **closed-form solution**: we can compute $\mathbf{w}^*$ directly, without iterating.

**Derivation.** Write the MSE in matrix form (dropping the $\frac{1}{n}$ factor, which does not affect the argmin):

$$\mathcal{L}(\mathbf{w}) = (\mathbf{y} - X\mathbf{w})^\top(\mathbf{y} - X\mathbf{w})$$

Expand:

$$\mathcal{L}(\mathbf{w}) = \mathbf{y}^\top\mathbf{y} - 2\mathbf{y}^\top X\mathbf{w} + \mathbf{w}^\top X^\top X \mathbf{w}$$

Take the gradient with respect to $\mathbf{w}$ and set it to zero. Recall the matrix calculus identities from [Week 00b](../../01_intro/week00b_math_and_data/theory.md#37-vector-calculus-identities-for-ml):

$$\frac{\partial}{\partial \mathbf{w}}\left(\mathbf{a}^\top \mathbf{w}\right) = \mathbf{a}, \qquad \frac{\partial}{\partial \mathbf{w}}\left(\mathbf{w}^\top A \mathbf{w}\right) = 2A\mathbf{w} \quad (\text{if } A \text{ is symmetric})$$

Applying these:

$$\nabla_{\mathbf{w}} \mathcal{L} = -2X^\top\mathbf{y} + 2X^\top X \mathbf{w} = \mathbf{0}$$

Solving for $\mathbf{w}$:

$$\boxed{X^\top X \, \mathbf{w}^* = X^\top \mathbf{y}}$$

These are the **normal equations**. If $X^\top X$ is invertible (which requires that $X$ has full column rank, i.e., no column is a linear combination of the others):

$$\boxed{\mathbf{w}^* = (X^\top X)^{-1} X^\top \mathbf{y}}$$

| Component           | Size                     | Name                       | Meaning                                     |
| ------------------- | ------------------------ | -------------------------- | ------------------------------------------- |
| $X^\top X$          | $(d{+}1) \times (d{+}1)$ | Gram matrix                | Captures feature correlations               |
| $X^\top \mathbf{y}$ | $(d{+}1) \times 1$       | Feature-target correlation | How each feature correlates with the target |
| $(X^\top X)^{-1}$   | $(d{+}1) \times (d{+}1)$ | Inverse Gram               | Decorrelates the features                   |
| $\mathbf{w}^*$      | $(d{+}1) \times 1$       | Optimal weights            | The unique MSE-minimising weights           |

> **Implementation note.** Never compute $(X^\top X)^{-1}$ explicitly. Instead, use `np.linalg.solve(XTX, XTy)`, which solves the linear system $X^\top X \mathbf{w} = X^\top \mathbf{y}$ using a numerically stable factorisation (Cholesky or LU). Computing the inverse is approximately $3\times$ slower and less numerically stable.

> **Notebook reference.** Cell 5 implements exactly this:
> ```python
> def normal_equations(X, y):
>     XTX = X.T @ X
>     XTy = X.T @ y
>     w = np.linalg.solve(XTX, XTy)
>     return w
> ```
> Cell 7 verifies that this matches `sklearn.linear_model.LinearRegression`.

**When $X^\top X$ is singular.** If $n < d + 1$ (more parameters than data points) or if features are collinear, $X^\top X$ is singular and the normal equations have infinitely many solutions. This is where **regularisation** ([Week 06](../week06_regularization/theory.md#3-ridge-regression-l2-regularisation)) becomes essential: Ridge regression adds $\lambda I$ to $X^\top X$, guaranteeing invertibility.

### The Pseudoinverse

A more general solution uses the **Moore-Penrose pseudoinverse** $X^+$:

$$\mathbf{w}^* = X^+ \mathbf{y}$$

If $X$ has full column rank, $X^+ = (X^\top X)^{-1}X^\top$ and the pseudoinverse reduces to the normal equations. If $X$ is rank-deficient, the pseudoinverse selects the **minimum-norm** solution from the infinite set. In NumPy: `np.linalg.pinv(X) @ y`.

---

### 3.4 Geometric Interpretation

The normal equations have a beautiful geometric interpretation.

The prediction vector $\hat{\mathbf{y}} = X\mathbf{w}$ lies in the **column space** of $X$, denoted $\text{Col}(X) \subset \mathbb{R}^n$. This is the set of all vectors that can be written as a linear combination of the columns of $X$.

The OLS solution finds the $\hat{\mathbf{y}}$ in $\text{Col}(X)$ that is **closest** to $\mathbf{y}$ in Euclidean distance. This is the **orthogonal projection** of $\mathbf{y}$ onto $\text{Col}(X)$.

The **residual vector** $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to every column of $X$:

$$X^\top(\mathbf{y} - X\mathbf{w}^*) = \mathbf{0}$$

This is precisely the normal equations! The name "normal" comes from the fact that the residual is **normal** (perpendicular) to the column space.

The **projection matrix** (hat matrix) is:

$$H = X(X^\top X)^{-1}X^\top$$

such that $\hat{\mathbf{y}} = H\mathbf{y}$.

| Object                                       | Dimension    | Lives in                                                             |
| -------------------------------------------- | ------------ | -------------------------------------------------------------------- |
| $\mathbf{y}$                                 | $n \times 1$ | $\mathbb{R}^n$ (ambient space)                                       |
| $\hat{\mathbf{y}} = H\mathbf{y}$             | $n \times 1$ | $\text{Col}(X)$ (a $(d{+}1)$-dimensional subspace of $\mathbb{R}^n$) |
| $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$ | $n \times 1$ | $\text{Col}(X)^\perp$ (orthogonal complement)                        |

> **Intuition (2D analogy).** Imagine $\mathbf{y}$ as a point in 3D space and $\text{Col}(X)$ as a 2D plane through the origin. The projection $\hat{\mathbf{y}}$ is the shadow cast by $\mathbf{y}$ onto the plane when the light shines perpendicular to the plane. The residual $\mathbf{r}$ is the vertical line from the shadow to the point — it is perpendicular to the plane, and its length $\|\mathbf{r}\|$ is the RMSE times $\sqrt{n}$.

---

### 3.5 Gradient Descent for Linear Regression

Although the normal equations give the exact solution, gradient descent is important because:
1. It scales to massive datasets where $X^\top X$ cannot be formed (see Section 3.6).
2. It introduces the iterative pattern used for every subsequent model in the course.

**The gradient.** Starting from $\mathcal{L}(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2$:

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{2}{n} X^\top(X\mathbf{w} - \mathbf{y})$$

**The update:**

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \frac{2}{n} X^\top(X\mathbf{w}_t - \mathbf{y})$$

**Convergence.** Since the MSE loss for linear regression is a convex quadratic, the convergence analysis from [Week 01](../week01_optimization/theory.md#43-convergence-analysis-for-quadratic-losses) applies directly. The Hessian is:

$$H = \frac{2}{n} X^\top X$$

The eigenvalues of $H$ are $\frac{2}{n}\lambda_j$, where $\lambda_j$ are the eigenvalues of $X^\top X$. The condition number of the Hessian equals the condition number of $X^\top X$:

$$\kappa = \frac{\lambda_{\max}(X^\top X)}{\lambda_{\min}(X^\top X)}$$

The maximum stable learning rate is:

$$\eta < \frac{n}{\lambda_{\max}(X^\top X)}$$

And the convergence rate per step is $\rho = \frac{\kappa - 1}{\kappa + 1}$ (from [Week 01](../week01_optimization/theory.md#43-convergence-analysis-for-quadratic-losses), Section 4.3).

> **Notebook reference.** Exercise 2 (Cell 14) asks you to implement gradient descent for linear regression and verify it converges to the same solution as the normal equations.
>
> **Suggested experiment.** Run GD on the synthetic data with learning rates $\eta \in \{0.001, 0.01, 0.05, 0.1\}$. Plot the loss curve for each. Use the condition number of $X^\top X$ (via `np.linalg.cond(XTX)`) to predict the maximum stable $\eta$ and verify experimentally.

---

### 3.6 Closed-Form vs. Gradient Descent

| Property                                        | Normal equations                              | Gradient descent                                     |
| ----------------------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| **Exact solution**                              | Yes (in exact arithmetic)                     | No (converges asymptotically)                        |
| **Time complexity**                             | $\mathcal{O}(nd^2 + d^3)$                     | $\mathcal{O}(ndT)$, $T$ = number of steps            |
| **Memory**                                      | $\mathcal{O}(d^2)$ for $X^\top X$             | $\mathcal{O}(d)$ for the gradient                    |
| **When $n \gg d$ (many samples, few features)** | Fast — $d$ is small, so $d^3$ is cheap        | Can use SGD with mini-batches                        |
| **When $d \gg n$ (few samples, many features)** | $X^\top X$ is singular — needs regularisation | Can use SGD; regularisation recommended              |
| **When $n$ and $d$ are both very large**        | $X^\top X$ may not fit in memory              | SGD + mini-batches — only $\mathcal{O}(Bd)$ per step |
| **Generalises to non-quadratic losses**         | No — specific to MSE on linear models         | Yes — works for any differentiable loss              |

**Rule of thumb:** use the normal equations when $d \lesssim 10{,}000$. Use SGD when $d$ or $n$ is very large or when using a non-quadratic loss (e.g., logistic regression).

> **Why this matters for the course.** From [Week 04](../week04_dimensionality_reduction/theory.md) onwards, every model uses gradient-based optimisation. The linear regression closed-form solution is the **only** case in the course where the exact answer can be computed without iteration. Appreciate this simplicity — it never returns.

---

## 4. Statistical Interpretation of Linear Regression

### 4.1 The Gaussian Noise Model

Linear regression can be derived from a **probabilistic generative model**. Assume each target $y_i$ is generated by:

$$y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \qquad \varepsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$

| Symbol                     | Meaning                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| $\varepsilon_i$            | Random noise for example $i$                                       |
| $\mathcal{N}(0, \sigma^2)$ | Gaussian (normal) distribution with mean 0 and variance $\sigma^2$ |
| i.i.d.                     | Independent and identically distributed                            |
| $\sigma^2$                 | Noise variance (irreducible error)                                 |

This says: the data lies on a hyperplane $\mathbf{w}^\top \mathbf{x}$, perturbed by isotropic Gaussian noise.

Equivalently:

$$y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^\top \mathbf{x}_i, \sigma^2)$$

The conditional distribution of $y_i$ given $\mathbf{x}_i$ is Gaussian with mean $\mathbf{w}^\top \mathbf{x}_i$ and variance $\sigma^2$.

---

### 4.2 Maximum Likelihood Estimation

Given the Gaussian noise model, we can find $\mathbf{w}$ by **maximum likelihood estimation** (MLE) — choosing the $\mathbf{w}$ that makes the observed data most probable.

The **likelihood** of the dataset is:

$$p(\mathbf{y} \mid X, \mathbf{w}, \sigma^2) = \prod_{i=1}^{n} p(y_i \mid \mathbf{x}_i, \mathbf{w}, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^\top\mathbf{x}_i)^2}{2\sigma^2}\right)$$

The **log-likelihood** is:

$$\ell(\mathbf{w}) = \log p(\mathbf{y} \mid X, \mathbf{w}, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{w}^\top\mathbf{x}_i)^2$$

Maximising $\ell(\mathbf{w})$ with respect to $\mathbf{w}$ is equivalent to minimising:

$$\sum_{i=1}^{n}(y_i - \mathbf{w}^\top\mathbf{x}_i)^2$$

which is the **sum of squared errors** — exactly the objective of OLS regression.

> **Key insight.** Minimising MSE is equivalent to maximum likelihood estimation under the assumption of Gaussian noise. The $\sigma^2$ term disappears from the optimisation because it does not depend on $\mathbf{w}$. This connection will be deepened in [Week 07](../../03_probability/week07_likelihood/theory.md#51-gaussian-noise-mse) (Likelihood).

**Estimating the noise variance.** After finding $\mathbf{w}^*$, the MLE of $\sigma^2$ is:

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{w}^{*\top}\mathbf{x}_i)^2 = \frac{1}{n}\|\mathbf{r}\|^2$$

(The unbiased estimator divides by $n - d - 1$ instead of $n$, accounting for the $d + 1$ parameters estimated.)

---

### 4.3 Properties of the OLS Estimator

Under the Gaussian noise model, the OLS estimator $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top\mathbf{y}$ has well-studied properties:

**1. Unbiasedness.** $\mathbb{E}[\hat{\mathbf{w}}] = \mathbf{w}_{\text{true}}$.

*Proof:*

$$\mathbb{E}[\hat{\mathbf{w}}] = (X^\top X)^{-1}X^\top \mathbb{E}[\mathbf{y}] = (X^\top X)^{-1}X^\top X\mathbf{w}_{\text{true}} = \mathbf{w}_{\text{true}}$$

**2. Covariance.**

$$\text{Cov}(\hat{\mathbf{w}}) = \sigma^2(X^\top X)^{-1}$$

The uncertainty in the estimated weights is proportional to the noise variance $\sigma^2$ and inversely related to the "information content" of the data (captured by $X^\top X$).

**3. Gauss-Markov Theorem.** Among all **linear unbiased estimators** of $\mathbf{w}$, the OLS estimator has the **smallest variance** (in the matrix sense: any other linear unbiased estimator has $\text{Cov}(\tilde{\mathbf{w}}) - \text{Cov}(\hat{\mathbf{w}}) \succeq 0$).

> **What this means.** If you restrict yourself to (i) linear functions of $\mathbf{y}$ that (ii) are unbiased, OLS is the best you can do. This is a strong result, but note the caveats: biased estimators (like Ridge regression, [Week 06](../week06_regularization/theory.md#3-ridge-regression-l2-regularisation)) can have **lower MSE** by trading bias for a large reduction in variance.

**4. Normality.** If $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$, then $\hat{\mathbf{w}} \sim \mathcal{N}(\mathbf{w}_{\text{true}}, \sigma^2(X^\top X)^{-1})$. This enables confidence intervals and hypothesis tests on individual weights.

---

## 5. Polynomial Regression and Feature Expansion

Linear regression models a **linear-in-parameters** function: $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$. This may seem limiting, but we can model nonlinear relationships by constructing **nonlinear features** of $\mathbf{x}$.

**Polynomial features.** For a single input $x$, define the feature map:

$$\phi(x) = [1, x, x^2, x^3, \ldots, x^p]^\top \in \mathbb{R}^{p+1}$$

The model becomes:

$$f(x) = \mathbf{w}^\top \phi(x) = w_0 + w_1 x + w_2 x^2 + \cdots + w_p x^p$$

This is a **degree-$p$ polynomial** in $x$, but it is **linear in the parameters** $\mathbf{w}$. All the theory above (normal equations, gradient, statistical properties) applies unchanged — we simply replace $X$ with the transformed design matrix $\Phi$.

Constructing $\Phi$ in code:
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=p)
X_poly = poly.fit_transform(X)  # adds columns for x^2, x^3, ..., x^p
```

**The key idea.** The model is nonlinear in $x$ but linear in $\mathbf{w}$. We get nonlinear decision functions for free, using the same linear regression machinery, by designing appropriate features.

> **Generalisation.** Any set of basis functions $\{\phi_1(\mathbf{x}), \phi_2(\mathbf{x}), \ldots, \phi_p(\mathbf{x})\}$ can be used: polynomials, radial basis functions, Fourier features, etc. The model class is:
>
> $$f(\mathbf{x}) = \sum_{j=1}^{p} w_j \phi_j(\mathbf{x})$$
>
> This is the **basis function regression** framework (Bishop, Chapter 3).

> **Notebook reference.** Cell 11 uses `PolynomialFeatures(degree=degree)` with degrees $\{1, 2, 3, 5, 10, 15\}$ on a sine-wave dataset. The target function $y = \sin(2\pi x) + \varepsilon$ is nonlinear, and the experiment shows how increasing the polynomial degree affects fit quality.

---

## 6. The Bias–Variance Tradeoff

This is arguably the most important conceptual framework in all of machine learning. It explains **why** simple models underfit (high bias) and complex models overfit (high variance), and **why** there exists an optimal level of complexity.

### 6.1 Decomposing Prediction Error

Suppose the true data-generating process is:

$$y = f_{\text{true}}(\mathbf{x}) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

We train a model $\hat{f}(\mathbf{x})$ on a dataset $\mathcal{D}$. The model is a random variable (it depends on which dataset we happened to draw). For a new test point $\mathbf{x}_0$, the **expected prediction error** (over random datasets and noise) decomposes as:

$$\boxed{\mathbb{E}\left[\left(y_0 - \hat{f}(\mathbf{x}_0)\right)^2\right] = \underbrace{\sigma^2}_{\text{irreducible noise}} + \underbrace{\left(\text{Bias}\left[\hat{f}(\mathbf{x}_0)\right]\right)^2}_{\text{bias}^2} + \underbrace{\text{Var}\left[\hat{f}(\mathbf{x}_0)\right]}_{\text{variance}}}$$

| Term                             | Definition                                                                                                                | Source                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Irreducible noise** $\sigma^2$ | $\text{Var}[\varepsilon]$                                                                                                 | Noise in the data; no model can reduce this                                   |
| **Bias²**                        | $\left(\mathbb{E}_\mathcal{D}[\hat{f}(\mathbf{x}_0)] - f_{\text{true}}(\mathbf{x}_0)\right)^2$                            | How far the *average* model is from the truth; reflects modelling assumptions |
| **Variance**                     | $\mathbb{E}_\mathcal{D}\left[\left(\hat{f}(\mathbf{x}_0) - \mathbb{E}_\mathcal{D}[\hat{f}(\mathbf{x}_0)]\right)^2\right]$ | How much the model *fluctuates* across different datasets                     |

> **Intuition (archery analogy).** Imagine shooting arrows at a target:
> - **Bias** = how far the centre of your arrow cluster is from the bullseye (systematic error).
> - **Variance** = how spread out your arrows are around their centre (random error).
> - **Irreducible noise** = the target wobbles unpredictably (nothing you can do).
>
> A simple model (e.g., degree-1 polynomial) is like always shooting in approximately the same direction (low variance), but if the target isn't straight ahead, you consistently miss (high bias). A complex model (e.g., degree-15 polynomial) aims precisely at each training point but jitters wildly (high variance, low bias).

---

### 6.2 Formal Derivation

Let $f_0 = f_{\text{true}}(\mathbf{x}_0)$, let $\hat{f} = \hat{f}(\mathbf{x}_0)$, and let $\bar{f} = \mathbb{E}_\mathcal{D}[\hat{f}]$.

$$\mathbb{E}[(y_0 - \hat{f})^2] = \mathbb{E}[(f_0 + \varepsilon - \hat{f})^2]$$

Expand by adding and subtracting $\bar{f}$:

$$= \mathbb{E}[(f_0 - \bar{f})^2 + (\bar{f} - \hat{f})^2 + \varepsilon^2 + 2(f_0 - \bar{f})(\bar{f} - \hat{f}) + 2\varepsilon(f_0 - \hat{f})]$$

**Cross-term 1:** $\mathbb{E}[(\bar{f} - \hat{f})] = \bar{f} - \mathbb{E}[\hat{f}] = 0$, so the first cross-term vanishes.

**Cross-term 2:** $\mathbb{E}[\varepsilon(\cdot)] = 0$ because $\varepsilon$ is independent of $\hat{f}$ and has zero mean.

What remains:

$$= \underbrace{(f_0 - \bar{f})^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f} - \bar{f})^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

$\blacksquare$

---

### 6.3 The Complexity Curve

As model complexity (e.g., polynomial degree $p$) increases:

| Complexity                       | Training error | Test error | Bias     | Variance  | Regime           |
| -------------------------------- | -------------- | ---------- | -------- | --------- | ---------------- |
| Very low ($p = 1$)               | High           | High       | High     | Low       | **Underfitting** |
| Optimal ($p \approx 3\text{–}5$) | Moderate       | Lowest     | Moderate | Moderate  | **Sweet spot**   |
| Very high ($p = 15$)             | Near zero      | Very high  | Very low | Very high | **Overfitting**  |

This produces the characteristic **U-shaped test error curve**: training error decreases monotonically with complexity, but test error first decreases (bias reduction dominates) then increases (variance increase dominates).

> **Notebook reference.** Cell 11 plots exactly this curve. The polynomial degrees tested are $\{1, 2, 3, 5, 10, 15\}$. You should observe:
> - Degree 1: high training and test error (underfitting — a line cannot fit a sine wave).
> - Degree 3–5: low test error (good fit).
> - Degree 10–15: near-zero training error but high test error (overfitting — the polynomial memorises the training noise).
>
> **Suggested experiment.** Repeat the bias–variance experiment with $n \in \{15, 30, 100, 500\}$ data points. For each $n$, find the optimal degree. Observe that with more data, the optimal degree increases — more data supports more complex models because the variance decreases at fixed complexity.

---

### 6.4 Detecting Bias and Variance from Learning Curves

In practice, you cannot compute bias and variance directly (you would need many independent training sets). Instead, you diagnose via **learning curves**:

| Observation                                             | Diagnosis                       | Action                                       |
| ------------------------------------------------------- | ------------------------------- | -------------------------------------------- |
| Training error ≈ test error, **both high**              | **High bias** (underfitting)    | Increase model complexity; add features      |
| Training error **low**, test error **high** (large gap) | **High variance** (overfitting) | Reduce complexity; regularise; get more data |
| Training error ≈ test error, **both low**               | **Good fit**                    | Done (or fine-tune)                          |

**Plotting learning curves.** Plot training and validation error as a function of **training set size** $n$:
- High bias: both curves converge to a high error plateau — more data does not help.
- High variance: large gap between curves; the gap shrinks slowly as $n$ increases — more data helps.

> **This diagnostic framework is the most practical skill in this week.** Every model you build for the rest of the course should be evaluated using this lens.

---

## 7. Logistic Regression

### 7.1 From Regression to Classification

In classification, the target $y \in \{0, 1\}$ (binary) or $y \in \{1, 2, \ldots, K\}$ (multi-class). Using linear regression directly to predict class labels is problematic:
1. The prediction $\hat{y} = \mathbf{w}^\top \mathbf{x}$ can take any real value, not just $\{0, 1\}$.
2. MSE loss treats errors symmetrically, but predicting 0.5 is equally wrong whether the true class is 0 or 1.

**Solution.** Transform the linear output through a function that maps $\mathbb{R} \to (0, 1)$, interpreting the output as a **probability**.

---

### 7.2 The Sigmoid (Logistic) Function

The **sigmoid** (logistic) function is:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

**Key properties:**

| Property         | Statement                                                                |
| ---------------- | ------------------------------------------------------------------------ |
| **Range**        | $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$                        |
| **Symmetry**     | $\sigma(-z) = 1 - \sigma(z)$                                             |
| **Monotonicity** | $\sigma'(z) > 0$ for all $z$ (strictly increasing)                       |
| **Midpoint**     | $\sigma(0) = 0.5$                                                        |
| **Limits**       | $\lim_{z \to +\infty}\sigma(z) = 1$, $\lim_{z \to -\infty}\sigma(z) = 0$ |
| **Derivative**   | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$                                  |

**Derivative proof.** Let $s = \sigma(z) = (1 + e^{-z})^{-1}$.

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot \frac{1 + e^{-z} - 1}{1 + e^{-z}} = \sigma(z)(1 - \sigma(z))$$

$\blacksquare$

This self-referential derivative ($\sigma' = \sigma(1 - \sigma)$) makes the gradient computation elegant and efficient.

---

### 7.3 The Model

**Logistic regression** models the probability that $y = 1$ given $\mathbf{x}$:

$$p(y = 1 \mid \mathbf{x}; \mathbf{w}) = \sigma(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1 + \exp(-\mathbf{w}^\top \mathbf{x})}$$

$$p(y = 0 \mid \mathbf{x}; \mathbf{w}) = 1 - \sigma(\mathbf{w}^\top \mathbf{x}) = \sigma(-\mathbf{w}^\top \mathbf{x})$$

Compactly, for $y \in \{0, 1\}$:

$$p(y \mid \mathbf{x}; \mathbf{w}) = \sigma(\mathbf{w}^\top \mathbf{x})^y \cdot (1 - \sigma(\mathbf{w}^\top \mathbf{x}))^{1-y}$$

**The log-odds (logit) interpretation.** Define the log-odds:

$$\log\frac{p(y=1 \mid \mathbf{x})}{p(y=0 \mid \mathbf{x})} = \log\frac{\sigma(\mathbf{w}^\top\mathbf{x})}{1 - \sigma(\mathbf{w}^\top\mathbf{x})} = \mathbf{w}^\top\mathbf{x}$$

The linear function $\mathbf{w}^\top\mathbf{x}$ directly models the **log-odds** of class 1 vs. class 0. Each weight $w_j$ represents the change in log-odds per unit increase in feature $x_j$ (all else equal). This is why logistic regression is widely used in interpretable settings (medicine, social science).

---

### 7.4 The Loss Function: Binary Cross-Entropy

MSE is inappropriate for classification (it is not convex in $\mathbf{w}$ when composed with the sigmoid). Instead, we use the **negative log-likelihood**, known as **binary cross-entropy** (BCE):

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]$$

where $\hat{p}_i = \sigma(\mathbf{w}^\top\mathbf{x}_i)$.

**Why this loss?** It is the negative log-likelihood of the Bernoulli model:

$$\ell(\mathbf{w}) = \sum_{i=1}^{n} \log p(y_i \mid \mathbf{x}_i; \mathbf{w}) = \sum_{i=1}^{n}\left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]$$

Maximising $\ell$ is equivalent to minimising $\mathcal{L}$.

**Properties of BCE:**

| Property                                          | Consequence                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------- |
| **Convex** in $\mathbf{w}$                        | A unique global minimum exists (if data is not perfectly separable) |
| **No closed-form solution**                       | Must use iterative methods (GD, Newton, L-BFGS)                     |
| **Penalises confident wrong predictions heavily** | If $y_i = 1$ and $\hat{p}_i \to 0$: loss $\to +\infty$              |
| **Gradient is elegant**                           | See Section 7.5                                                     |

**Numerical stability.** Computing $\log(\hat{p})$ directly can produce $-\infty$ when $\hat{p} \to 0$. The numerically stable form is:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i z_i - \log(1 + e^{z_i})\right]$$

where $z_i = \mathbf{w}^\top\mathbf{x}_i$ (the logit). This avoids computing $\sigma$ explicitly. In code:
```python
# Numerically stable BCE
z = X @ w
loss = -np.mean(y * z - np.logaddexp(0, z))  # np.logaddexp(0, z) = log(1 + exp(z))
```

---

### 7.5 Gradient of the Cross-Entropy Loss

The gradient has a remarkably simple form. The loss for one example is:

$$\ell_i = -y_i \log\sigma(z_i) - (1-y_i)\log(1 - \sigma(z_i)), \qquad z_i = \mathbf{w}^\top\mathbf{x}_i$$

Using $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:

$$\frac{\partial \ell_i}{\partial z_i} = -y_i \cdot \frac{\sigma'(z_i)}{\sigma(z_i)} + (1-y_i) \cdot \frac{\sigma'(z_i)}{1 - \sigma(z_i)}$$

$$= -y_i(1 - \sigma(z_i)) + (1-y_i)\sigma(z_i) = \sigma(z_i) - y_i$$

Therefore:

$$\frac{\partial \ell_i}{\partial \mathbf{w}} = (\sigma(z_i) - y_i)\,\mathbf{x}_i = (\hat{p}_i - y_i)\,\mathbf{x}_i$$

Averaging over all examples:

$$\boxed{\nabla_\mathbf{w} \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)\,\mathbf{x}_i = \frac{1}{n}X^\top(\hat{\mathbf{p}} - \mathbf{y})}$$

| Gradient component | Meaning                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------- |
| $\hat{p}_i - y_i$  | Prediction error for example $i$ (positive if overconfident, negative if underconfident) |
| $\mathbf{x}_i$     | The direction in which to adjust $\mathbf{w}$                                            |

> **Observation.** This gradient has the same form as the linear regression gradient: $\nabla \mathcal{L} = \frac{1}{n}X^\top(\hat{\mathbf{y}} - \mathbf{y})$. The only difference is that $\hat{\mathbf{y}}$ is computed through the sigmoid. This structural similarity is not a coincidence — it reflects the fact that both models belong to the **generalised linear model** (GLM) family.

---

### 7.6 Decision Boundary

The predicted class is:

$$\hat{y} = \begin{cases} 1 & \text{if } \sigma(\mathbf{w}^\top\mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

Since $\sigma(z) \geq 0.5 \iff z \geq 0$, the decision boundary is the set:

$$\{\mathbf{x} : \mathbf{w}^\top\mathbf{x} = 0\}$$

This is a **hyperplane** in $\mathbb{R}^d$. In 2D ($d = 2$):

$$w_0 + w_1 x_1 + w_2 x_2 = 0 \quad \Longrightarrow \quad x_2 = -\frac{w_1}{w_2}x_1 - \frac{w_0}{w_2}$$

This is a straight line. The weight vector $\mathbf{w}$ (excluding bias) is **perpendicular** to the decision boundary.

| Distance from decision boundary                              | Probability $\hat{p}$ | Confidence                  |
| ------------------------------------------------------------ | --------------------- | --------------------------- |
| Far on the positive side ($\mathbf{w}^\top\mathbf{x} \gg 0$) | $\to 1$               | High confidence for class 1 |
| On the boundary ($\mathbf{w}^\top\mathbf{x} = 0$)            | $= 0.5$               | Maximum uncertainty         |
| Far on the negative side ($\mathbf{w}^\top\mathbf{x} \ll 0$) | $\to 0$               | High confidence for class 0 |

> **Notebook reference.** Cell 9 and the `plot_decision_boundary` function visualise this for the 2D classification dataset. The coloured regions show which side of the boundary each region falls on.
>
> **Suggested experiment.** Vary the `class_sep` parameter in `make_classification` (e.g., try $\{0.5, 1.0, 1.5, 2.0\}$). Smaller separation makes the problem harder (more overlap between classes). Observe how the decision boundary and accuracy change.

---

### 7.7 Multi-Class Extension: Softmax Regression

For $K > 2$ classes, logistic regression generalises to **softmax regression** (multinomial logistic regression). Instead of one weight vector, we have $K$ weight vectors $\mathbf{w}_1, \ldots, \mathbf{w}_K$:

$$p(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \mathbf{x})}{\sum_{j=1}^{K}\exp(\mathbf{w}_j^\top \mathbf{x})}, \qquad k = 1, \ldots, K$$

The function $\text{softmax}: \mathbb{R}^K \to \Delta^{K-1}$ (the probability simplex) is:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}$$

| Property               | Statement                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| **Output**             | A valid probability distribution: $\sum_k p_k = 1$ and $p_k > 0$                          |
| **Reduces to sigmoid** | For $K = 2$: $\text{softmax}(z_1, z_2)_1 = \sigma(z_1 - z_2)$                             |
| **Invariance**         | $\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$ (shift-invariant) |

The loss is **categorical cross-entropy**:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\log p(y_i \mid \mathbf{x}_i)$$

> **Notebook reference.** Exercise 3 (Cell 15) asks you to fit `LogisticRegression` on the Iris dataset (3 classes). Scikit-learn handles the multi-class extension automatically.
>
> **Forward pointer.** Softmax regression is the output layer of every classification neural network ([Weeks 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#43-output-layer-and-loss-functions)–[12](../../04_neural_networks/week12_training_pathologies/theory.md)). Understanding it here means you already understand how neural network classifiers make predictions.

---

## 8. Evaluation Metrics

### 8.1 Regression Metrics

| Metric    | Formula                                                     | Interpretation                                                             |
| --------- | ----------------------------------------------------------- | -------------------------------------------------------------------------- |
| **MSE**   | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$                        | Average squared error; penalises large errors disproportionately           |
| **RMSE**  | $\sqrt{\text{MSE}}$                                         | Same units as $y$; easier to interpret                                     |
| **MAE**   | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$                        | Average absolute error; robust to outliers                                 |
| **$R^2$** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Fraction of variance explained; 1 is perfect, 0 is baseline (predict mean) |

**$R^2$ in depth.** The denominator $\text{SS}_{\text{tot}} = \sum(y_i - \bar{y})^2$ is the variance of the target. The numerator $\text{SS}_{\text{res}} = \sum(y_i - \hat{y}_i)^2$ is the residual variance. $R^2$ measures how much of the target's variance the model explains.

| $R^2$ value | Interpretation                                                         |
| ----------- | ---------------------------------------------------------------------- |
| $= 1$       | Perfect fit (all residuals zero)                                       |
| $= 0$       | Model is no better than predicting $\bar{y}$                           |
| $< 0$       | Model is **worse** than predicting $\bar{y}$ (possible with test data) |

> **Notebook reference.** Cell 5 could report $R^2$: `r2_score(y, y_pred)`. The import is already present.

---

### 8.2 Classification Metrics

| Metric        | Formula                                                                                 | When to use                                               |
| ------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Accuracy**  | $\frac{\text{correct predictions}}{n}$                                                  | Balanced classes                                          |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$                                               | Cost of false positives is high (e.g., spam filter)       |
| **Recall**    | $\frac{\text{TP}}{\text{TP} + \text{FN}}$                                               | Cost of false negatives is high (e.g., disease detection) |
| **F1 Score**  | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Imbalanced classes (harmonic mean of P and R)             |
| **AUC-ROC**   | Area under the ROC curve                                                                | Threshold-independent evaluation                          |

| Symbol | Meaning                                                             |
| ------ | ------------------------------------------------------------------- |
| TP     | True Positives — correctly predicted as positive                    |
| FP     | False Positives — incorrectly predicted as positive (Type I error)  |
| FN     | False Negatives — incorrectly predicted as negative (Type II error) |
| TN     | True Negatives — correctly predicted as negative                    |

**The confusion matrix** organises these counts:

$$\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}$$

> **When accuracy is misleading.** If 99% of emails are not spam, a model that always predicts "not spam" has 99% accuracy but 0% recall — it never catches spam. In imbalanced settings, precision, recall, and F1 are much more informative.

---

## 9. Regularisation Preview

The bias–variance analysis reveals that complex models overfit. **Regularisation** constrains the model to reduce variance at the cost of a small increase in bias.

The main approaches (detailed in [Week 04](../week04_dimensionality_reduction/theory.md#5-pca-via-the-singular-value-decomposition)/[Week 06](../week06_regularization/theory.md#3-ridge-regression-l2-regularisation)):

**Ridge regression (L2 regularisation):** Add a penalty on the weight magnitudes:

$$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2$$

The closed-form solution becomes:

$$\mathbf{w}^*_{\text{Ridge}} = (X^\top X + \lambda I)^{-1}X^\top\mathbf{y}$$

The $\lambda I$ term ensures $X^\top X + \lambda I$ is always invertible (even when $X^\top X$ is singular), and shrinks all weights toward zero.

**Lasso regression (L1 regularisation):**

$$\mathcal{L}_{\text{Lasso}}(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|_1$$

L1 promotes **sparsity** — some weights are driven exactly to zero, performing automatic feature selection. No closed-form solution exists; solved by coordinate descent or proximal gradient methods.

> **Notebook reference.** Exercise 1 (Cell 13) asks you to implement Ridge regression from scratch using the modified normal equations. The code hint in the README:
> ```python
> w = np.linalg.solve(XTX + lam*np.eye(X.shape[1]), X.T @ y)
> ```
>
> **Suggested experiment.** Sweep $\lambda \in \{0.001, 0.01, 0.1, 1, 10\}$ for the polynomial regression problem (degree 15). Plot test error vs. $\lambda$. At $\lambda = 0$: overfitting (the degree-15 polynomial oscillates wildly). As $\lambda$ increases: the weights shrink, the polynomial smooths out, and test error decreases — until $\lambda$ is too large and the model underfits (high bias). The optimal $\lambda$ balances bias and variance.

---

## 10. Connections to the Rest of the Course

Linear and logistic regression are not just foundations — they reappear throughout the course in transformed guises:

| Week                          | Connection                                                                                                                        |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **[Week 06](../week06_regularization/theory.md#3-ridge-regression-l2-regularisation) (Regularisation)**  | Ridge and Lasso add L2/L1 penalties to the models built here                                                                      |
| **[Week 07](../../03_probability/week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle) (Likelihood)**      | The MLE derivation of Section 4.2 is generalised to arbitrary distributions                                                       |
| **[Week 08](../../03_probability/week08_uncertainty/theory.md#8-bayesian-linear-regression) (Uncertainty)**     | Bayesian linear regression puts a prior on $\mathbf{w}$ — Ridge regression is the MAP estimate with a Gaussian prior              |
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#2-from-linear-models-to-neural-networks) (NN from scratch)** | A single-layer neural network without activation is exactly linear regression; with sigmoid activation, it is logistic regression |
| **[Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md#43-common-layer-types) (PyTorch)**         | `nn.Linear(d, 1)` implements $\mathbf{w}^\top\mathbf{x} + b$                                                                      |
| **[Week 18](../../06_sequence_models/week18_transformers/theory.md#41-sub-layer-1-multi-head-self-attention) (Transformers)**    | The attention mechanism outputs a weighted sum — a *learned* linear regression                                                    |

> **The unifying principle.** Linear models are the **atoms** of machine learning. Complex models are molecules built from linear models + nonlinearities + composition. Breaking any complex model down into its linear components is a powerful debugging strategy.

---

## 11. Notebook Reference Guide

| Cell                    | Section                            | What to observe                                             | Theory reference |
| ----------------------- | ---------------------------------- | ----------------------------------------------------------- | ---------------- |
| 2 (Imports)             | Setup                              | Libraries loaded: numpy, matplotlib, sklearn                | —                |
| 3 (Cache)               | Caching utility                    | Results saved/loaded from `cache_week03/`                   | —                |
| 5 (Normal equations)    | Linear regression fit + plot       | `w0 ≈ 2, w1 ≈ 3`; fitted line through data                  | Section 3.3      |
| 7 (sklearn comparison)  | Verify implementation              | `np.allclose()` should return `True`                        | Section 3.3      |
| 9 (Logistic regression) | Classification + decision boundary | Two subplots: train/test; linear boundary separates classes | Section 7.6      |
| 11 (Bias-variance)      | Polynomial degree sweep            | U-shaped test error curve; optimal degree ≈ 3–5             | Section 6.3      |
| Ex.1 (Ridge)            | Regularised normal equations       | Coefficient norm shrinks as $\lambda$ increases             | Section 9        |
| Ex.2 (GD fit)           | Gradient descent for lin. reg.     | GD converges to same $\mathbf{w}$ as normal equations       | Section 3.5      |
| Ex.3 (Iris)             | Multi-class logistic regression    | Setosa perfectly separable; versicolor/virginica overlap    | Section 7.7      |
| Ex.5 (Real dataset)     | End-to-end mini-project            | Compare train/test metrics; interpret bias/variance         | Section 6.4      |

**Suggested modifications across exercises:**

| Modification                                       | What it reveals                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------------- |
| Increase noise $\sigma$ in synthetic data (Cell 5) | $R^2$ decreases; irreducible error dominates                                    |
| Use $n = 10$ samples instead of 100                | High variance: weights fluctuate across random seeds                            |
| Set all features to same column in $X$             | $X^\top X$ becomes singular; `np.linalg.solve` fails (motivates regularisation) |
| Plot logistic loss surface for 2 weights           | Convex bowl shape; compare to MSE loss surface                                  |
| Add polynomial features to logistic regression     | Nonlinear decision boundary; observe overfitting at high degree                 |

---

## 12. Symbol Reference

| Symbol                                       | Name                         | Meaning                                                 |
| -------------------------------------------- | ---------------------------- | ------------------------------------------------------- |
| $\mathbf{x} \in \mathbb{R}^d$                | Feature vector               | Input measurements (one example)                        |
| $\tilde{\mathbf{x}} \in \mathbb{R}^{d+1}$    | Augmented feature vector     | $[1, x_1, \ldots, x_d]^\top$ (bias absorbed)            |
| $y$                                          | Target                       | The quantity to predict (scalar)                        |
| $\mathbf{y} \in \mathbb{R}^n$                | Target vector                | All $n$ targets stacked                                 |
| $\hat{y}$                                    | Prediction                   | Model output for one example                            |
| $\hat{\mathbf{y}} \in \mathbb{R}^n$          | Prediction vector            | $X\mathbf{w}$                                           |
| $X \in \mathbb{R}^{n \times (d+1)}$          | Design matrix                | All features stacked (with bias column)                 |
| $\mathbf{w} \in \mathbb{R}^{d+1}$            | Weight vector                | Learnable parameters (including bias)                   |
| $b$                                          | Bias (intercept)             | Constant offset; absorbed into $\mathbf{w}$             |
| $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$ | Residual vector              | Prediction errors                                       |
| $n$                                          | Number of examples           | Training set size                                       |
| $d$                                          | Number of features           | Input dimensionality                                    |
| $p$                                          | Polynomial degree            | Degree of feature expansion                             |
| $\varepsilon_i$                              | Noise                        | $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$           |
| $\sigma^2$                                   | Noise variance               | Irreducible error                                       |
| $\sigma(z)$                                  | Sigmoid function             | $1/(1 + e^{-z})$                                        |
| $\hat{p}_i$                                  | Predicted probability        | $\sigma(\mathbf{w}^\top\mathbf{x}_i)$                   |
| $z_i$                                        | Logit                        | $\mathbf{w}^\top\mathbf{x}_i$ (pre-sigmoid)             |
| $\mathcal{L}(\mathbf{w})$                    | Loss function                | MSE (regression) or BCE (classification)                |
| $\ell(\mathbf{w})$                           | Log-likelihood               | $\log p(\mathbf{y} \mid X, \mathbf{w})$                 |
| $H = X(X^\top X)^{-1}X^\top$                 | Hat matrix                   | Projects $\mathbf{y}$ onto $\text{Col}(X)$              |
| $R^2$                                        | Coefficient of determination | $1 - \text{SS}_\text{res}/\text{SS}_\text{tot}$         |
| $\lambda$                                    | Regularisation strength      | Controls bias–variance balance in Ridge/Lasso           |
| $\phi(\mathbf{x})$                           | Feature map                  | Nonlinear transformation of inputs (e.g., polynomial)   |
| $\Phi$                                       | Transformed design matrix    | $[\phi(\mathbf{x}_1), \ldots, \phi(\mathbf{x}_n)]^\top$ |
| $\kappa$                                     | Condition number             | $\lambda_\max / \lambda_\min$ of $X^\top X$             |
| $K$                                          | Number of classes            | For multi-class classification                          |
| $\Delta^{K-1}$                               | Probability simplex          | $\{p \in \mathbb{R}^K : p_k \geq 0, \sum p_k = 1\}$     |
| TP, FP, FN, TN                               | Confusion matrix entries     | True/False Positives/Negatives                          |

---

## 13. References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 3 ("Linear Models for Regression") and Chapter 4 ("Linear Models for Classification"). Springer. — The definitive reference.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Chapters 3 and 7. Springer. — Detailed treatment of bias-variance, regularisation, model selection. Free PDF: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/).
3. Ng, A. (2012). Stanford CS229 Lecture Notes. — Clear derivations of linear and logistic regression with the MLE connection.
4. Gauss, C. F. (1809). *Theoria motus corporum coelestium*. — The original derivation of least squares (!).
5. Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*, Chapter 9. Cambridge University Press. — Rigorous treatment of linear predictors.
6. Scikit-learn User Guide: Linear Models. [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html). — Practical documentation for `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`.
7. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning* (ISL), Chapters 3–4. Springer. — Accessible introduction with R examples. Free PDF: [https://www.statlearning.com/](https://www.statlearning.com/).
8. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapter 7. MIT Press. — Bayesian perspective on linear regression.
9. Strang, G. (2019). *Linear Algebra and Learning from Data*. Wellesley-Cambridge. — The geometric view of least squares as projection.
