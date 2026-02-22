# Mathematical and Data Foundations: A Rigorous Introduction

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Part I — Linear Algebra](#2-part-i--linear-algebra)
    - 2.1 [Scalars, Vectors, and Matrices](#21-scalars-vectors-and-matrices)
    - 2.2 [Vector Operations](#22-vector-operations)
    - 2.3 [Matrix Operations](#23-matrix-operations)
    - 2.4 [Systems of Linear Equations](#24-systems-of-linear-equations)
    - 2.5 [Linear Independence, Span, and Rank](#25-linear-independence-span-and-rank)
    - 2.6 [Norms](#26-norms)
    - 2.7 [Eigendecomposition](#27-eigendecomposition)
    - 2.8 [Singular Value Decomposition (SVD)](#28-singular-value-decomposition-svd)
    - 2.9 [Positive Definite Matrices](#29-positive-definite-matrices)
3. [Part II — Calculus and Optimisation](#3-part-ii--calculus-and-optimisation)
    - 3.1 [The Derivative as a Rate of Change](#31-the-derivative-as-a-rate-of-change)
    - 3.2 [Differentiation Rules](#32-differentiation-rules)
    - 3.3 [Partial Derivatives and the Gradient](#33-partial-derivatives-and-the-gradient)
    - 3.4 [The Jacobian and the Hessian](#34-the-jacobian-and-the-hessian)
    - 3.5 [Taylor Expansion](#35-taylor-expansion)
    - 3.6 [The Chain Rule in Depth](#36-the-chain-rule-in-depth)
    - 3.7 [Vector Calculus Identities for ML](#37-vector-calculus-identities-for-ml)
    - 3.8 [Numerical Differentiation](#38-numerical-differentiation)
4. [Part III — Probability and Statistics](#4-part-iii--probability-and-statistics)
    - 4.1 [Probability Axioms](#41-probability-axioms)
    - 4.2 [Conditional Probability and Bayes' Theorem](#42-conditional-probability-and-bayes-theorem)
    - 4.3 [Random Variables and Distributions](#43-random-variables-and-distributions)
    - 4.4 [Expectation, Variance, and Covariance](#44-expectation-variance-and-covariance)
    - 4.5 [Important Distributions](#45-important-distributions)
    - 4.6 [The Law of Large Numbers and the Central Limit Theorem](#46-the-law-of-large-numbers-and-the-central-limit-theorem)
    - 4.7 [Maximum Likelihood Estimation — Preview](#47-maximum-likelihood-estimation--preview)
    - 4.8 [Information Theory Basics](#48-information-theory-basics)
5. [Part IV — Data Representation and Exploratory Analysis](#5-part-iv--data-representation-and-exploratory-analysis)
    - 5.1 [Data as Matrices](#51-data-as-matrices)
    - 5.2 [Feature Types and Encoding](#52-feature-types-and-encoding)
    - 5.3 [The EDA Protocol](#53-the-eda-protocol)
    - 5.4 [Common Pitfalls](#54-common-pitfalls)
6. [Part V — Numerical Computing with NumPy](#6-part-v--numerical-computing-with-numpy)
    - 6.1 [The Shape Discipline](#61-the-shape-discipline)
    - 6.2 [Broadcasting](#62-broadcasting)
    - 6.3 [Vectorisation](#63-vectorisation)
    - 6.4 [Numerical Stability](#64-numerical-stability)
7. [Symbol Reference](#7-symbol-reference)
8. [References](#8-references)

---

## 1. Scope and Purpose

This document provides the mathematical foundations required from [Week 01](../../02_fundamentals/week01_optimization/theory.md) onward. It is not a textbook on linear algebra or probability; it is a targeted, rigorous treatment of exactly the concepts that arise in machine learning. Every definition and theorem included here is used explicitly in subsequent weeks.

The treatment follows a deliberate pattern: **formal definition → geometric or physical intuition → concrete example → code reference → suggested experiment**. Readers are strongly encouraged to implement the examples in the accompanying `starter.ipynb` (Week 00b) or in a scratch notebook.

**Prerequisite.** Comfort with basic algebra and high-school-level function notation ($f(x) = \ldots$). Everything else is developed here.

---

## 2. Part I — Linear Algebra

Linear algebra is the language of data in machine learning. A dataset is a matrix. A model's forward pass is a sequence of matrix operations. Understanding these operations — what they compute, what they mean geometrically, and how they behave numerically — is essential.

### 2.1 Scalars, Vectors, and Matrices

**Scalar.** A single real number $x \in \mathbb{R}$. In ML contexts: a loss value, a learning rate, a single feature.

**Vector.** An ordered list of $d$ real numbers, written as a column:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \in \mathbb{R}^d$$

| Symbol         | Meaning                                       |
| -------------- | --------------------------------------------- |
| $\mathbf{x}$   | A vector (bold lowercase)                     |
| $x_j$          | The $j$-th component (scalar entry)           |
| $d$            | Dimensionality — the number of entries        |
| $\mathbb{R}^d$ | The space of all $d$-dimensional real vectors |

A vector can represent:
- A **data point**: one row of a dataset, where each entry is a measured feature.
- A **parameter vector**: the weights of a model.
- A **gradient**: the direction and magnitude of steepest ascent of a loss function.

> **Geometric interpretation.** A vector in $\mathbb{R}^2$ is an arrow in the plane. A vector in $\mathbb{R}^d$ is an arrow in $d$-dimensional space — impossible to draw for $d > 3$, but the algebraic operations remain identical.

**Matrix.** A rectangular array of $n \times d$ real numbers:

$$X = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1d} \\ x_{21} & x_{22} & \cdots & x_{2d} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n1} & x_{n2} & \cdots & x_{nd} \end{bmatrix} \in \mathbb{R}^{n \times d}$$

| Symbol                    | Meaning                                     |
| ------------------------- | ------------------------------------------- |
| $X$                       | A matrix (uppercase italic)                 |
| $x_{ij}$                  | Entry in row $i$, column $j$                |
| $n$                       | Number of rows (samples in a dataset)       |
| $d$                       | Number of columns (features in a dataset)   |
| $\mathbb{R}^{n \times d}$ | The space of all $n \times d$ real matrices |

**Convention.** Throughout this course, rows of $X$ are data points and columns are features. The $i$-th row, written $\mathbf{x}_i^\top$, is the feature vector for sample $i$.

**Tensor.** A generalisation to higher dimensions (3D, 4D, ...). A colour image is a 3D tensor of shape (height, width, channels). In PyTorch ([Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md#2-tensors)+), a batch of images is a 4D tensor of shape (batch, channels, height, width). For now, scalars, vectors, and matrices suffice.

```python
import numpy as np

# Scalar
lr = 0.01                          # shape: () — 0-dimensional

# Vector
x = np.array([1.0, 2.0, 3.0])     # shape: (3,) — 1D array

# Column vector (explicit)
x_col = x.reshape(3, 1)           # shape: (3, 1)

# Matrix
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])             # shape: (3, 2) — 3 samples, 2 features
```

> **Suggested experiment.** Create arrays of shapes `(3,)`, `(3, 1)`, and `(1, 3)` and multiply them pairwise with `@`. Observe which combinations broadcast to an outer product, which to a dot product, and which raise errors. This builds the shape intuition that prevents bugs in every subsequent week.

---

### 2.2 Vector Operations

#### Dot Product (Inner Product)

The dot product of two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$ is:

$$\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^\top \mathbf{b} = \sum_{j=1}^{d} a_j b_j$$

This yields a **scalar**.

| Symbol                       | Meaning                                              |
| ---------------------------- | ---------------------------------------------------- |
| $\mathbf{a}^\top$            | Row vector (transpose of column vector $\mathbf{a}$) |
| $\mathbf{a}^\top \mathbf{b}$ | Dot product (inner product)                          |

**Geometric interpretation.** The dot product equals:

$$\mathbf{a}^\top \mathbf{b} = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos \alpha$$

where $\alpha$ is the angle between the two vectors and $\|\cdot\|$ is the Euclidean norm (length). Therefore:

- $\mathbf{a}^\top \mathbf{b} > 0$: the vectors point in roughly the same direction.
- $\mathbf{a}^\top \mathbf{b} = 0$: the vectors are **orthogonal** (perpendicular).
- $\mathbf{a}^\top \mathbf{b} < 0$: the vectors point in roughly opposite directions.

> **Why this matters in ML.** A linear model computes $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$. The prediction is literally the dot product of the weight vector and the input vector, plus a bias. A large positive $\hat{y}$ means $\mathbf{x}$ aligns with $\mathbf{w}$; a large negative $\hat{y}$ means they point in opposite directions. In classification, the sign of $\hat{y}$ determines the predicted class.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Three equivalent ways to compute the dot product
print(a @ b)            # 32
print(np.dot(a, b))     # 32
print(np.sum(a * b))    # 32 — element-wise multiply, then sum
```

#### Outer Product

The outer product of $\mathbf{a} \in \mathbb{R}^m$ and $\mathbf{b} \in \mathbb{R}^n$ is:

$$\mathbf{a} \mathbf{b}^\top = \begin{bmatrix} a_1 b_1 & a_1 b_2 & \cdots & a_1 b_n \\ a_2 b_1 & a_2 b_2 & \cdots & a_2 b_n \\ \vdots & \vdots & \ddots & \vdots \\ a_m b_1 & a_m b_2 & \cdots & a_m b_n \end{bmatrix} \in \mathbb{R}^{m \times n}$$

This yields a **matrix**. It appears in gradient computations for weight matrices in neural networks ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#74-he-kaiming-initialisation)): if the loss gradient with respect to the output is $\boldsymbol{\delta}$ and the layer input is $\mathbf{h}$, the gradient with respect to the weight matrix is $\boldsymbol{\delta} \mathbf{h}^\top$.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5])
print(np.outer(a, b))   # shape (3, 2)
# [[ 4,  5],
#  [ 8, 10],
#  [12, 15]]
```

#### Element-wise (Hadamard) Product

$$(\mathbf{a} \odot \mathbf{b})_j = a_j \cdot b_j$$

This is not a standard linear algebra operation, but it is ubiquitous in ML code (loss computation, gating mechanisms in LSTMs, attention masking).

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a * b)   # [4, 10, 18] — element-wise in NumPy
```

---

### 2.3 Matrix Operations

#### Matrix–Vector Multiplication

Given $A \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$, the product $\mathbf{y} = A\mathbf{x} \in \mathbb{R}^m$ is defined as:

$$y_i = \sum_{j=1}^{n} a_{ij} \, x_j \qquad \text{for } i = 1, \ldots, m$$

Each entry $y_i$ is the dot product of the $i$-th row of $A$ with $\mathbf{x}$.

> **ML interpretation.** When $A$ is the design matrix $X \in \mathbb{R}^{n \times d}$ and $\mathbf{x}$ is replaced by a weight vector $\mathbf{w} \in \mathbb{R}^d$, the product $X\mathbf{w}$ computes all $n$ predictions simultaneously:
>
> $$\hat{\mathbf{y}} = X\mathbf{w} = \begin{bmatrix} \mathbf{x}_1^\top \mathbf{w} \\ \vdots \\ \mathbf{x}_n^\top \mathbf{w} \end{bmatrix}$$
>
> This is why vectorised code (one matrix multiply) is vastly faster than a Python loop over samples.

**Geometric interpretation.** Multiplying by a matrix is a **linear transformation**: it can rotate, scale, shear, or project vectors. A $2 \times 2$ matrix maps every point in the plane to a new point. Understanding this is key to understanding what each layer of a neural network does ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#74-he-kaiming-initialisation)).

#### Matrix–Matrix Multiplication

Given $A \in \mathbb{R}^{m \times p}$ and $B \in \mathbb{R}^{p \times n}$, the product $C = AB \in \mathbb{R}^{m \times n}$ is:

$$c_{ij} = \sum_{k=1}^{p} a_{ik} \, b_{kj}$$

The inner dimensions must match: columns of $A$ = rows of $B$.

```python
A = np.array([[1, 2], [3, 4]])    # (2, 2)
B = np.array([[5, 6], [7, 8]])    # (2, 2)
print(A @ B)
# [[19, 22],
#  [43, 50]]
```

> **Key properties:**
> - Matrix multiplication is **associative**: $(AB)C = A(BC)$.
> - Matrix multiplication is **not commutative**: $AB \neq BA$ in general.
> - Matrix multiplication is **distributive**: $A(B + C) = AB + AC$.

#### Transpose

The transpose of $A \in \mathbb{R}^{m \times n}$ is $A^\top \in \mathbb{R}^{n \times m}$, defined by $(A^\top)_{ij} = a_{ji}$.

Key identities:
- $(A^\top)^\top = A$
- $(AB)^\top = B^\top A^\top$ (the order reverses)
- $(\mathbf{x}^\top \mathbf{y})^\top = \mathbf{y}^\top \mathbf{x} = \mathbf{x}^\top \mathbf{y}$ (dot product is a scalar, so transpose is itself)

A matrix $A$ is **symmetric** if $A = A^\top$. Covariance matrices (Section 4.4) are always symmetric.

#### Inverse

A square matrix $A \in \mathbb{R}^{n \times n}$ is **invertible** if there exists $A^{-1}$ such that $A A^{-1} = A^{-1} A = I$, where $I$ is the identity matrix.

$$I = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

The inverse exists if and only if $\det(A) \neq 0$ (the matrix is **non-singular**). In ML, the inverse appears in closed-form solutions: the normal equation for linear regression ([Week 03](../../02_fundamentals/week03_linear_models/theory.md#33-the-normal-equations-closed-form-solution)) is $\mathbf{w}^* = (X^\top X)^{-1} X^\top \mathbf{y}$.

> **Practical warning.** Computing $A^{-1}$ explicitly is numerically unstable and slow. In practice, one solves the linear system $A\mathbf{w} = \mathbf{b}$ directly using `np.linalg.solve(A, b)`, which uses LU decomposition internally.

```python
A = np.array([[2, 1], [5, 3]])
b = np.array([4, 7])

# BAD: explicit inverse
w_bad = np.linalg.inv(A) @ b

# GOOD: solve the system directly
w_good = np.linalg.solve(A, b)

print(np.allclose(w_bad, w_good))   # True, but w_good is faster and more stable
```

#### The Trace

The trace of a square matrix $A$ is the sum of its diagonal entries:

$$\text{tr}(A) = \sum_{i=1}^{n} a_{ii}$$

Useful properties:
- $\text{tr}(AB) = \text{tr}(BA)$ (cyclic permutation)
- $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(A) = \sum_{i} \lambda_i$ where $\lambda_i$ are the eigenvalues of $A$

The trace appears in loss functions involving matrices (e.g., PCA objective in [Week 04](../../02_fundamentals/week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view)) and in the Frobenius norm: $\|A\|_F = \sqrt{\text{tr}(A^\top A)}$.

---

### 2.4 Systems of Linear Equations

A system of $m$ linear equations in $n$ unknowns can be written as:

$$A\mathbf{x} = \mathbf{b}$$

where $A \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$.

Three cases arise:
1. **Unique solution** ($m = n$, $A$ invertible): $\mathbf{x} = A^{-1}\mathbf{b}$.
2. **No solution** ($m > n$, overdetermined system): common in ML, where $n$ equations cannot be exactly satisfied. The **least-squares solution** minimises $\|A\mathbf{x} - \mathbf{b}\|^2$ — this is exactly linear regression ([Week 03](../../02_fundamentals/week03_linear_models/theory.md#3-linear-regression)).
3. **Infinitely many solutions** ($m < n$, underdetermined system): arises in overparameterised neural networks (more parameters than data points).

> **Connection.** Linear regression is the problem: given $X$ (data) and $\mathbf{y}$ (targets), find $\mathbf{w}$ that "best" solves $X\mathbf{w} \approx \mathbf{y}$ in the least-squares sense. The solution is the **normal equation**: $\mathbf{w}^* = (X^\top X)^{-1} X^\top \mathbf{y}$. This is derived rigorously in [Week 03](../../02_fundamentals/week03_linear_models/theory.md#33-the-normal-equations-closed-form-solution).

---

### 2.5 Linear Independence, Span, and Rank

**Linear independence.** A set of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is linearly independent if the only solution to:

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k = \mathbf{0}$$

is $c_1 = c_2 = \cdots = c_k = 0$. In other words, no vector in the set can be written as a linear combination of the others.

> **Intuition.** In $\mathbb{R}^2$, two vectors are linearly independent if they do not point in the same (or exactly opposite) direction. Together they can reach any point in the plane. If they are dependent (one is a scaled version of the other), they can only reach points along a single line.

**Span.** The span of $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is the set of all vectors expressible as $\sum_{i} c_i \mathbf{v}_i$. It is a subspace of $\mathbb{R}^d$.

**Rank.** The rank of a matrix $A$ is the maximum number of linearly independent columns (equivalently, rows). A matrix $A \in \mathbb{R}^{m \times n}$ has $\text{rank}(A) \leq \min(m, n)$.

- **Full rank**: $\text{rank}(A) = \min(m, n)$. The columns are linearly independent.
- **Rank deficient**: $\text{rank}(A) < \min(m, n)$. Some columns are redundant.

> **ML relevance.** If the data matrix $X$ is rank-deficient (e.g., two features are perfectly correlated), the normal equation $X^\top X$ is singular and the least-squares solution is not unique. This motivates regularisation ([Week 06](../../02_fundamentals/week06_regularization/theory.md#2-why-regularisation-the-overfitting-problem-revisited)). PCA ([Week 04](../../02_fundamentals/week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view)) explicitly seeks a low-rank approximation of the data.

```python
X = np.array([[1, 2], [2, 4], [3, 6]])   # rank 1: column 2 = 2 * column 1
print(np.linalg.matrix_rank(X))           # 1
```

---

### 2.6 Norms

A **norm** is a function that assigns a non-negative "size" to a vector. Formally, $\|\cdot\| : \mathbb{R}^d \to \mathbb{R}_{\geq 0}$ must satisfy:

1. $\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}$
2. $\|\alpha \mathbf{x}\| = |\alpha| \, \|\mathbf{x}\|$ (homogeneity)
3. $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$ (triangle inequality)

The most important norms in ML:

| Norm            | Formula                                          | Notation           | ML Usage                                                   |
| --------------- | ------------------------------------------------ | ------------------ | ---------------------------------------------------------- |
| $L_1$ norm      | $\|\mathbf{x}\|_1 = \sum_{j=1}^{d}               | x_j                | $                                                          | Manhattan distance | Lasso regularisation ([Week 06](../../02_fundamentals/week06_regularization/theory.md#4-lasso-regression-l1-regularisation)): promotes **sparsity** |
| $L_2$ norm      | $\|\mathbf{x}\|_2 = \sqrt{\sum_{j=1}^{d} x_j^2}$ | Euclidean distance | Ridge regularisation ([Week 06](../../02_fundamentals/week06_regularization/theory.md#3-ridge-regression-l2-regularisation)): promotes **small** weights |
| $L_\infty$ norm | $\|\mathbf{x}\|_\infty = \max_j                  | x_j                | $                                                          | Chebyshev distance | Robustness bounds                                     |
| Frobenius       | $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$           | Matrix "size"      | Weight decay for matrix parameters                         |

> **Intuition.** In $\mathbb{R}^2$, the set of points with $\|\mathbf{x}\| = 1$ (the "unit ball") has different shapes depending on the norm:
> - $L_2$: a circle (all directions are treated equally).
> - $L_1$: a diamond (axis-aligned directions are cheapest).
> - $L_\infty$: a square (the largest coordinate determines the norm).
>
> The $L_1$ diamond has corners on the axes, which is why $L_1$ regularisation pushes weights exactly to zero — the optimum tends to land at a corner. This geometric insight is the key to understanding sparsity in [Week 06](../../02_fundamentals/week06_regularization/theory.md#42-sparsity-why-lasso-produces-zeros).

```python
x = np.array([3, -4])

print(np.linalg.norm(x, ord=1))     # 7.0   — L1
print(np.linalg.norm(x, ord=2))     # 5.0   — L2
print(np.linalg.norm(x, ord=np.inf)) # 4.0  — L∞
```

> **Suggested experiment.** Plot the unit balls of $L_1$, $L_2$, and $L_\infty$ in 2D. Then add contours of a loss function $\mathcal{L}(\mathbf{w})$ and visually identify where the norm constraint touches the loss contour. This illustrates how $L_1$ promotes corner solutions (sparse weights) while $L_2$ shrinks weights smoothly.

---

### 2.7 Eigendecomposition

An **eigenvector** of a square matrix $A \in \mathbb{R}^{n \times n}$ is a non-zero vector $\mathbf{v}$ such that:

$$A \mathbf{v} = \lambda \mathbf{v}$$

| Symbol       | Meaning                                                            |
| ------------ | ------------------------------------------------------------------ |
| $\mathbf{v}$ | Eigenvector — a direction that $A$ merely scales (does not rotate) |
| $\lambda$    | Eigenvalue — the scaling factor along direction $\mathbf{v}$       |

If $A$ has $n$ linearly independent eigenvectors, $A$ can be decomposed as:

$$A = V \Lambda V^{-1}$$

where $V = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_n]$ is the matrix of eigenvectors and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ is the diagonal matrix of eigenvalues.

If $A$ is **real and symmetric** ($A = A^\top$), then all eigenvalues are real, the eigenvectors are orthogonal, and:

$$A = Q \Lambda Q^\top$$

where $Q$ is an orthogonal matrix ($Q^\top Q = I$). This is the **spectral theorem**.

> **Geometric interpretation.** A symmetric matrix $A$ acts on any vector by: (1) decomposing it into components along the eigenvectors, (2) scaling each component by the corresponding eigenvalue, and (3) reassembling. The eigenvectors define a natural coordinate system for the transformation.

> **ML relevance.** The covariance matrix $\Sigma$ is real and symmetric. Its eigenvectors are the **principal components** — the directions of maximum variance in the data. Its eigenvalues are the variances along those directions. PCA ([Week 04](../../02_fundamentals/week04_dimensionality_reduction/theory.md#3-covariance-and-correlation)) computes this decomposition and discards the directions with small eigenvalues.

```python
A = np.array([[2, 1],
              [1, 3]])
eigenvalues, eigenvectors = np.linalg.eigh(A)   # eigh for symmetric matrices
print("Eigenvalues:", eigenvalues)     # [1.38..., 3.61...]
print("Eigenvectors:\n", eigenvectors) # columns are eigenvectors
```

> **Suggested experiment.** Generate a 2D dataset with correlation (e.g., height and weight). Compute the covariance matrix, find its eigenvectors, and plot them as arrows on top of a scatter plot. The longer arrow (larger eigenvalue) should point along the "spread" of the data. Rotate the data to align with the eigenvectors — this is what PCA does.

---

### 2.8 Singular Value Decomposition (SVD)

The SVD generalises eigendecomposition to **any** matrix, not just square ones. For $A \in \mathbb{R}^{m \times n}$:

$$A = U \Sigma V^\top$$

| Symbol   | Shape        | Meaning                                                                            |
| -------- | ------------ | ---------------------------------------------------------------------------------- |
| $U$      | $m \times m$ | Left singular vectors (columns are orthonormal)                                    |
| $\Sigma$ | $m \times n$ | Diagonal matrix of **singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ |
| $V$      | $n \times n$ | Right singular vectors (columns are orthonormal)                                   |

Properties:
- The singular values are always non-negative.
- The number of non-zero singular values equals $\text{rank}(A)$.
- The eigenvalues of $A^\top A$ are $\sigma_i^2$, and the columns of $V$ are the eigenvectors of $A^\top A$.

> **ML relevance.** The **truncated SVD** retains only the top $k$ singular values and their associated vectors, yielding the best rank-$k$ approximation of $A$ (in the Frobenius norm sense — the Eckart–Young theorem). This is the mathematical foundation of PCA ([Week 04](../../02_fundamentals/week04_dimensionality_reduction/theory.md#5-pca-via-the-singular-value-decomposition)) and low-rank matrix factorisation.

> **Intuition.** Every matrix transformation can be decomposed into three steps: (1) rotate/reflect the input ($V^\top$), (2) scale along each axis ($\Sigma$), (3) rotate/reflect the output ($U$). The singular values tell you how much each axis is stretched. If some singular values are near zero, the corresponding dimensions carry little information and can be discarded.

```python
A = np.random.randn(5, 3)
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
print(U.shape, sigma.shape, Vt.shape)   # (5, 3), (3,), (3, 3)

# Reconstruct A from SVD
A_reconstructed = U @ np.diag(sigma) @ Vt
print(np.allclose(A, A_reconstructed))   # True
```

---

### 2.9 Positive Definite Matrices

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is:

- **Positive semi-definite (PSD)** if $\mathbf{x}^\top A \mathbf{x} \geq 0$ for all $\mathbf{x} \in \mathbb{R}^n$. Equivalently, all eigenvalues are $\geq 0$.
- **Positive definite (PD)** if $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$. Equivalently, all eigenvalues are $> 0$.

| Symbol/Term | Meaning                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------ |
| PSD         | Positive semi-definite: $\mathbf{x}^\top A \mathbf{x} \geq 0$ for all $\mathbf{x}$         |
| PD          | Positive definite: $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$ |

> **Why this matters.** The Hessian matrix of a function (Section 3.4) is PD at a point if and only if that point is a **strict local minimum**. The matrix $X^\top X$ in linear regression is always PSD, and it is PD if and only if $X$ has full column rank — which guarantees a unique least-squares solution. Covariance matrices are always PSD.

> **Intuition.** A positive definite matrix defines a "bowl-shaped" quadratic function $f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$. The contours of this function are ellipses (in 2D), with the axes aligned to the eigenvectors of $A$ and the widths proportional to $1/\sqrt{\lambda_i}$. Gradient descent on such a bowl converges to the unique minimum at the origin. If $A$ has very different eigenvalues ($\lambda_{\max} \gg \lambda_{\min}$), the ellipses are highly elongated and gradient descent zigzags — the **condition number** $\kappa(A) = \lambda_{\max}/\lambda_{\min}$ quantifies this difficulty ([Week 01](../../02_fundamentals/week01_optimization/theory.md#4-gradient-descent)–02).

```python
# X^T X is always PSD
X = np.random.randn(100, 5)
XtX = X.T @ X
eigenvalues = np.linalg.eigvalsh(XtX)
print("All eigenvalues >= 0:", np.all(eigenvalues >= -1e-10))   # True
print("Condition number:", eigenvalues.max() / eigenvalues.min())
```

---

## 3. Part II — Calculus and Optimisation

Calculus provides the tools to find the parameters $\theta^*$ that minimise the loss $\mathcal{L}(\theta)$. The central idea: **the gradient tells you in which direction the loss increases fastest**, so you step in the opposite direction.

### 3.1 The Derivative as a Rate of Change

For a function $f : \mathbb{R} \to \mathbb{R}$, the derivative at a point $x$ is defined as:

$$f'(x) = \frac{df}{dx} = \lim_{\epsilon \to 0} \frac{f(x + \epsilon) - f(x)}{\epsilon}$$

| Symbol                     | Meaning                               |
| -------------------------- | ------------------------------------- |
| $f'(x)$ or $\frac{df}{dx}$ | The derivative of $f$ at $x$          |
| $\epsilon$                 | An infinitesimally small perturbation |

**Interpretation.** The derivative is:
1. The **slope** of the tangent line to the graph of $f$ at $x$.
2. The **rate of change**: if $x$ increases by a small amount $\Delta x$, then $f$ changes by approximately $f'(x) \cdot \Delta x$.
3. The **sensitivity**: how much the output responds to a perturbation of the input.

**Example.** For $f(x) = x^2$:

$$f'(x) = \lim_{\epsilon \to 0} \frac{(x + \epsilon)^2 - x^2}{\epsilon} = \lim_{\epsilon \to 0} \frac{2x\epsilon + \epsilon^2}{\epsilon} = \lim_{\epsilon \to 0} (2x + \epsilon) = 2x$$

At $x = 3$: $f'(3) = 6$. A small increase of $0.01$ in $x$ causes $f$ to increase by approximately $6 \times 0.01 = 0.06$. Verification: $f(3.01) = 9.0601$, and $f(3) = 9$, so the actual change is $0.0601$ — close to $0.06$.

> **Connecting to ML.** In gradient descent, $f$ is the loss and $x$ is a parameter. $f'(x) > 0$ means increasing $x$ increases the loss, so you should *decrease* $x$. $f'(x) < 0$ means increasing $x$ decreases the loss, so you should *increase* $x$. The update rule $x \leftarrow x - \eta f'(x)$ encodes precisely this logic.

---

### 3.2 Differentiation Rules

These rules must be second nature before [Week 01](../../02_fundamentals/week01_optimization/theory.md). Each is presented with its ML context.

| Rule        | Formula                                        | Example                                  |
| ----------- | ---------------------------------------------- | ---------------------------------------- |
| Constant    | $\frac{d}{dx}[c] = 0$                          | Bias term doesn't affect weight gradient |
| Power       | $\frac{d}{dx}[x^n] = nx^{n-1}$                 | Derivatives of polynomial features       |
| Sum         | $\frac{d}{dx}[f + g] = f' + g'$                | Loss = data term + regularisation term   |
| Product     | $\frac{d}{dx}[fg] = f'g + fg'$                 | Less common; appears in some derivations |
| Chain       | $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$ | Fundamental to backpropagation           |
| Exponential | $\frac{d}{dx}[e^x] = e^x$                      | Softmax, Gaussian PDF                    |
| Logarithm   | $\frac{d}{dx}[\ln x] = \frac{1}{x}$            | Log-likelihood, cross-entropy loss       |

**Special ML functions and their derivatives:**

| Function | $f(x)$                                         | $f'(x)$                                            | Where it appears                 |
| -------- | ---------------------------------------------- | -------------------------------------------------- | -------------------------------- |
| Sigmoid  | $\sigma(x) = \frac{1}{1 + e^{-x}}$             | $\sigma(x)(1 - \sigma(x))$                         | Logistic regression ([Week 03](../../02_fundamentals/week03_linear_models/theory.md#7-logistic-regression))    |
| Tanh     | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$                                   | RNN activations ([Week 17](../../06_sequence_models/week17_attention/theory.md#3-attention-as-soft-lookup))        |
| ReLU     | $\max(0, x)$                                   | $\begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$ | Default NN activation ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#32-activation-functions)+) |

> **Note on ReLU.** The derivative is undefined at $x = 0$. In practice, this is irrelevant: the probability of landing exactly at $x = 0$ during floating-point computation is negligible. By convention, $f'(0) = 0$.

---

### 3.3 Partial Derivatives and the Gradient

When a function depends on multiple variables $f : \mathbb{R}^d \to \mathbb{R}$, the **partial derivative** with respect to $w_j$ treats all other variables as constants:

$$\frac{\partial f}{\partial w_j} = \lim_{\epsilon \to 0} \frac{f(w_1, \ldots, w_j + \epsilon, \ldots, w_d) - f(w_1, \ldots, w_j, \ldots, w_d)}{\epsilon}$$

**The gradient** collects all partial derivatives into a vector:

$$\nabla_{\mathbf{w}} f = \begin{bmatrix} \frac{\partial f}{\partial w_1} \\ \frac{\partial f}{\partial w_2} \\ \vdots \\ \frac{\partial f}{\partial w_d} \end{bmatrix} \in \mathbb{R}^d$$

| Symbol                            | Meaning                                                |
| --------------------------------- | ------------------------------------------------------ |
| $\frac{\partial f}{\partial w_j}$ | Partial derivative of $f$ with respect to $w_j$        |
| $\nabla_{\mathbf{w}} f$           | Gradient of $f$: the vector of all partial derivatives |

**Fundamental property:** the gradient points in the direction of **steepest ascent** of $f$. Its magnitude $\|\nabla f\|$ is the rate of increase in that direction.

> **Proof sketch.** By the first-order Taylor expansion, $f(\mathbf{w} + \boldsymbol{\delta}) \approx f(\mathbf{w}) + \nabla f^\top \boldsymbol{\delta}$. The inner product $\nabla f^\top \boldsymbol{\delta}$ is maximised (for fixed $\|\boldsymbol{\delta}\|$) when $\boldsymbol{\delta}$ is parallel to $\nabla f$ (by the Cauchy–Schwarz inequality). Hence $\nabla f$ points in the direction of greatest increase.

**Worked example.** Consider the MSE loss for a 2-parameter linear model:

$$\mathcal{L}(w_1, w_2) = \frac{1}{n}\sum_{i=1}^{n}(y_i - w_1 x_{i1} - w_2 x_{i2})^2$$

Let $r_i = y_i - w_1 x_{i1} - w_2 x_{i2}$ (the residual for sample $i$). Then:

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{1}{n}\sum_{i=1}^{n} 2 r_i \cdot (-x_{i1}) = -\frac{2}{n}\sum_{i=1}^{n} r_i \, x_{i1}$$

$$\frac{\partial \mathcal{L}}{\partial w_2} = -\frac{2}{n}\sum_{i=1}^{n} r_i \, x_{i2}$$

In matrix form (dropping the factor of 2 by convention or absorbing it into the learning rate):

$$\nabla_{\mathbf{w}} \mathcal{L} = -\frac{2}{n} X^\top (\mathbf{y} - X\mathbf{w}) = -\frac{2}{n} X^\top \mathbf{r}$$

| Symbol              | Meaning                                                              |
| ------------------- | -------------------------------------------------------------------- |
| $r_i$               | Residual: difference between true and predicted value for sample $i$ |
| $\mathbf{r}$        | Vector of all residuals                                              |
| $X^\top \mathbf{r}$ | Matrix-vector product that sums contributions from all samples       |

> **Intuition.** The gradient of the MSE loss is proportional to the correlation between the residuals $\mathbf{r}$ (how wrong the model is) and the features $X$. If a feature is positively correlated with the error, the corresponding weight should increase; if negatively correlated, it should decrease. This is exactly what gradient descent does.

---

### 3.4 The Jacobian and the Hessian

#### The Jacobian

For a vector-valued function $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** is the $m \times n$ matrix of all first-order partial derivatives:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

Row $i$ is the gradient of $f_i$. The Jacobian generalises the gradient (a gradient is a Jacobian for $m = 1$) and is the fundamental object in the backpropagation algorithm ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#74-he-kaiming-initialisation)).

#### The Hessian

For a scalar-valued function $f : \mathbb{R}^d \to \mathbb{R}$, the **Hessian** is the $d \times d$ matrix of second-order partial derivatives:

$$H = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial w_1^2} & \frac{\partial^2 f}{\partial w_1 \partial w_2} & \cdots \\ \frac{\partial^2 f}{\partial w_2 \partial w_1} & \frac{\partial^2 f}{\partial w_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

The Hessian is symmetric (by Schwarz's theorem: $\frac{\partial^2 f}{\partial w_i \partial w_j} = \frac{\partial^2 f}{\partial w_j \partial w_i}$, assuming $f$ is twice continuously differentiable).

| Hessian property                             | Geometric meaning                                     |
| -------------------------------------------- | ----------------------------------------------------- |
| PD at $\mathbf{w}^*$ (all eigenvalues $> 0$) | $\mathbf{w}^*$ is a **strict local minimum**          |
| PSD at $\mathbf{w}^*$                        | $\mathbf{w}^*$ is a **local minimum** (possibly flat) |
| Indefinite (mixed signs)                     | $\mathbf{w}^*$ is a **saddle point**                  |
| Negative definite                            | $\mathbf{w}^*$ is a **strict local maximum**          |

> **ML relevance.** The loss surface of a neural network has many saddle points (more than local minima, in high dimensions). The Hessian's eigenvalues at a critical point reveal whether gradient descent is stuck at a minimum or a saddle. Second-order optimisers (Newton's method) use the Hessian to take better-informed steps, but computing the full Hessian for a neural network with millions of parameters is prohibitive.

> **Condition number revisited.** The condition number of the Hessian at the optimum, $\kappa(H) = \lambda_{\max} / \lambda_{\min}$, determines how "elongated" the loss valley is. Gradient descent converges slowly when $\kappa$ is large (ill-conditioned problems). Adaptive optimisers like Adam ([Week 02](../../02_fundamentals/week02_advanced_optimizers/theory.md#7-adam-combining-momentum-and-adaptivity)) implicitly approximate the Hessian's diagonal to handle this.

**Example.** For $f(w_1, w_2) = w_1^2 + 10 w_2^2$ (a stretched quadratic):

$$H = \begin{bmatrix} 2 & 0 \\ 0 & 20 \end{bmatrix}$$

The eigenvalues are $2$ and $20$, so $\kappa = 10$. Gradient descent will oscillate along $w_2$ (steep) and converge slowly along $w_1$ (shallow). This is explored in [Week 01](../../02_fundamentals/week01_optimization/theory.md#11-notebook-reference-guide)'s notebook.

---

### 3.5 Taylor Expansion

The Taylor expansion approximates a smooth function around a point $\mathbf{w}_0$:

**Univariate (scalar):**

$$f(w) = f(w_0) + f'(w_0)(w - w_0) + \frac{1}{2}f''(w_0)(w - w_0)^2 + \mathcal{O}((w - w_0)^3)$$

**Multivariate (vector):**

$$f(\mathbf{w}) \approx f(\mathbf{w}_0) + \nabla f(\mathbf{w}_0)^\top (\mathbf{w} - \mathbf{w}_0) + \frac{1}{2}(\mathbf{w} - \mathbf{w}_0)^\top H(\mathbf{w}_0)(\mathbf{w} - \mathbf{w}_0)$$

| Term                                                                        | Name         | Interpretation                            |
| --------------------------------------------------------------------------- | ------------ | ----------------------------------------- |
| $f(\mathbf{w}_0)$                                                           | Zeroth order | The function value at the expansion point |
| $\nabla f^\top (\mathbf{w} - \mathbf{w}_0)$                                 | First order  | Linear approximation (gradient-based)     |
| $\frac{1}{2}(\mathbf{w} - \mathbf{w}_0)^\top H (\mathbf{w} - \mathbf{w}_0)$ | Second order | Curvature correction                      |

> **Why this matters.**
> - The **first-order** approximation justifies gradient descent: neighbouring the current point, the loss is approximately linear, so stepping against the gradient reduces it.
> - The **second-order** approximation justifies Newton's method: $\mathbf{w}_{\text{new}} = \mathbf{w}_0 - H^{-1} \nabla f$, which accounts for curvature and converges faster.
> - The quality of the approximation depends on how far $\mathbf{w}$ is from $\mathbf{w}_0$ — this is why **small learning rates** work better (they keep steps within the region where the linear approximation is valid).

---

### 3.6 The Chain Rule in Depth

The chain rule is the single most important calculus result for machine learning. It enables **backpropagation** — the algorithm that computes gradients through multi-layer networks.

**Univariate chain rule.** If $f(x) = g(h(x))$, then:

$$\frac{df}{dx} = \frac{dg}{dh} \cdot \frac{dh}{dx}$$

**Multivariate chain rule.** If $f(\mathbf{x}) = g(\mathbf{h}(\mathbf{x}))$, where $\mathbf{h} : \mathbb{R}^n \to \mathbb{R}^m$ and $g : \mathbb{R}^m \to \mathbb{R}$, then:

$$\frac{\partial f}{\partial x_j} = \sum_{k=1}^{m} \frac{\partial g}{\partial h_k} \cdot \frac{\partial h_k}{\partial x_j}$$

In matrix form, this is:

$$\nabla_{\mathbf{x}} f = J_{\mathbf{h}}^\top \nabla_{\mathbf{h}} g$$

where $J_{\mathbf{h}}$ is the Jacobian of $\mathbf{h}$ with respect to $\mathbf{x}$.

> **Connection to neural networks.** A neural network is a composition of $L$ layers: $f = g_L \circ g_{L-1} \circ \cdots \circ g_1$. By the chain rule, the gradient of the loss with respect to the parameters of layer $\ell$ involves the product of Jacobians from all subsequent layers. This product is computed efficiently by **backpropagation**, which caches intermediate results and propagates gradients backward through the network. This is the subject of [Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#74-he-kaiming-initialisation).

**Worked example.** Consider a single-neuron model: $\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x} + b)$, where $\sigma$ is the sigmoid function, and the loss is binary cross-entropy $\mathcal{L} = -[y \ln \hat{y} + (1 - y) \ln(1 - \hat{y})]$.

Define intermediate variables:
- $z = \mathbf{w}^\top \mathbf{x} + b$ (the pre-activation, a scalar)
- $\hat{y} = \sigma(z)$ (the prediction)

Then:

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$

Computing each piece:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}$$

$$\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

Combining (and simplifying — this is a classic result):

$$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$

Then:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}} = (\hat{y} - y)\mathbf{x}$$

$$\frac{\partial \mathcal{L}}{\partial b} = \hat{y} - y$$

> **Elegant result.** The gradient of the cross-entropy loss for logistic regression is simply $(\hat{y} - y)\mathbf{x}$ — the prediction error scaled by the input. This is identical in form to the gradient of MSE for linear regression (up to a constant), which is not a coincidence — both arise from the same exponential family structure ([Week 07](../../03_probability/week07_likelihood/theory.md#51-gaussian-noise-mse)).

---

### 3.7 Vector Calculus Identities for ML

These identities are used without derivation in many ML textbooks. Knowing them removes the magic.

Let $\mathbf{a}, \mathbf{x} \in \mathbb{R}^d$, $A \in \mathbb{R}^{d \times d}$ (symmetric), and $f : \mathbb{R}^d \to \mathbb{R}$.

| Expression                                      | Gradient $\nabla_{\mathbf{x}}$ | Where it appears                  |
| ----------------------------------------------- | ------------------------------ | --------------------------------- |
| $\mathbf{a}^\top \mathbf{x}$                    | $\mathbf{a}$                   | Linear model prediction           |
| $\mathbf{x}^\top \mathbf{x} = \|\mathbf{x}\|^2$ | $2\mathbf{x}$                  | L2 regularisation                 |
| $\mathbf{x}^\top A \mathbf{x}$                  | $2A\mathbf{x}$                 | Quadratic forms (normal equation) |
| $\mathbf{a}^\top A \mathbf{x}$                  | $A^\top \mathbf{a}$            | Mixed terms in loss               |

**Derivation of $\nabla_{\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = 2A\mathbf{x}$ (for symmetric $A$).**

Expand: $\mathbf{x}^\top A \mathbf{x} = \sum_i \sum_j a_{ij} x_i x_j$.

Take the partial derivative with respect to $x_k$:

$$\frac{\partial}{\partial x_k} \sum_i \sum_j a_{ij} x_i x_j = \sum_j a_{kj} x_j + \sum_i a_{ik} x_i = (A\mathbf{x})_k + (A^\top \mathbf{x})_k$$

For symmetric $A$ ($A = A^\top$): $= 2(A\mathbf{x})_k$.

Hence $\nabla_{\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = 2A\mathbf{x}$.

> **This identity is used in [Week 03](../../02_fundamentals/week03_linear_models/theory.md#33-the-normal-equations-closed-form-solution)** to derive the normal equation for linear regression. The MSE loss in matrix form is $\mathcal{L}(\mathbf{w}) = \frac{1}{n}(\mathbf{y} - X\mathbf{w})^\top(\mathbf{y} - X\mathbf{w})$. Expanding and differentiating using the identities above yields $\nabla \mathcal{L} = -\frac{2}{n}X^\top(\mathbf{y} - X\mathbf{w})$. Setting this to zero gives the normal equation.

---

### 3.8 Numerical Differentiation

Analytical derivatives can be verified numerically using **finite differences**:

**Forward difference:**

$$f'(x) \approx \frac{f(x + \epsilon) - f(x)}{\epsilon}$$

**Central difference (more accurate):**

$$f'(x) \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

The central difference has error $\mathcal{O}(\epsilon^2)$ compared to $\mathcal{O}(\epsilon)$ for the forward difference, because the odd-order Taylor terms cancel.

> **Recommended value:** $\epsilon \approx 10^{-5}$. Too small and floating-point cancellation dominates; too large and the approximation is poor.

```python
def numerical_gradient(f, x, eps=1e-5):
    """Central-difference gradient for a scalar function f: R^d -> R."""
    grad = np.zeros_like(x)
    for j in range(len(x)):
        x_plus = x.copy(); x_plus[j] += eps
        x_minus = x.copy(); x_minus[j] -= eps
        grad[j] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Example: verify gradient of f(w1, w2) = w1^2 + 3*w1*w2
def f(w):
    return w[0]**2 + 3*w[0]*w[1]

w = np.array([2.0, 1.0])
analytic_grad = np.array([2*w[0] + 3*w[1], 3*w[0]])   # [7.0, 6.0]
numeric_grad = numerical_gradient(f, w)

print("Analytic:", analytic_grad)
print("Numeric: ", numeric_grad)
print("Match:", np.allclose(analytic_grad, numeric_grad))   # True
```

> **Suggested experiment.** Implement this gradient checker and use it throughout [Weeks 01](../../02_fundamentals/week01_optimization/theory.md#4-gradient-descent)–04 to verify your analytical gradient derivations. It is the single most effective debugging tool in ML. Every time you derive a gradient by hand and code it, check it against the numerical gradient. If they disagree, the bug is in your derivation or implementation.

> **How it connects to autograd.** PyTorch's automatic differentiation ([Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md#3-automatic-differentiation-autograd)) computes exact analytical gradients using the chain rule, applied algorithmically. The numerical gradient check remains useful as an independent verification even when using autograd.

---

## 4. Part III — Probability and Statistics

Probability theory is the mathematical framework for reasoning about uncertainty. In ML, it provides:
1. The **justification for loss functions** — every standard loss is a negative log-likelihood under some probabilistic model ([Week 07](../../03_probability/week07_likelihood/theory.md#5-the-mleloss-function-connection)).
2. The **language for generalisation** — overfitting is a probabilistic phenomenon (the model fits noise drawn from a distribution).
3. The **tools for uncertainty quantification** — predicting not just $\hat{y}$ but a distribution over possible $y$ values ([Week 08](../../03_probability/week08_uncertainty/theory.md#3-two-kinds-of-uncertainty)).

### 4.1 Probability Axioms

A **probability space** consists of:
- A **sample space** $\Omega$: the set of all possible outcomes.
- An **event space** $\mathcal{F}$: a collection of subsets of $\Omega$ (the "events" we can assign probabilities to).
- A **probability measure** $P : \mathcal{F} \to [0, 1]$ satisfying:
  1. $P(\Omega) = 1$ (something always happens).
  2. $P(A) \geq 0$ for all $A \in \mathcal{F}$ (probabilities are non-negative).
  3. For mutually exclusive events $A_1, A_2, \ldots$: $P\left(\bigcup_{i} A_i\right) = \sum_{i} P(A_i)$ (probabilities of disjoint events add up).

From these axioms, one can derive: $P(\bar{A}) = 1 - P(A)$, $P(\emptyset) = 0$, and $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.

> **In ML terms.** The sample space might be "all possible datasets from the true data-generating process." The probability measure encodes our belief about what datasets are likely. These concepts remain mostly in the background, but they provide the rigorous foundation for everything in [Weeks 07](../../03_probability/week07_likelihood/theory.md#3-likelihood-from-data-to-models)–10.

---

### 4.2 Conditional Probability and Bayes' Theorem

**Conditional probability.** The probability of $A$ given that $B$ has occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Independence.** Events $A$ and $B$ are independent if $P(A \cap B) = P(A) P(B)$, equivalently $P(A \mid B) = P(A)$.

**Bayes' theorem.** Relating the "forward" probability $P(B \mid A)$ to the "inverse" probability $P(A \mid B)$:

$$P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}$$

| Term          | Name                | Meaning                                          |
| ------------- | ------------------- | ------------------------------------------------ |
| $P(A \mid B)$ | Posterior           | Updated belief about $A$ after observing $B$     |
| $P(B \mid A)$ | Likelihood          | How likely $B$ is, assuming $A$ is true          |
| $P(A)$        | Prior               | Belief about $A$ before observing $B$            |
| $P(B)$        | Evidence (marginal) | Overall probability of $B$, normalising constant |

> **ML interpretation.** Let $A = \theta$ (model parameters) and $B = \mathcal{D}$ (observed data):
>
> $$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) \, P(\theta)}{P(\mathcal{D})}$$
>
> This is the foundation of **Bayesian inference** ([Week 08](../../03_probability/week08_uncertainty/theory.md#7-bayesian-inference)):
> - $P(\mathcal{D} \mid \theta)$: the **likelihood** — how well the parameters explain the data.
> - $P(\theta)$: the **prior** — our initial belief about good parameter values (e.g., small weights → L2 regularisation is a Gaussian prior).
> - $P(\theta \mid \mathcal{D})$: the **posterior** — the updated belief after seeing data.
>
> **Maximum Likelihood Estimation (MLE)** ignores the prior and maximises $P(\mathcal{D} \mid \theta)$ alone. **Maximum A Posteriori (MAP)** estimation includes the prior and maximises $P(\theta \mid \mathcal{D})$. MAP with a Gaussian prior is equivalent to L2-regularised MLE (proved in [Week 07](../../03_probability/week07_likelihood/theory.md#9-from-mle-to-map-the-bridge-to-regularisation)).

**Example.** A medical test for a disease is 99% accurate (sensitivity = specificity = 0.99). The disease prevalence is 0.1%. You test positive. What is the probability you actually have the disease?

$$P(\text{disease} \mid \text{positive}) = \frac{P(\text{positive} \mid \text{disease}) \, P(\text{disease})}{P(\text{positive})}$$

$$= \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.01 \times 0.999} = \frac{0.00099}{0.00099 + 0.00999} \approx 0.09$$

Only 9% — despite a 99% accurate test! The low prior (0.1% prevalence) dominates. This is the **base rate fallacy**, and it illustrates why priors matter in probabilistic reasoning.

---

### 4.3 Random Variables and Distributions

A **random variable** $X$ is a function that maps outcomes in the sample space to real numbers. It represents a quantity whose value is uncertain.

**Discrete random variable.** Takes values from a countable set. Characterised by a **probability mass function (PMF)** $P(X = k)$, where $\sum_k P(X = k) = 1$.

**Continuous random variable.** Takes values in $\mathbb{R}$ (or a subset). Characterised by a **probability density function (PDF)** $p(x)$, where:

$$P(a \leq X \leq b) = \int_a^b p(x) \, dx, \qquad \int_{-\infty}^{\infty} p(x) \, dx = 1$$

> **Important subtlety.** For a continuous random variable, $P(X = x) = 0$ for any specific value $x$. Only probabilities over intervals are meaningful. The PDF $p(x)$ is a density (probability per unit length), not a probability. It can exceed 1.

**Cumulative distribution function (CDF).** For any random variable:

$$F(x) = P(X \leq x)$$

The CDF is non-decreasing, $F(-\infty) = 0$, $F(\infty) = 1$. For continuous RVs, $p(x) = F'(x)$.

**Joint distribution.** Two random variables $X, Y$ have a joint distribution $p(x, y)$. The **marginal** distributions are obtained by integration (or summation):

$$p(x) = \int p(x, y) \, dy \qquad (\text{continuous case})$$

Two variables are **independent** if $p(x, y) = p(x) \, p(y)$.

> **ML connection.** The assumption that data points $(\mathbf{x}_i, y_i)$ are drawn **independently and identically distributed (i.i.d.)** from a joint distribution $p(\mathbf{x}, y)$ is the foundational assumption of nearly all supervised learning. The i.i.d. assumption means the likelihood of the entire dataset factorises: $P(\mathcal{D} \mid \theta) = \prod_{i=1}^{n} p(y_i \mid \mathbf{x}_i; \theta)$, which makes the log-likelihood a sum — the basis of all gradient-based training.

---

### 4.4 Expectation, Variance, and Covariance

#### Expectation

The **expected value** (mean) of a random variable is its "average" over the distribution:

$$\mathbb{E}[X] = \begin{cases} \sum_x x \, P(X = x) & \text{(discrete)} \\ \int x \, p(x) \, dx & \text{(continuous)} \end{cases}$$

| Property                 | Formula                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| Linearity                | $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ (always, even if $X, Y$ dependent) |
| Constant                 | $\mathbb{E}[c] = c$                                                                        |
| Product (if independent) | $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$                                              |

> **In practice**, the expected value is estimated by the **sample mean**: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$.

#### Variance

The **variance** measures the spread of a distribution around its mean:

$$\text{Var}(X) = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

| Symbol                          | Meaning                                             |
| ------------------------------- | --------------------------------------------------- |
| $\text{Var}(X)$ or $\sigma^2$   | Variance — expected squared deviation from the mean |
| $\sigma = \sqrt{\text{Var}(X)}$ | Standard deviation — same units as $X$              |

Properties:
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$ (shifting doesn't change spread; scaling does)
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$
- If $X, Y$ independent: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

> **In practice**, the sample variance is $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$. The $n-1$ denominator (Bessel's correction) makes it an unbiased estimator of the population variance.

#### Covariance and Correlation

**Covariance** measures how two variables co-vary:

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

- $\text{Cov}(X, Y) > 0$: $X$ and $Y$ tend to be large or small together.
- $\text{Cov}(X, Y) < 0$: when one is large, the other tends to be small.
- $\text{Cov}(X, Y) = 0$: no linear relationship (but possibly nonlinear!).

**Correlation** (Pearson) normalises covariance to $[-1, 1]$:

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Covariance matrix.** For a random vector $\mathbf{X} \in \mathbb{R}^d$:

$$\Sigma = \text{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top] \in \mathbb{R}^{d \times d}$$

where $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$. The entry $\Sigma_{jk} = \text{Cov}(X_j, X_k)$. The diagonal entries are variances: $\Sigma_{jj} = \text{Var}(X_j)$.

Properties:
- $\Sigma$ is **symmetric**: $\Sigma = \Sigma^\top$.
- $\Sigma$ is **positive semi-definite**: $\mathbf{v}^\top \Sigma \mathbf{v} \geq 0$ for all $\mathbf{v}$.
- The eigenvectors of $\Sigma$ are the principal components (PCA, [Week 04](../../02_fundamentals/week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view)).

```python
# Compute sample covariance matrix
X = np.random.randn(100, 3)          # 100 samples, 3 features
X[:, 2] = X[:, 0] + 0.5 * X[:, 1]   # make feature 3 correlated
cov_matrix = np.cov(X.T)             # shape (3, 3)
print(cov_matrix)
# Off-diagonal entries reveal correlations
```

> **Suggested experiment.** Generate a 2D dataset from a multivariate Gaussian with known covariance matrix $\Sigma = \begin{bmatrix} 1 & 0.8 \\ 0.8 & 1 \end{bmatrix}$. Scatter-plot the data, compute the sample covariance matrix, and compare it to the true $\Sigma$. Increase $n$ and observe the estimate converging to the truth.

---

### 4.5 Important Distributions

The following distributions appear repeatedly in ML. For each, we give the PDF/PMF, parameters, key properties, and ML relevance.

#### Bernoulli Distribution

$$P(X = k) = p^k (1-p)^{1-k}, \qquad k \in \{0, 1\}$$

| Parameter      | Meaning                          |
| -------------- | -------------------------------- |
| $p \in [0, 1]$ | Probability of success ($X = 1$) |

- $\mathbb{E}[X] = p$, $\text{Var}(X) = p(1 - p)$.
- **ML usage**: binary classification labels. The model's output $\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x} + b)$ is interpreted as $P(y = 1 \mid \mathbf{x})$.

#### Categorical Distribution

Generalisation to $K$ classes: $P(X = k) = p_k$, where $\sum_k p_k = 1$.

- **ML usage**: multi-class classification. The softmax output is a categorical distribution over classes.

#### Gaussian (Normal) Distribution

$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

| Parameter  | Meaning                               |
| ---------- | ------------------------------------- |
| $\mu$      | Mean — centre of the distribution     |
| $\sigma^2$ | Variance — spread of the distribution |
| $\sigma$   | Standard deviation                    |

Notation: $X \sim \mathcal{N}(\mu, \sigma^2)$.

- $\mathbb{E}[X] = \mu$, $\text{Var}(X) = \sigma^2$.
- The **68-95-99.7 rule**: approximately 68%, 95%, and 99.7% of values fall within 1, 2, and 3 standard deviations of the mean.
- The **standard normal** has $\mu = 0$, $\sigma^2 = 1$.

> **ML significance.** Assuming Gaussian noise ($y = f_\theta(\mathbf{x}) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$) and maximising the likelihood leads directly to the MSE loss. This is proved in [Week 07](../../03_probability/week07_likelihood/theory.md#51-gaussian-noise-mse).

#### Multivariate Gaussian

For a random vector $\mathbf{X} \in \mathbb{R}^d$:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

| Parameter          | Shape        | Meaning                               |
| ------------------ | ------------ | ------------------------------------- |
| $\boldsymbol{\mu}$ | $d \times 1$ | Mean vector                           |
| $\Sigma$           | $d \times d$ | Covariance matrix (PSD)               |
| $                  | \Sigma       | $                                     | scalar | Determinant of $\Sigma$ |
| $\Sigma^{-1}$      | $d \times d$ | Precision matrix (inverse covariance) |

The contours of constant probability density are **ellipsoids**, with axes aligned to the eigenvectors of $\Sigma$ and radii proportional to $\sqrt{\lambda_i}$.

> **ML relevance.** The multivariate Gaussian appears in:
> - **PCA** ([Week 04](../../02_fundamentals/week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view)): modelling data as drawn from a Gaussian → the eigenvectors of $\Sigma$ are the principal components.
> - **Gaussian Mixture Models** ([Week 05](../../02_fundamentals/week05_clustering/theory.md#6-gaussian-mixture-models-soft-clustering)): clustering via mixtures of multivariate Gaussians.
> - **Bayesian inference** ([Week 08](../../03_probability/week08_uncertainty/theory.md#72-prior-likelihood-posterior)): Gaussian priors on parameters.
> - **Gaussian Processes** ([Week 10](../../03_probability/week10_surrogate_models/theory.md#3-gaussian-processes-intuition)): defining distributions over functions.

```python
from scipy.stats import multivariate_normal, norm

# Univariate Gaussian
x = np.linspace(-4, 4, 200)
for mu, sigma in [(0, 1), (0, 2), (2, 1)]:
    plt.plot(x, norm.pdf(x, mu, sigma), label=f'μ={mu}, σ={sigma}')
plt.legend(); plt.title('Gaussian PDF'); plt.xlabel('x'); plt.ylabel('p(x)')
plt.show()
```

> **Suggested experiment.** Plot Gaussians with different $\mu$ and $\sigma$. Observe that $\mu$ shifts the centre and $\sigma$ controls the width. Then plot a 2D Gaussian with $\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ (circular contours) and $\Sigma = \begin{bmatrix} 2 & 1.5 \\ 1.5 & 2 \end{bmatrix}$ (elongated, tilted ellipses). The tilt direction is the first eigenvector of $\Sigma$.

#### Uniform Distribution

$$p(x) = \frac{1}{b - a}, \quad x \in [a, b]$$

- $\mathbb{E}[X] = \frac{a + b}{2}$, $\text{Var}(X) = \frac{(b-a)^2}{12}$.
- Used for random initialisation of parameters and for generating random permutations (stochastic gradient descent).

---

### 4.6 The Law of Large Numbers and the Central Limit Theorem

These two theorems justify virtually all practical statistics and estimation in ML.

#### Law of Large Numbers (LLN)

If $X_1, X_2, \ldots$ are i.i.d. with mean $\mu$, then:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{n \to \infty} \mu$$

(convergence in probability). The sample mean converges to the true mean as sample size grows.

> **ML interpretation.** The empirical risk (training loss averaged over $n$ samples) converges to the true risk (expected loss over the data distribution) as $n \to \infty$. This is why more data generally improves model quality: the training loss becomes a better proxy for the true loss.

#### Central Limit Theorem (CLT)

If $X_1, X_2, \ldots$ are i.i.d. with mean $\mu$ and variance $\sigma^2$, then the standardised sample mean converges in distribution to a standard normal:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

Equivalently: $\bar{X}_n \approx \mathcal{N}(\mu, \sigma^2 / n)$ for large $n$.

> **ML interpretation.**
> - The CLT explains why SGD (Stochastic Gradient Descent) works: even though individual mini-batch gradient estimates are noisy, the mean of many estimates converges to the true gradient, with error proportional to $1/\sqrt{n_{\text{batch}}}$.
> - It justifies constructing confidence intervals for model performance estimates: if a metric is averaged over $n$ test samples, its standard error is $\sigma / \sqrt{n}$.

```python
# CLT demonstration: average of many non-Gaussian samples approaches Gaussian
rng = np.random.default_rng(42)
sample_means = []
for _ in range(10000):
    # Each sample: 50 draws from a uniform(0,1) — distinctly non-Gaussian
    sample = rng.uniform(0, 1, size=50)
    sample_means.append(sample.mean())

plt.hist(sample_means, bins=50, density=True, alpha=0.7, label='Sample means')
# Overlay the theoretical Gaussian
x = np.linspace(0.3, 0.7, 200)
mu_theory, sigma_theory = 0.5, np.sqrt(1/12) / np.sqrt(50)
plt.plot(x, norm.pdf(x, mu_theory, sigma_theory), 'r-', label='Gaussian approx')
plt.legend(); plt.title('CLT: Means of Uniform samples → Gaussian')
plt.show()
```

---

### 4.7 Maximum Likelihood Estimation — Preview

This section previews the key idea of [Week 07](../../03_probability/week07_likelihood/theory.md#5-the-mleloss-function-connection), because it provides the probabilistic motivation for loss functions used from [Week 01](../../02_fundamentals/week01_optimization/theory.md#3-loss-functions) onward.

**Setup.** Assume data $\{y_i\}$ are generated by a model with parameters $\theta$, corrupted by noise:

$$y_i = f_\theta(\mathbf{x}_i) + \epsilon_i, \qquad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

This means $y_i \mid \mathbf{x}_i \sim \mathcal{N}(f_\theta(\mathbf{x}_i), \sigma^2)$.

**The likelihood** is the probability of observing the data given the parameters:

$$P(\mathcal{D} \mid \theta) = \prod_{i=1}^{n} p(y_i \mid \mathbf{x}_i; \theta) = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y_i - f_\theta(\mathbf{x}_i))^2}{2\sigma^2}\right)$$

**The log-likelihood** (easier to work with — products become sums):

$$\ell(\theta) = \log P(\mathcal{D} \mid \theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(\mathbf{x}_i))^2$$

**Maximising** the log-likelihood is equivalent to **minimising**:

$$\sum_{i=1}^{n}(y_i - f_\theta(\mathbf{x}_i))^2$$

This is the **mean squared error loss** (up to constant factors).

> **Key insight.** MSE is not an arbitrary choice — it is the mathematically justified loss when you assume Gaussian noise. Under other noise models:
> - **Laplace noise** → minimise the Mean Absolute Error (MAE): $\sum |y_i - \hat{y}_i|$, which is more robust to outliers.
> - **Bernoulli model** (binary classification) → minimise the Binary Cross-Entropy.
> - **Categorical model** (multi-class) → minimise the Categorical Cross-Entropy.
>
> Every standard loss function in ML is a negative log-likelihood under some probabilistic model. This unifying perspective is the subject of [Week 07](../../03_probability/week07_likelihood/theory.md#5-the-mleloss-function-connection).

---

### 4.8 Information Theory Basics

Information theory provides an alternative lens on the same loss functions, and is essential for understanding cross-entropy and KL divergence.

#### Entropy

The **entropy** of a discrete distribution $P$ measures its **uncertainty** or **information content**:

$$H(P) = -\sum_{k} p_k \log p_k$$

| Property        | Value                                                                       |
| --------------- | --------------------------------------------------------------------------- |
| Minimum entropy | $H = 0$ when one outcome has probability 1 (no uncertainty)                 |
| Maximum entropy | $H = \log K$ when all $K$ outcomes are equally likely (maximum uncertainty) |

> **Intuition.** Entropy measures how "surprising" the outcomes are on average. A fair coin (50/50) has maximum entropy for two outcomes: $H = -0.5 \log 0.5 - 0.5 \log 0.5 = \log 2 \approx 0.693$ nats. A biased coin (99/1) has low entropy: outcomes are predictable.

#### Cross-Entropy

The **cross-entropy** between a true distribution $P$ and a predicted distribution $Q$:

$$H(P, Q) = -\sum_{k} p_k \log q_k$$

This is the loss function used in classification: $P$ is the one-hot label and $Q$ is the model's predicted probabilities (softmax output).

- $H(P, Q) \geq H(P)$ (cross-entropy is never less than entropy).
- $H(P, Q) = H(P)$ if and only if $Q = P$ (the model perfectly predicts the true distribution).

#### Kullback-Leibler Divergence

$$D_{\text{KL}}(P \| Q) = H(P, Q) - H(P) = \sum_{k} p_k \log \frac{p_k}{q_k} \geq 0$$

The KL divergence measures how much $Q$ differs from $P$. It is zero if and only if $P = Q$. Minimising cross-entropy $H(P, Q)$ is equivalent to minimising $D_{\text{KL}}(P \| Q)$ since $H(P)$ is a constant with respect to the model.

> **ML usage.** Minimising the cross-entropy loss in classification is equivalent to minimising the KL divergence between the true label distribution and the model's predicted distribution. This connection is formalised in [Week 07](../../03_probability/week07_likelihood/theory.md#52-bernoulli-noise-binary-cross-entropy).

---

## 5. Part IV — Data Representation and Exploratory Analysis

### 5.1 Data as Matrices

A tabular dataset with $n$ samples and $d$ features is represented as a matrix $X \in \mathbb{R}^{n \times d}$:

$$X = \begin{bmatrix} \text{---} \; \mathbf{x}_1^\top \; \text{---} \\ \text{---} \; \mathbf{x}_2^\top \; \text{---} \\ \vdots \\ \text{---} \; \mathbf{x}_n^\top \; \text{---} \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

Each row $\mathbf{x}_i^\top$ is a sample (data point). Each column $X_{:,j}$ is a feature (variable, attribute).

This matrix representation enables:
- **Vectorised computation**: $\hat{\mathbf{y}} = X\mathbf{w} + b$ computes all predictions simultaneously.
- **Statistical operations**: the mean vector $\bar{\mathbf{x}} = \frac{1}{n}\sum_i \mathbf{x}_i$ and covariance matrix $\Sigma = \frac{1}{n-1}(X - \bar{X})^\top(X - \bar{X})$ are both simple matrix expressions.

```python
import pandas as pd

# From pandas DataFrame to NumPy matrix
df = pd.DataFrame({'temp': [20, 25, 30], 'humidity': [40, 50, 60], 'demand': [100, 150, 200]})
X = df[['temp', 'humidity']].values    # shape (3, 2)
y = df['demand'].values                # shape (3,)
```

---

### 5.2 Feature Types and Encoding

Not all data is numeric. Understanding feature types determines preprocessing.

| Type                   | Examples                       | Encoding for ML                                   |
| ---------------------- | ------------------------------ | ------------------------------------------------- |
| **Continuous**         | Temperature, price, voltage    | Use directly (possibly standardise)               |
| **Discrete (ordinal)** | Education level (low/med/high) | Integer encoding ($0, 1, 2$) — preserves order    |
| **Discrete (nominal)** | Colour (red/blue/green)        | One-hot encoding: $[1,0,0]$, $[0,1,0]$, $[0,0,1]$ |
| **Binary**             | Yes/No, Male/Female            | $0$ or $1$                                        |
| **Text**               | Product reviews                | Tokenisation → embeddings ([Week 17](../../06_sequence_models/week17_attention/theory.md#3-attention-as-soft-lookup)–18)            |

**Standardisation (Z-score normalisation).**

$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

This transforms each feature to have mean $0$ and standard deviation $1$. It is critical when features have different scales (e.g., temperature in °C vs. pressure in kPa), because gradient-based optimisation is sensitive to scale mismatches — the condition number of $X^\top X$ is directly affected.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # mean 0, std 1 per feature
```

> **Suggested experiment.** In [Week 01](../../02_fundamentals/week01_optimization/theory.md#11-notebook-reference-guide)'s notebook, run gradient descent on raw features and on standardised features. Compare convergence speed (number of iterations to reach a given loss threshold). Standardisation typically reduces iterations by 5–50×.

---

### 5.3 The EDA Protocol

Before training any model, perform **Exploratory Data Analysis** to understand the data. This habit prevents silent failures.

**Step 1: Shape and types.**

```python
print(df.shape)        # (n, d)
print(df.dtypes)       # feature types
print(df.info())       # non-null counts, memory usage
```

**Step 2: Missing values.**

```python
print(df.isnull().sum())               # count NaNs per column
print(df.isnull().sum() / len(df))      # fraction missing
```

Strategies: drop rows (if few), impute with mean/median (if random), flag missingness as a feature (if informative).

**Step 3: Distribution of each feature.**

```python
df.hist(bins=30, figsize=(12, 8))
plt.tight_layout(); plt.show()
```

Look for: skewness (consider $\log$ transform), outliers (investigate cause), bimodal shapes (possibly mixed populations).

**Step 4: Correlations.**

```python
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.show()
```

Highly correlated features carry redundant information. Perfectly correlated features cause rank deficiency in $X^\top X$ (Section 2.5).

**Step 5: Target distribution.**

For regression: is the target skewed? Heavy-tailed? Consider transforming it ($\log$, Box-Cox).

For classification: are the classes balanced? If 99% of samples are class 0, a naive model predicting "always 0" has 99% accuracy but is useless. Use stratified splits and appropriate metrics (F1, AUC-ROC).

**Step 6: Temporal structure (if applicable).**

```python
df.set_index('date')['target'].plot()
plt.show()
```

Look for trends, seasonality, regime changes, and gaps. If temporal structure exists, **random train/test splitting is invalid** — use temporal splits ([Week 09](../../03_probability/week09_time_series/theory.md#22-seasonality)).

---

### 5.4 Common Pitfalls

| Pitfall                 | Consequence                                                                | Prevention                                                                   |
| ----------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Data leakage**        | Model uses information unavailable at prediction time → inflated metrics   | Audit feature availability; split data before any preprocessing              |
| **Target leakage**      | A feature that is a proxy for the target → perfect accuracy, useless model | Check correlations; understand the data-generating process                   |
| **Look-ahead bias**     | Using future data to predict the past (time series)                        | Always split temporally for time series                                      |
| **Survivor bias**       | Training only on successful cases → model cannot predict failure           | Ensure data includes failures                                                |
| **Scaling after split** | Fitting scaler on full data including test set → information leaks         | Fit scaler on training set only; transform test set with training statistics |

> **Rule of thumb.** Any statistic computed from data (mean, standard deviation, vocabulary) must be computed on the **training set only** and applied to validation/test sets. This ensures the model never sees test data, even indirectly.

---

## 6. Part V — Numerical Computing with NumPy

### 6.1 The Shape Discipline

The most common source of bugs in ML code is **shape mismatches**. Build the habit of annotating shapes.

```python
# Always know the shape of every array
X = np.random.randn(100, 5)       # (100, 5) — 100 samples, 5 features
w = np.random.randn(5)            # (5,)     — weight vector
b = 0.1                            # ()       — scalar bias

y_hat = X @ w + b                  # (100,)  — predictions for all samples
```

**The shape of common ML objects:**

| Object                               | Shape                             | Notation                                                |
| ------------------------------------ | --------------------------------- | ------------------------------------------------------- |
| Data matrix                          | $(n, d)$                          | $n$ samples, $d$ features                               |
| Weight vector                        | $(d,)$ or $(d, 1)$                | One weight per feature                                  |
| Weight matrix (NN)                   | $(d_{\text{in}}, d_{\text{out}})$ | Maps inputs to outputs                                  |
| Predictions                          | $(n,)$ or $(n, K)$                | One per sample (or $K$ class probabilities)             |
| Gradient of loss w.r.t. $\mathbf{w}$ | Same shape as $\mathbf{w}$        | The gradient always has the same shape as the parameter |

> **Golden rule.** The gradient of a loss with respect to a parameter always has the **same shape** as the parameter. If $\mathbf{w}$ is $(d,)$, then $\nabla_\mathbf{w} \mathcal{L}$ is $(d,)$. If $W$ is $(d_\text{in}, d_\text{out})$, then $\nabla_W \mathcal{L}$ is $(d_\text{in}, d_\text{out})$. This is a universal sanity check.

---

### 6.2 Broadcasting

NumPy's **broadcasting** allows operations between arrays of different shapes by automatically expanding dimensions.

**Rules:**
1. Compare shapes element-wise from the **trailing** (rightmost) dimension.
2. Dimensions are compatible if they are equal or one of them is 1.
3. Missing dimensions (shorter array) are treated as 1.

```python
X = np.random.randn(100, 5)    # (100, 5)
mu = X.mean(axis=0)            # (5,)     — mean of each feature

# Subtract the mean from each row: broadcasting (100, 5) - (5,) 
X_centred = X - mu             # (100, 5) — works because (5,) is broadcast to (100, 5)

# Common pattern: adding bias to each sample
b = np.array([0.1, 0.2, 0.3]) # (3,)
Z = np.random.randn(50, 3)    # (50, 3)
Z_biased = Z + b              # (50, 3) — b is broadcast across rows
```

**Debugging broadcasting issues:**

```python
a = np.array([1, 2, 3])       # shape (3,)
b = np.array([4, 5])          # shape (2,)
# a + b  → ValueError! Trailing dimensions 3 and 2 are incompatible

a = np.array([[1], [2], [3]]) # shape (3, 1)
b = np.array([4, 5])          # shape (2,)
print((a + b).shape)          # (3, 2) — outer-product-like broadcasting
```

> **Suggested experiment.** Deliberately create arrays with shapes `(3,)` vs `(3, 1)` and multiply/add them with a `(3, 3)` matrix. Print the result shapes. Understanding why `(3,)` and `(3, 1)` behave differently prevents hours of debugging.

---

### 6.3 Vectorisation

**Vectorised code** replaces Python loops with NumPy array operations, which are implemented in C and use SIMD instructions.

```python
n = 100000
y_true = np.random.randn(n)
y_pred = np.random.randn(n)

# SLOW: Python loop — O(n) interpreted iterations
mse_loop = 0
for i in range(n):
    mse_loop += (y_true[i] - y_pred[i]) ** 2
mse_loop /= n

# FAST: Vectorized — single call to compiled C code
mse_vec = np.mean((y_true - y_pred) ** 2)

# Both give the same result, but vectorized is 10–1000× faster
```

**Common vectorisation patterns in ML:**

| Operation       | Loop version                          | Vectorised                     |
| --------------- | ------------------------------------- | ------------------------------ |
| MSE loss        | `sum((y[i] - yh[i])**2) / n`          | `np.mean((y - yh)**2)`         |
| All predictions | `for i: yh[i] = w @ X[i]`             | `yh = X @ w`                   |
| Gradient        | `for i: g += ...`                     | `g = -2/n * X.T @ (y - X @ w)` |
| Feature scaling | `for j: X[:,j] = (X[:,j]-mu[j])/s[j]` | `X = (X - mu) / s`             |

> **Habit for this course.** Never write a Python loop over samples or features when a matrix operation exists. If you find yourself writing `for i in range(n)`, stop and ask whether there is a vectorised equivalent. There almost always is.

---

### 6.4 Numerical Stability

Floating-point arithmetic introduces errors that can compound in ML workloads.

#### Log-Sum-Exp Trick

Computing $\log\left(\sum_k e^{z_k}\right)$ directly overflows when $z_k$ is large. The stable version subtracts the maximum:

$$\log\sum_k e^{z_k} = c + \log\sum_k e^{z_k - c}, \qquad c = \max_k z_k$$

This arises in the softmax function ([Week 03](../../02_fundamentals/week03_linear_models/theory.md#77-multi-class-extension-softmax-regression), classification) and in log-likelihood computation.

```python
z = np.array([1000, 1001, 1002])   # direct exp will overflow

# UNSTABLE:
# np.log(np.sum(np.exp(z)))  → inf

# STABLE:
c = z.max()
log_sum_exp = c + np.log(np.sum(np.exp(z - c)))   # ≈ 1002.41
print(log_sum_exp)

# Or use scipy directly:
from scipy.special import logsumexp
print(logsumexp(z))   # same result
```

#### Avoiding Division by Zero

```python
eps = 1e-8   # small constant

# Cross-entropy: -y * log(y_hat) — y_hat can be 0
loss = -np.sum(y * np.log(y_hat + eps))

# Normalising: division by std that could be 0
X_normed = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
```

#### Condition Numbers

A matrix with a large condition number amplifies errors:

```python
# Well-conditioned
A_good = np.array([[1, 0], [0, 1]])
print(np.linalg.cond(A_good))   # 1.0

# Ill-conditioned
A_bad = np.array([[1, 1], [1, 1.0001]])
print(np.linalg.cond(A_bad))    # ~40000
```

When $X^\top X$ is ill-conditioned, the normal equation solution is numerically unreliable. This is another reason to use regularisation ([Week 06](../../02_fundamentals/week06_regularization/theory.md#2-why-regularisation-the-overfitting-problem-revisited)) or iterative solvers (gradient descent).

---

## 7. Symbol Reference

A consolidated reference for all notation used in this document and throughout the course.

| Symbol                               | Name                    | Meaning                                                            |
| ------------------------------------ | ----------------------- | ------------------------------------------------------------------ |
| $x, y, z$                            | Scalars                 | Single real numbers                                                |
| $\mathbf{x}, \mathbf{w}, \mathbf{b}$ | Vectors                 | Column vectors (bold lowercase)                                    |
| $X, W, A$                            | Matrices                | Rectangular arrays (uppercase italic)                              |
| $\mathbf{x}_i$                       | Data point              | The $i$-th sample (a vector)                                       |
| $x_{ij}$                             | Matrix entry            | Row $i$, column $j$                                                |
| $\theta$                             | Parameters              | Generic parameter vector                                           |
| $n$                                  | Sample count            | Number of training examples                                        |
| $d$                                  | Dimensionality          | Number of input features                                           |
| $K$                                  | Class/cluster count     | Number of categories or clusters                                   |
| $\eta$                               | Learning rate           | Step size in gradient descent                                      |
| $\lambda$                            | Regularisation strength | Penalty weight for regularisation terms                            |
| $\mathcal{L}(\theta)$                | Loss function           | Scalar measuring prediction error                                  |
| $\nabla_\theta \mathcal{L}$          | Gradient                | Vector of partial derivatives of loss w.r.t. $\theta$              |
| $H$ or $\nabla^2 f$                  | Hessian                 | Matrix of second derivatives                                       |
| $J$                                  | Jacobian                | Matrix of first derivatives of a vector function                   |
| $\hat{y}$                            | Prediction              | Model output (the "hat" denotes an estimate)                       |
| $r_i$                                | Residual                | $y_i - \hat{y}_i$: distance between truth and prediction           |
| $\mathcal{D}$                        | Dataset                 | $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$                                |
| $p(\cdot)$                           | Density/mass            | Probability density or mass function                               |
| $\mathbb{E}[\cdot]$                  | Expectation             | Expected (average) value over a distribution                       |
| $\text{Var}(\cdot)$                  | Variance                | Expected squared deviation from the mean                           |
| $\text{Cov}(\cdot, \cdot)$           | Covariance              | Joint variability of two variables                                 |
| $\Sigma$                             | Covariance matrix       | $d \times d$ matrix of pairwise covariances                        |
| $\sigma$                             | Standard deviation      | Square root of variance (or sigmoid, by context)                   |
| $\sigma(\cdot)$                      | Sigmoid function        | $\frac{1}{1 + e^{-x}}$ — maps $\mathbb{R}$ to $(0, 1)$             |
| $\sim$                               | Distributed as          | $X \sim \mathcal{N}(\mu, \sigma^2)$                                |
| $\mathbb{R}^d$                       | Real space              | The space of $d$-dimensional real vectors                          |
| $\|\mathbf{x}\|$                     | Norm                    | $L_2$ (Euclidean) norm unless subscripted                          |
| $\|\mathbf{x}\|_1$                   | $L_1$ norm              | Sum of absolute values                                             |
| $\odot$                              | Hadamard product        | Element-wise multiplication                                        |
| $\circ$                              | Composition             | $(f \circ g)(\mathbf{x}) = f(g(\mathbf{x}))$                       |
| $I$                                  | Identity matrix         | Diagonal matrix of ones                                            |
| $\text{tr}(A)$                       | Trace                   | Sum of diagonal entries                                            |
| $\det(A)$ or $                       | A                       | $                                                                  | Determinant | Scalar encoding volume change under $A$ |
| $\text{rank}(A)$                     | Rank                    | Number of linearly independent columns                             |
| $\kappa(A)$                          | Condition number        | $\lambda_{\max} / \lambda_{\min}$ — measures numerical sensitivity |
| $\lambda_i$                          | Eigenvalue              | Scaling factor for the $i$-th eigenvector                          |
| $\sigma_i$                           | Singular value          | Non-negative; from SVD: $A = U\Sigma V^\top$                       |
| $H(P)$                               | Entropy                 | Uncertainty of distribution $P$                                    |
| $H(P, Q)$                            | Cross-entropy           | Expected surprise when using $Q$ to encode $P$                     |
| $D_{\text{KL}}(P \| Q)$              | KL divergence           | "Distance" from $Q$ to $P$ (asymmetric)                            |

**Index conventions.** Subscript $i$ indexes samples ($i = 1, \ldots, n$), subscript $j$ indexes features ($j = 1, \ldots, d$), subscript $k$ indexes classes or clusters ($k = 1, \ldots, K$), and subscript $t$ indexes time or iteration steps.

---

## 8. References

1. Strang, G. (2016). *Introduction to Linear Algebra*, 5th ed. Wellesley-Cambridge Press. Chapters 1–7.
2. Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. Appendix A (linear algebra review). Available at [stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/).
3. 3Blue1Brown. "Essence of Linear Algebra" (YouTube). Chapters 1–11 for visual intuition on vectors, transformations, eigenvalues.
4. 3Blue1Brown. "Essence of Calculus" (YouTube). Chapters 1–7 for derivatives, chain rule, Taylor series.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapters 2–3 (linear algebra and probability). MIT Press. Available at [deeplearningbook.org](https://www.deeplearningbook.org).
6. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Chapter 2 (probability), Chapter 7 (linear algebra).
7. VanderPlas, J. (2016). *Python Data Science Handbook*, Chapter 2 (NumPy). O'Reilly.
8. Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley. Chapters 1–2 for entropy and KL divergence.
9. Khan Academy — "Probability & Statistics" module: Random variables, Normal distributions, Bayes' theorem.
10. NumPy documentation — [numpy.org/doc/stable/user/quickstart.html](https://numpy.org/doc/stable/user/quickstart.html).
