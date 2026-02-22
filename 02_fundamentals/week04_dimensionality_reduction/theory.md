# Dimensionality Reduction & Principal Component Analysis

## Table of Contents

- [Dimensionality Reduction \& Principal Component Analysis](#dimensionality-reduction--principal-component-analysis)
  - [Table of Contents](#table-of-contents)
  - [1. Scope and Purpose](#1-scope-and-purpose)
  - [2. The Curse of Dimensionality](#2-the-curse-of-dimensionality)
  - [3. Covariance and Correlation](#3-covariance-and-correlation)
    - [3.1 The Covariance Matrix](#31-the-covariance-matrix)
    - [3.2 Eigendecomposition of the Covariance Matrix](#32-eigendecomposition-of-the-covariance-matrix)
  - [4. Principal Component Analysis — The Optimisation View](#4-principal-component-analysis--the-optimisation-view)
    - [4.1 Maximum Variance Formulation](#41-maximum-variance-formulation)
    - [4.2 Minimum Reconstruction Error Formulation](#42-minimum-reconstruction-error-formulation)
    - [4.3 Equivalence of the Two Views](#43-equivalence-of-the-two-views)
  - [5. PCA via the Singular Value Decomposition](#5-pca-via-the-singular-value-decomposition)
    - [5.1 The SVD: Definition and Geometry](#51-the-svd-definition-and-geometry)
    - [5.2 Connecting SVD to PCA](#52-connecting-svd-to-pca)
    - [5.3 The Eckart–Young Theorem (Best Low-Rank Approximation)](#53-the-eckartyoung-theorem-best-low-rank-approximation)
  - [6. Explained Variance and Choosing $k$](#6-explained-variance-and-choosing-k)
    - [6.1 Explained Variance Ratio](#61-explained-variance-ratio)
    - [6.2 The Scree Plot](#62-the-scree-plot)
    - [6.3 Variance Retention Threshold](#63-variance-retention-threshold)
    - [6.4 Reconstruction Error](#64-reconstruction-error)
  - [7. PCA Whitening](#7-pca-whitening)
  - [8. Practical Considerations](#8-practical-considerations)
    - [8.1 Centering and Scaling](#81-centering-and-scaling)
    - [8.2 Computational Complexity](#82-computational-complexity)
    - [8.3 When PCA Fails](#83-when-pca-fails)
  - [9. Non-Linear Alternatives: t-SNE and UMAP](#9-non-linear-alternatives-t-sne-and-umap)
    - [9.1 t-SNE](#91-t-sne)
    - [9.2 UMAP](#92-umap)
    - [9.3 PCA vs. t-SNE vs. UMAP](#93-pca-vs-t-sne-vs-umap)
  - [10. Connections to the Rest of the Course](#10-connections-to-the-rest-of-the-course)
  - [11. Notebook Reference Guide](#11-notebook-reference-guide)
  - [12. Symbol Reference](#12-symbol-reference)
  - [13. References](#13-references)

---

## 1. Scope and Purpose

Real-world datasets often have many features — hundreds or thousands of dimensions. **Dimensionality reduction** finds a lower-dimensional representation that preserves the essential structure of the data while discarding noise and redundancy.

**Principal Component Analysis (PCA)** is the most fundamental dimensionality reduction method. It is:
- **Linear:** the reduced representation is a linear projection of the original data.
- **Variance-maximising:** it finds the directions along which the data varies most.
- **Unsupervised:** it uses only $X$, not labels $y$.

This week develops PCA from three complementary perspectives: (i) as a variance maximisation problem, (ii) as a reconstruction error minimisation problem, and (iii) as a consequence of the Singular Value Decomposition. Understanding all three builds strong geometric intuition that transfers to autoencoders ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md)), attention mechanisms ([Week 17](../../06_sequence_models/week17_attention/theory.md)), and any pipeline that benefits from decorrelation or compression.

**Prerequisites.** [Week 00b](../../01_intro/week00b_math_and_data/theory.md) (eigenvectors, eigenvalues, matrix multiplication, dot products), [Week 03](../week03_linear_models/theory.md) (covariance, linear models).

---

## 2. The Curse of Dimensionality

Before building the solution, it is important to understand the problem.

As the number of features $d$ increases while the number of examples $n$ stays fixed, several pathologies emerge:

**1. Volume explosion.** The volume of the unit hypercube in $d$ dimensions is $1^d = 1$, but the volume of the inscribed hypersphere $\to 0$ as $d \to \infty$. Most of the volume concentrates in the "corners" — data points are spread thin.

**2. Distance concentration.** For random points in high dimensions, the distance between the nearest and farthest neighbours converges:

$$\lim_{d \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$$

All pairwise distances become approximately equal. This is catastrophic for distance-based algorithms like KNN.

**3. Sample complexity.** To maintain the same density of examples, $n$ must grow exponentially with $d$. With $d = 100$ features and 1000 examples, each data point is essentially isolated in its own pocket of the feature space.

> **The remedy.** If the data actually lies near a lower-dimensional structure (a common assumption called the **manifold hypothesis**), we can project onto that structure without losing important information. PCA finds the best linear approximation to that structure.

---

## 3. Covariance and Correlation

### 3.1 The Covariance Matrix

Given $n$ centred data points $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^d$ (i.e., $\sum_i \mathbf{x}_i = \mathbf{0}$), the **sample covariance matrix** is:

$$\Sigma = \frac{1}{n-1}\sum_{i=1}^{n}\mathbf{x}_i\mathbf{x}_i^\top = \frac{1}{n-1}\tilde{X}^\top\tilde{X}$$

where $\tilde{X} \in \mathbb{R}^{n \times d}$ is the mean-centred data matrix (each row is a centred example).

| Entry                      | Formula                            | Meaning                                 |
| -------------------------- | ---------------------------------- | --------------------------------------- |
| $\Sigma_{jj}$              | $\frac{1}{n-1}\sum_i x_{ij}^2$     | Variance of feature $j$                 |
| $\Sigma_{jk}$ ($j \neq k$) | $\frac{1}{n-1}\sum_i x_{ij}x_{ik}$ | Covariance between features $j$ and $k$ |

**Properties of $\Sigma$:**

| Property                   | Statement                                                     |
| -------------------------- | ------------------------------------------------------------- |
| **Symmetric**              | $\Sigma = \Sigma^\top$                                        |
| **Positive semi-definite** | $\mathbf{z}^\top\Sigma\mathbf{z} \geq 0$ for all $\mathbf{z}$ |
| **Size**                   | $d \times d$ — does not depend on $n$                         |

The diagonal entries are the variances; the off-diagonal entries capture linear dependencies between features. A large positive $\Sigma_{jk}$ means features $j$ and $k$ tend to increase together; a value near zero means they are (linearly) uncorrelated.

In NumPy: `np.cov(X.T)` computes $\Sigma$ with the $\frac{1}{n-1}$ normalisation.

---

### 3.2 Eigendecomposition of the Covariance Matrix

Since $\Sigma$ is symmetric and positive semi-definite, it has a complete set of real, non-negative eigenvalues and orthogonal eigenvectors:

$$\Sigma = V \Lambda V^\top = \sum_{k=1}^{d} \lambda_k \mathbf{v}_k \mathbf{v}_k^\top$$

| Symbol                                                       | Size         | Meaning                                                  |
| ------------------------------------------------------------ | ------------ | -------------------------------------------------------- |
| $V = [\mathbf{v}_1, \ldots, \mathbf{v}_d]$                   | $d \times d$ | Orthogonal matrix of eigenvectors                        |
| $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$        | $d \times d$ | Diagonal matrix of eigenvalues                           |
| $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$ | —            | Eigenvalues in descending order                          |
| $\mathbf{v}_k$                                               | $d \times 1$ | $k$-th eigenvector (unit length, $\|\mathbf{v}_k\| = 1$) |

**The key fact:**

$$\mathbf{v}_k^\top\Sigma\mathbf{v}_k = \lambda_k$$

The variance of the data projected onto eigenvector $\mathbf{v}_k$ equals the eigenvalue $\lambda_k$. The eigenvector with the largest eigenvalue points in the direction of **maximum variance** — this is the first principal component.

> **Intuition.** The covariance matrix is an ellipsoid. The eigenvectors are its axes; the eigenvalues are the squared semi-axis lengths. PCA rotates the coordinate system so that the axes align with the ellipsoid's axes, ordering them from longest (most variance) to shortest (least variance).

> **Notebook reference.** Cell 5 computes `np.linalg.eigh(cov_hat)` on a 2D correlated Gaussian dataset. The two eigenvectors are drawn as arrows on the scatter plot, scaled by $\sqrt{\lambda_k}$ — representing the standard deviation along each principal direction.

---

## 4. Principal Component Analysis — The Optimisation View

### 4.1 Maximum Variance Formulation

**Goal:** find a unit vector $\mathbf{v}_1 \in \mathbb{R}^d$ such that the variance of the data projected onto $\mathbf{v}_1$ is maximised.

The projection of centred data point $\mathbf{x}_i$ onto $\mathbf{v}_1$ is $z_i = \mathbf{v}_1^\top \mathbf{x}_i$. The variance of these projected values is:

$$\text{Var}(z) = \frac{1}{n-1}\sum_{i=1}^{n}(\mathbf{v}_1^\top \mathbf{x}_i)^2 = \mathbf{v}_1^\top \left(\frac{1}{n-1}\tilde{X}^\top\tilde{X}\right)\mathbf{v}_1 = \mathbf{v}_1^\top \Sigma \mathbf{v}_1$$

The optimisation problem:

$$\max_{\mathbf{v}_1} \;\; \mathbf{v}_1^\top \Sigma \mathbf{v}_1 \qquad \text{subject to} \quad \|\mathbf{v}_1\| = 1$$

**Solution via Lagrange multipliers.** Form the Lagrangian:

$$\mathcal{L}(\mathbf{v}_1, \lambda) = \mathbf{v}_1^\top \Sigma \mathbf{v}_1 - \lambda(\mathbf{v}_1^\top \mathbf{v}_1 - 1)$$

Take the derivative with respect to $\mathbf{v}_1$ and set to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_1} = 2\Sigma \mathbf{v}_1 - 2\lambda \mathbf{v}_1 = \mathbf{0}$$

$$\boxed{\Sigma \mathbf{v}_1 = \lambda \mathbf{v}_1}$$

This is an **eigenvalue equation**. The solution $\mathbf{v}_1$ is an eigenvector of $\Sigma$, and the corresponding eigenvalue $\lambda$ is the projected variance:

$$\mathbf{v}_1^\top \Sigma \mathbf{v}_1 = \mathbf{v}_1^\top (\lambda \mathbf{v}_1) = \lambda$$

To maximise variance, choose the eigenvector with the **largest eigenvalue**. This is the **first principal component** $\mathbf{v}_1$.

**Subsequent components.** The $k$-th principal component $\mathbf{v}_k$ maximises $\mathbf{v}_k^\top \Sigma \mathbf{v}_k$ subject to $\|\mathbf{v}_k\| = 1$ **and** $\mathbf{v}_k \perp \mathbf{v}_j$ for all $j < k$ (orthogonal to all previous components). The solution is the eigenvector with the $k$-th largest eigenvalue $\lambda_k$.

**In summary:** the principal components are the eigenvectors of $\Sigma$, ordered by descending eigenvalue.

---

### 4.2 Minimum Reconstruction Error Formulation

**Goal:** find a $k$-dimensional linear subspace that **best approximates** the data.

Let $V_k = [\mathbf{v}_1, \ldots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$ be the basis of the subspace. The projection (encoding) and reconstruction (decoding) of $\mathbf{x}_i$ are:

$$\mathbf{z}_i = V_k^\top \mathbf{x}_i \in \mathbb{R}^k \qquad \text{(encode: project onto subspace)}$$

$$\hat{\mathbf{x}}_i = V_k \mathbf{z}_i = V_k V_k^\top \mathbf{x}_i \in \mathbb{R}^d \qquad \text{(decode: reconstruct in original space)}$$

The **reconstruction error** is:

$$\mathcal{E}_k = \frac{1}{n}\sum_{i=1}^{n}\|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 = \frac{1}{n}\sum_{i=1}^{n}\|\mathbf{x}_i - V_k V_k^\top \mathbf{x}_i\|^2$$

The optimisation problem:

$$\min_{V_k : V_k^\top V_k = I_k} \;\; \frac{1}{n}\sum_{i=1}^{n}\|\mathbf{x}_i - V_k V_k^\top \mathbf{x}_i\|^2$$

**Theorem.** The optimal $V_k$ consists of the $k$ eigenvectors of $\Sigma$ with the $k$ largest eigenvalues. The minimum reconstruction error is:

$$\boxed{\mathcal{E}_k^* = \sum_{j=k+1}^{d}\lambda_j}$$

*Proof sketch.* Using $V = [\mathbf{v}_1, \ldots, \mathbf{v}_d]$ (all eigenvectors), we can decompose:

$$\mathbf{x}_i = \sum_{j=1}^{d} (\mathbf{v}_j^\top \mathbf{x}_i)\mathbf{v}_j$$

The reconstruction using only the first $k$ components is:

$$\hat{\mathbf{x}}_i = \sum_{j=1}^{k} (\mathbf{v}_j^\top \mathbf{x}_i)\mathbf{v}_j$$

The error is:

$$\|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 = \left\|\sum_{j=k+1}^{d}(\mathbf{v}_j^\top \mathbf{x}_i)\mathbf{v}_j\right\|^2 = \sum_{j=k+1}^{d}(\mathbf{v}_j^\top \mathbf{x}_i)^2$$

(using orthogonality of the $\mathbf{v}_j$). Averaging over $i$:

$$\mathcal{E}_k = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=k+1}^{d}(\mathbf{v}_j^\top \mathbf{x}_i)^2 = \sum_{j=k+1}^{d}\underbrace{\frac{1}{n}\sum_{i=1}^{n}(\mathbf{v}_j^\top \mathbf{x}_i)^2}_{\propto \lambda_j} \propto \sum_{j=k+1}^{d}\lambda_j$$

The error equals the sum of the eigenvalues **not** included in the projection. To minimise this, include the $k$ largest eigenvalues — exactly the maximum variance solution. $\blacksquare$

---

### 4.3 Equivalence of the Two Views

The total variance of the centred data is:

$$\text{Total variance} = \text{tr}(\Sigma) = \sum_{j=1}^{d} \lambda_j$$

This can be split into the variance **retained** by the first $k$ components and the variance **lost** (reconstruction error):

$$\underbrace{\sum_{j=1}^{k}\lambda_j}_{\text{retained}} + \underbrace{\sum_{j=k+1}^{d}\lambda_j}_{\text{lost (= reconstruction error)}} = \sum_{j=1}^{d}\lambda_j$$

Therefore: **maximising retained variance** $\iff$ **minimising reconstruction error**. The two formulations yield the same solution.

> **Intuition.** Imagine an elliptical cloud of points in 3D. PCA finds the plane that captures the most "spread" of the cloud. The spread captured is the retained variance; the thickness of the cloud perpendicular to the plane is the reconstruction error. These are complementary views of the same decomposition.

---

## 5. PCA via the Singular Value Decomposition

### 5.1 The SVD: Definition and Geometry

Every matrix $A \in \mathbb{R}^{m \times n}$ has a **Singular Value Decomposition**:

$$A = U S V^\top$$

| Factor                          | Size                       | Properties                        |
| ------------------------------- | -------------------------- | --------------------------------- |
| $U \in \mathbb{R}^{m \times m}$ | Left singular vectors      | Orthogonal: $U^\top U = I$        |
| $S \in \mathbb{R}^{m \times n}$ | Singular values (diagonal) | $s_1 \geq s_2 \geq \cdots \geq 0$ |
| $V \in \mathbb{R}^{n \times n}$ | Right singular vectors     | Orthogonal: $V^\top V = I$        |

**Thin (economy) SVD.** When $m > n$, only the first $n$ columns of $U$ and the top $n \times n$ block of $S$ matter:

$$A = U_n S_n V^\top, \qquad U_n \in \mathbb{R}^{m \times n}, \quad S_n \in \mathbb{R}^{n \times n}$$

This is what `np.linalg.svd(A, full_matrices=False)` returns.

**Geometric interpretation.** The SVD decomposes the linear transformation $\mathbf{y} = A\mathbf{x}$ into three steps:

$$\mathbf{x} \xrightarrow{V^\top} \text{rotate} \xrightarrow{S} \text{scale} \xrightarrow{U} \text{rotate}$$

1. $V^\top$ rotates $\mathbf{x}$ to align with the "natural" axes of $A$.
2. $S$ stretches along each axis by the corresponding singular value.
3. $U$ rotates the result into the output space.

**Relationship to eigendecomposition.** For the centred data matrix $\tilde{X}$:

$$\tilde{X}^\top \tilde{X} = V S^\top U^\top U S V^\top = V S^2 V^\top$$

Comparing with $\Sigma = \frac{1}{n-1}\tilde{X}^\top\tilde{X} = V\Lambda V^\top$:

$$\boxed{\lambda_k = \frac{s_k^2}{n - 1}}$$

The eigenvalues of $\Sigma$ are the squared singular values of $\tilde{X}$ divided by $n - 1$. The right singular vectors $V$ of $\tilde{X}$ are the eigenvectors of $\Sigma$.

---

### 5.2 Connecting SVD to PCA

Given centred data $\tilde{X} \in \mathbb{R}^{n \times d}$ with thin SVD $\tilde{X} = U_r S_r V_r^\top$ (where $r = \min(n, d)$):

| PCA quantity                    | SVD expression                                                      |
| ------------------------------- | ------------------------------------------------------------------- |
| **Principal components** (axes) | Rows of $V_r^\top$ (= columns of $V_r$)                             |
| **Scores** (projected data)     | $Z = \tilde{X}V_r = U_r S_r$                                        |
| **Explained variance**          | $\lambda_k = s_k^2 / (n-1)$                                         |
| **Reconstruction**              | $\hat{X}_k = U_k S_k V_k^\top + \boldsymbol{\mu}$ (first $k$ terms) |

**Why use SVD instead of eigendecomposition?**

1. **Numerical stability.** Computing $\tilde{X}^\top\tilde{X}$ squares the condition number ($\kappa(\tilde{X}^\top\tilde{X}) = \kappa(\tilde{X})^2$). The SVD works directly on $\tilde{X}$, avoiding this squaring.
2. **Efficiency when $n < d$.** The thin SVD's cost is $\mathcal{O}(\min(nd^2, n^2d))$, which is cheaper than forming and decomposing the $d \times d$ covariance matrix when $n \ll d$.
3. **Standard implementation.** `sklearn.decomposition.PCA` uses the SVD internally.

> **Notebook reference.** Cell 7 implements `pca_svd`:
> ```python
> U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
> components = Vt[:n_components]       # principal axes
> scores = X_c @ components.T          # projected data
> expl_var = (S ** 2) / (n - 1)        # eigenvalues
> ```
> Cell 9 verifies that the explained variance ratios match `sklearn.decomposition.PCA` to within $10^{-6}$.

---

### 5.3 The Eckart–Young Theorem (Best Low-Rank Approximation)

**Theorem (Eckart–Young–Mirsky, 1936).** For any matrix $A$ with SVD $A = USV^\top$, the best rank-$k$ approximation (in both the Frobenius and spectral norms) is the **truncated SVD**:

$$A_k = U_k S_k V_k^\top = \sum_{j=1}^{k} s_j \mathbf{u}_j \mathbf{v}_j^\top$$

And the approximation error is:

$$\|A - A_k\|_F^2 = \sum_{j=k+1}^{r} s_j^2$$

**Connection to PCA.** For the centred data matrix $\tilde{X}$:

$$\tilde{X}_k = U_k S_k V_k^\top$$

is the best rank-$k$ approximation. The reconstruction error is $\|\tilde{X} - \tilde{X}_k\|_F^2 = \sum_{j=k+1}^{r}s_j^2$. Dividing by $n$, this is proportional to $\sum_{j=k+1}^{d}\lambda_j$ — exactly the PCA reconstruction error from Section 4.2.

> **The deep result.** PCA's optimality is not just among orthogonal projections — it is the best rank-$k$ linear approximation **period** (over all matrices of rank $\leq k$). No linear method can do better.

---

## 6. Explained Variance and Choosing $k$

### 6.1 Explained Variance Ratio

The **explained variance ratio** of the $k$-th component is:

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^{d}\lambda_j} = \frac{s_k^2}{\sum_{j=1}^{d}s_j^2}$$

The **cumulative explained variance ratio** for $k$ components:

$$\text{CEVR}_k = \sum_{j=1}^{k}\text{EVR}_j = \frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{d}\lambda_j}$$

| Quantity            | Meaning                                                                  |
| ------------------- | ------------------------------------------------------------------------ |
| $\text{EVR}_k$      | Fraction of total variance captured by component $k$ alone               |
| $\text{CEVR}_k$     | Fraction of total variance captured by the first $k$ components together |
| $1 - \text{CEVR}_k$ | Fraction of variance lost = normalised reconstruction error              |

---

### 6.2 The Scree Plot

A **scree plot** displays $\text{EVR}_k$ vs. $k$ (often as a bar chart). Named after the geological term "scree" (loose rocks at the base of a cliff), the plot typically shows a steep drop followed by a flat "elbow."

**Heuristic: the elbow rule.** Choose $k$ at the point where the plot transitions from steep to flat — the "elbow." Components after the elbow contribute little variance and are primarily noise.

> **Notebook reference.** Cell 11 (left panel) plots the scree plot for the first 20 components of the digits dataset. The first few components capture large chunks of variance; later components contribute diminishing amounts.

---

### 6.3 Variance Retention Threshold

A more principled approach: choose the smallest $k$ such that:

$$\text{CEVR}_k \geq \tau$$

Common thresholds: $\tau = 0.90$ (90%), $\tau = 0.95$ (95%), $\tau = 0.99$ (99%).

For the digits dataset ($d = 64$):

$$k_{95\%} \approx 28 \text{ components (retaining 95% of variance from 64 features)}$$

This achieves a compression ratio of $64/28 \approx 2.3\times$ while preserving 95% of the information (as measured by variance).

> **Notebook reference.** Cell 11 (right panel) plots the cumulative EVR curve with a horizontal line at 95% and a vertical line at $k_{95}$.

---

### 6.4 Reconstruction Error

The reconstruction error at $k$ components is:

$$\text{MSE}_k = \frac{1}{n}\|\tilde{X} - \tilde{X}_k\|_F^2 = \frac{1}{n}\sum_{j=k+1}^{d}s_j^2 = \sum_{j=k+1}^{d}\frac{n-1}{n}\lambda_j \approx \sum_{j=k+1}^{d}\lambda_j$$

| $k$  | Variance retained | Reconstruction MSE | Visual quality (digits)                   |
| ---- | ----------------- | ------------------ | ----------------------------------------- |
| $5$  | Low               | High               | Blurry blobs                              |
| $15$ | Moderate          | Moderate           | Recognisable digits, missing fine details |
| $30$ | ~95%              | Low                | High quality, slight smoothing            |
| $64$ | 100%              | 0                  | Perfect reconstruction                    |

> **Notebook reference.** Cell 13 visualises reconstructions of digit images at $k \in \{5, 15, 30, 64\}$. The progression from blurry to sharp illustrates how information is encoded across components.
>
> **Suggested experiment.** Add $k = 1$ and $k = 2$ to the reconstruction grid. At $k = 1$, all digit images should look nearly identical — the first PC captures only the overall brightness pattern. At $k = 2$, digits with different global orientations should become distinguishable.

---

## 7. PCA Whitening

**Whitening** (sphering) transforms the data so that the covariance is the identity matrix $I$. This means all transformed features are uncorrelated and have unit variance.

**The procedure:**

1. **Centre** the data: $\tilde{X} = X - \boldsymbol{\mu}$.
2. **Project** onto the principal components: $Z = \tilde{X}V_k$ (scores).
3. **Normalise** each component by its standard deviation: $Z_{\text{white}} = Z \cdot \text{diag}(\lambda_1, \ldots, \lambda_k)^{-1/2}$.

In matrix form (using the SVD of $\tilde{X}$):

$$X_{\text{white}} = U_k \quad (\text{since } Z = U_k S_k \text{ and } Z_{\text{white}} = U_k S_k S_k^{-1} = U_k)$$

Or equivalently:

$$X_{\text{white}} = \tilde{X}V_k\Lambda_k^{-1/2}$$

**Verification.** The covariance of $X_{\text{white}}$ is:

$$\frac{1}{n-1}X_{\text{white}}^\top X_{\text{white}} = \Lambda_k^{-1/2}V_k^\top \underbrace{\frac{1}{n-1}\tilde{X}^\top\tilde{X}}_{\Sigma} V_k \Lambda_k^{-1/2} = \Lambda_k^{-1/2}\Lambda_k\Lambda_k^{-1/2} = I_k$$

$\blacksquare$

**Why whiten?**
- Many algorithms assume uncorrelated, unit-variance inputs (e.g., some neural network training schemes, ICA).
- Whitening removes the effect of feature scaling, making the optimisation landscape more isotropic ([[Week 01](../week01_optimization/theory.md)](../week01_optimization/theory.md)–[02](../week02_advanced_optimizers/theory.md): this improves the condition number of the loss surface).

> **Notebook reference.** Cell 15 implements `pca_whitening` and verifies that `np.cov(X_white.T)` is close to $I$.
>
> **Suggested experiment.** Whiten the digits dataset and train KNN on the whitened features vs. the PCA-only features (without normalisation). The whitened version may have slightly different accuracy — the effect depends on whether KNN is sensitive to feature scaling (it is, since it uses Euclidean distance).

---

## 8. Practical Considerations

### 8.1 Centering and Scaling

**Centering is mandatory.** PCA seeks directions of maximum variance. If the data is not centred, the first PC will point toward the mean — this is an artefact of the offset, not meaningful structure.

**Standardisation (scaling to unit variance) is often desirable.** If features have very different scales (e.g., one feature is in metres and another in kilogrammes), PCA will be dominated by the high-variance feature purely because of its units. Standardising ($z$-scoring) each feature to mean 0 and variance 1 before PCA ensures that all features contribute equally.

| Situation                                            | Pre-processing                                            |
| ---------------------------------------------------- | --------------------------------------------------------- |
| All features in same units (e.g., pixel intensities) | Centre only                                               |
| Features in different units or different scales      | Centre + standardise (use `StandardScaler`)               |
| Correlation PCA (focus on correlation structure)     | Standardise (equivalent to PCA on the correlation matrix) |

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)  # z-score: (x - mean) / std
```

---

### 8.2 Computational Complexity

| Method                         | Time complexity                 | Space complexity                | Best when                    |
| ------------------------------ | ------------------------------- | ------------------------------- | ---------------------------- |
| Full SVD                       | $\mathcal{O}(\min(nd^2, n^2d))$ | $\mathcal{O}(nd)$               | $n$ and $d$ are moderate     |
| Eigendecomposition of $\Sigma$ | $\mathcal{O}(nd^2 + d^3)$       | $\mathcal{O}(d^2)$ for $\Sigma$ | $n \gg d$                    |
| Randomised SVD                 | $\mathcal{O}(ndk)$              | $\mathcal{O}(nk + dk)$          | $k \ll d$ and large datasets |
| Incremental PCA                | $\mathcal{O}(ndk)$ per batch    | $\mathcal{O}(dk)$               | Data does not fit in memory  |

Scikit-learn uses randomised SVD by default when `n_components` is much smaller than $\min(n, d)$ (`PCA(svd_solver='auto')` selects the fastest method automatically).

---

### 8.3 When PCA Fails

PCA is a **linear** method. It can fail when:

1. **The important structure is nonlinear.** A spiral or Swiss roll in 3D cannot be unrolled by a linear projection. Non-linear methods (t-SNE, UMAP, autoencoders) are needed.

2. **Variance ≠ importance.** PCA finds directions of maximum variance, but high-variance directions are not always the most informative for a downstream task. A feature with low variance could be the most discriminative (e.g., a rare but diagnostic symptom). **Supervised** dimensionality reduction methods (LDA, supervised PCA) address this.

3. **Non-Gaussian data.** PCA captures only second-order statistics (mean and covariance). For data with important higher-order structure (e.g., skewness, heavy tails), **Independent Component Analysis (ICA)** may be more appropriate.

> **When PCA is ideal.** The data is approximately Gaussian (or at least has elliptical structure), the features are on comparable scales, and you want an efficient, interpretable, globally optimal linear projection. This covers a surprisingly large fraction of real-world datasets.

---

## 9. Non-Linear Alternatives: t-SNE and UMAP

These methods are primarily used for **visualisation** (reducing to 2D or 3D). Unlike PCA, they are non-linear and focus on preserving **local neighbourhood structure** rather than global variance.

### 9.1 t-SNE

**t-Distributed Stochastic Neighbour Embedding** (van der Maaten & Hinton, 2008) works by:

1. Defining pairwise similarities in high-dimensional space using Gaussian kernels:

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

2. Defining pairwise similarities in low-dimensional space using **Student-t** kernels (heavier tails):

$$q_{ij} = \frac{(1 + \|\mathbf{z}_i - \mathbf{z}_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{z}_k - \mathbf{z}_l\|^2)^{-1}}$$

3. Minimising the KL divergence $\text{KL}(P \| Q)$ via gradient descent in the $\mathbf{z}_i$ coordinates.

| Property                      | Consequence                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| Preserves **local** structure | Nearby points stay nearby; clusters are visible                       |
| Distorts **global** structure | Distances between clusters are meaningless                            |
| Non-parametric                | No closed-form mapping from new data; must re-run on the full dataset |
| Perplexity parameter          | Controls neighbourhood size; must be tuned                            |

**Limitations:**
- Non-deterministic (depends on initialisation and random seed).
- Computationally expensive: $\mathcal{O}(n^2)$ for exact; $\mathcal{O}(n \log n)$ for Barnes-Hut approximation.
- Cannot project new data points without re-running.

---

### 9.2 UMAP

**Uniform Manifold Approximation and Projection** (McInnes et al., 2018) is a more recent alternative to t-SNE. It is grounded in topological data analysis and Riemannian geometry.

**Practical advantages over t-SNE:**
- Faster (better scaling to large datasets).
- Better preservation of **global** structure (inter-cluster distances are more meaningful).
- Provides a parametric mapping that can be applied to new data.
- Supports arbitrary target dimensions (not just 2D).

Both t-SNE and UMAP are available in scikit-learn and the `umap-learn` package:
```python
from sklearn.manifold import TSNE
from umap import UMAP

z_tsne = TSNE(n_components=2, perplexity=30).fit_transform(X)
z_umap = UMAP(n_components=2, n_neighbors=15).fit_transform(X)
```

---

### 9.3 PCA vs. t-SNE vs. UMAP

| Property          | PCA                                                    | t-SNE                             | UMAP                                   |
| ----------------- | ------------------------------------------------------ | --------------------------------- | -------------------------------------- |
| **Type**          | Linear                                                 | Non-linear                        | Non-linear                             |
| **Preserves**     | Global variance                                        | Local neighbourhood               | Local + moderate global                |
| **Deterministic** | Yes                                                    | No                                | No                                     |
| **Scalable**      | Very ($\mathcal{O}(ndk)$)                              | Moderate ($\mathcal{O}(n\log n)$) | Good ($\mathcal{O}(n^{1.14})$ approx.) |
| **New data**      | $\mathbf{z} = V_k^\top(\mathbf{x} - \boldsymbol{\mu})$ | Must re-run                       | Parametric version available           |
| **Interpretable** | Components are linear combos of features               | No clear interpretation           | No clear interpretation                |
| **Use case**      | Preprocessing, denoising, compression                  | Visualisation                     | Visualisation, clustering              |

> **Best practice.** Use PCA for preprocessing and compression (it is deterministic, fast, and reversible). Use t-SNE or UMAP for **visualisation only** — never for downstream modelling, because the mapping is non-invertible and non-parametric.

---

## 10. Connections to the Rest of the Course

| Week                          | Connection                                                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **[Week 03](../week03_linear_models/theory.md) (Linear models)**   | PCA decorrelates features → improves condition number of $X^\top X$ → faster convergence                                 |
| **[Week 05](../week05_clustering/theory.md) (Clustering)**      | PCA projections used to visualise cluster structure in 2D                                                                |
| **[Week 06](../week06_regularization/theory.md) (Regularisation)**  | Ridge regression shrinks weights along low-eigenvalue principal component directions — a spectral view of regularisation |
| **[Week 07](../../03_probability/week07_likelihood/theory.md) (Likelihood)**      | Probabilistic PCA (PPCA) gives a generative model for PCA                                                                |
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md) (NN from scratch)** | A linear autoencoder (no activation) learns the same subspace as PCA                                                     |
| **[Week 15](../../05_deep_learning/week15_cnn_representations/theory.md) (CNNs)**            | Convolutional features can be analysed via PCA to understand what the network learns                                     |
| **[Week 17](../../06_sequence_models/week17_attention/theory.md) (Attention)**       | Query/Key projections in attention are learned linear projections — analogous to PCA but task-specific                   |

> **The unifying principle.** PCA is the simplest example of **representation learning**: finding a lower-dimensional representation that preserves the essential structure. Every method from autoencoders to transformers can be viewed as a nonlinear, task-specific generalisation of PCA.

---

## 11. Notebook Reference Guide

| Cell                   | Section                         | What to observe                                                     | Theory reference |
| ---------------------- | ------------------------------- | ------------------------------------------------------------------- | ---------------- |
| 5 (2D Gaussian)        | Eigenvectors on scatter plot    | Two arrows: long = high-variance PC, short = low-variance PC        | Section 3.2, 4.1 |
| 7 (`pca_svd`)          | From-scratch PCA implementation | SVD → components, scores, explained variance                        | Section 5.2      |
| 9 (sklearn comparison) | Verify match                    | EVR values identical to sklearn PCA (exact to $10^{-6}$)            | Section 5.2      |
| 9 (Iris 2D plot)       | Coloured by species             | Three clusters; setosa separates cleanly on PC1                     | Section 4.1      |
| 11 (Scree plot)        | Bar chart of EVR                | Steep drop then flat tail — "elbow" visible                         | Section 6.2      |
| 11 (Cumulative EVR)    | Line plot with thresholds       | 95% line crossed at $k \approx 28$                                  | Section 6.3      |
| 13 (Reconstructions)   | Image grid at various $k$       | Sharp progression from blurry to crisp                              | Section 6.4      |
| 15 (Whitening)         | Covariance of whitened data     | Diagonal ≈ 1, off-diagonal ≈ 0                                      | Section 7        |
| 17 (Classification)    | KNN accuracy table              | PCA-compressed often matches or exceeds original (noise removed)    | Section 2        |
| Ex.1                   | SVD vs sklearn on digits        | Full verification at $10^{-6}$ tolerance                            | Section 5.2      |
| Ex.2                   | MSE vs $k$ curve                | Monotonically decreasing; steep early, flat later                   | Section 6.4      |
| Ex.5                   | Speed benchmark                 | PCA-compressed KNN is faster (fewer features) with similar accuracy | Section 8.2      |

**Suggested modifications across exercises:**

| Modification                                               | What it reveals                                                          |
| ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| Skip centering in `pca_svd`                                | First PC will point toward the mean — components become meaningless      |
| Apply PCA to un-scaled features with different units       | PC1 dominated by the high-magnitude feature; standardisation fixes this  |
| Increase noise in synthetic 2D data                        | Eigenvalues become more equal; harder to choose $k$                      |
| Try $k = 2$ for digits classification                      | Much worse accuracy — too much information lost for a 10-class problem   |
| Compare PCA + KNN to PCA + logistic regression             | Tests whether the downstream classifier matters more than the projection |
| Plot PCA 2D projection and t-SNE 2D side by side on digits | t-SNE reveals clusters; PCA shows global spread but clusters overlap     |

---

## 12. Symbol Reference

| Symbol                               | Name                             | Meaning                                                            |
| ------------------------------------ | -------------------------------- | ------------------------------------------------------------------ |
| $\mathbf{x} \in \mathbb{R}^d$        | Data point                       | One example with $d$ features                                      |
| $X \in \mathbb{R}^{n \times d}$      | Data matrix                      | $n$ examples stacked as rows                                       |
| $\tilde{X}$                          | Centred data matrix              | $X$ with column means subtracted                                   |
| $\boldsymbol{\mu} \in \mathbb{R}^d$  | Mean vector                      | $\frac{1}{n}\sum_i \mathbf{x}_i$                                   |
| $n$                                  | Number of examples               | Rows of $X$                                                        |
| $d$                                  | Number of features (dimension)   | Columns of $X$                                                     |
| $k$                                  | Number of retained components    | Chosen by the user ($k \leq d$)                                    |
| $\Sigma \in \mathbb{R}^{d \times d}$ | Sample covariance matrix         | $\frac{1}{n-1}\tilde{X}^\top\tilde{X}$                             |
| $\lambda_k$                          | $k$-th eigenvalue of $\Sigma$    | Variance along the $k$-th PC                                       |
| $\mathbf{v}_k \in \mathbb{R}^d$      | $k$-th eigenvector of $\Sigma$   | $k$-th principal component direction (unit length)                 |
| $V_k \in \mathbb{R}^{d \times k}$    | Matrix of first $k$ eigenvectors | Projection basis: columns are $\mathbf{v}_1, \ldots, \mathbf{v}_k$ |
| $\Lambda$                            | Diagonal matrix of eigenvalues   | $\text{diag}(\lambda_1, \ldots, \lambda_d)$                        |
| $z_i = V_k^\top \mathbf{x}_i$        | Score (projected data)           | Coordinates of $\mathbf{x}_i$ in the PC basis                      |
| $Z \in \mathbb{R}^{n \times k}$      | Score matrix                     | All projected data points                                          |
| $U \in \mathbb{R}^{n \times r}$      | Left singular vectors            | From SVD of $\tilde{X}$                                            |
| $S \in \mathbb{R}^{r \times r}$      | Singular values (diagonal)       | $s_k = \sqrt{(n-1)\lambda_k}$                                      |
| $V^\top \in \mathbb{R}^{r \times d}$ | Right singular vectors           | Rows are the principal components                                  |
| $r = \text{rank}(\tilde{X})$         | Rank                             | $r \leq \min(n, d)$                                                |
| $\text{EVR}_k$                       | Explained variance ratio         | $\lambda_k / \sum_j \lambda_j$                                     |
| $\text{CEVR}_k$                      | Cumulative EVR                   | $\sum_{j=1}^{k}\text{EVR}_j$                                       |
| $\tau$                               | Variance retention threshold     | E.g., 0.95                                                         |
| $\mathcal{E}_k$                      | Reconstruction error             | $\sum_{j=k+1}^{d}\lambda_j$                                        |
| $H = VV^\top$                        | Projection matrix                | Projects onto the subspace spanned by $V$                          |
| $\hat{\mathbf{x}}_i$                 | Reconstructed data point         | $V_k V_k^\top \mathbf{x}_i + \boldsymbol{\mu}$                     |
| $\kappa$                             | Condition number                 | $\lambda_1 / \lambda_d$ (or $s_1 / s_d$)                           |
| $p_{j                                | i}$, $q_{ij}$                    | t-SNE similarities                                                 | High-dim and low-dim pairwise probabilities |
| $\text{KL}(P\|Q)$                    | KL divergence                    | Measure of distribution mismatch (used in t-SNE)                   |

---

## 13. References

1. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer. — The standard monograph on PCA.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 12.1. Springer. — Bayesian treatment and probabilistic PCA.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Section 14.5. Springer. — PCA and its variants. Free: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/).
4. Strang, G. (2019). *Linear Algebra and Learning from Data*. Wellesley-Cambridge. — SVD and low-rank approximation with data science perspective.
5. Eckart, C. & Young, G. (1936). "The approximation of one matrix by another of lower rank." *Psychometrika*, 1(3), 211–218. — The best low-rank approximation theorem.
6. van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using t-SNE." *JMLR*, 9, 2579–2605. — t-SNE.
7. McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv:1802.03426*. — UMAP.
8. Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." *SIAM Review*, 53(2), 217–288. — Randomised SVD.
9. Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space." *Philosophical Magazine*, 2(11), 559–572. — The original PCA paper.
10. Hotelling, H. (1933). "Analysis of a complex of statistical variables into principal components." *Journal of Educational Psychology*, 24(6), 417–441. — PCA independently derived.
11. Scikit-learn User Guide: Decompositions. [https://scikit-learn.org/stable/modules/decomposition.html](https://scikit-learn.org/stable/modules/decomposition.html).
12. Shlens, J. (2014). "A Tutorial on Principal Component Analysis." *arXiv:1404.1100*. — Concise, accessible tutorial.
