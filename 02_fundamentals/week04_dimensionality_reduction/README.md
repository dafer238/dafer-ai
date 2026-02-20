# Week 03 — Dimensionality Reduction & PCA

## Prerequisites

- **Week 00b** — linear algebra: matrix multiply, eigenvectors/eigenvalues, dot products.
- **Week 03 (linear models)** — understanding of covariance and linear transformations.
- **Week 01–02** — optimization intuition (gradient descent) helps contextualize SVD.

## What this week delivers

High-dimensional data is everywhere. PCA gives you a principled way to compress, visualize and denoise data by finding the directions of maximum variance. Understanding PCA deeply prepares you for autoencoders, attention mechanisms (which reduce sequence dimensionality), and any pipeline step that requires decorrelation or whitening.

## Overview

Build a complete picture of PCA: from the geometric intuition of variance-maximizing projections, through the mathematical machinery of SVD, to practical applications such as visualization and classification with reduced features.

## Study

- Covariance matrix and its eigenvectors as principal components
- SVD derivation of PCA; equivalence to eigendecomposition of covariance
- Explained variance ratio, scree plots, variance-retention thresholds
- Reconstruction error vs number of components
- PCA whitening and its role in preprocessing
- t-SNE / UMAP as non-linear alternatives (overview only — not implemented)

## Practical libraries & tools

- NumPy (`np.linalg.svd`, `np.cov`) for from-scratch PCA
- `sklearn.decomposition.PCA` for comparison and production use
- Matplotlib/Seaborn for scree plots, 2D projections, reconstruction panels

## Datasets & examples

- Synthetic correlated Gaussian data (2D → 1D intuition)
- `sklearn.datasets.load_iris()` — 4-D → 2-D with class separation
- `sklearn.datasets.load_digits()` — 64-D → 2-D visualisation + downstream classification

## Exercises

1. **PCA from SVD** — implement `pca_svd(X, n_components)` using `np.linalg.svd`; verify that explained variances match `sklearn.decomposition.PCA` (tolerance `1e-6`).

2. **Variance retention** — for `digits`, plot cumulative explained variance vs k and choose k that retains ≥ 95% variance; report reconstruction MSE vs k.

3. **Reconstruction visualisation** — reconstruct `digits` images at k = 5, 15, 30, 60 components; show a grid of originals vs reconstructions.

4. **Whitening** — implement PCA whitening and verify that the resulting covariance is approximately `I`.

5. **Downstream classification** — reduce `digits` with PCA (use the k from exercise 2); train a `KNeighborsClassifier` on original and reduced data; report accuracy and speed-up.

## Code hints

```python
# PCA via SVD
def pca_svd(X, n_components):
    X_c = X - X.mean(axis=0)        # centre
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    components = Vt[:n_components]  # principal axes
    scores = X_c @ components.T     # projected data
    explained_variance = (S**2) / (len(X) - 1)
    return scores, components, explained_variance[:n_components]
```

## Deliverables

- [ ] `pca_svd` implementation that matches sklearn (exercise 1).
- [ ] Scree plot showing explained variance ratio.
- [ ] Reconstruction error vs k curve and sample image grids.
- [ ] Whitening demo.
- [ ] Classification accuracy table: original vs PCA-compressed features.

## What comes next

- **Week 03 (clustering)** uses PCA projections to visualise cluster structure.
- **Week 11 (CNNs)** learns hierarchical representations that are related to learned PCA.
- **Week 14 (Transformers)** — attention heads operate on projected query/key spaces, which mirrors PCA projections.
