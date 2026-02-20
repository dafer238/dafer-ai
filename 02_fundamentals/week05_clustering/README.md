# Week 03 — Clustering

## Prerequisites

- **Week 00b** — distance metrics, basic linear algebra.
- **Week 03 (PCA / dimensionality reduction)** — PCA projections are used to visualise high-dimensional cluster structure in 2D.
- **Week 03 (linear models)** — familiarity with model evaluation concepts (train/test).

## What this week delivers

Clustering is the entry point to unsupervised learning. It recurs everywhere: initial data exploration, customer segmentation, image quantisation, anomaly detection, and latent-space analysis of deep models. Understanding when each algorithm is appropriate, and how to evaluate clusters without ground-truth labels, is an essential practical skill.

## Overview

Survey the three dominant algorithmic families (centroid-based, density-based, linkage-based), implement k-means from scratch, and compare all approaches on synthetic and real datasets. Evaluate results with both external (ARI) and internal (silhouette, Davies-Bouldin) metrics.

## Study

- K-means objective (inertia / WCSS) and Lloyd's algorithm
- Elbow method and silhouette analysis for choosing k
- Hierarchical clustering: linkage methods, dendrograms, cutting
- DBSCAN: ε-neighbourhood, core/border/noise points, choosing ε and min_samples
- Gaussian Mixture Models (soft clustering, EM algorithm overview)
- Cluster evaluation: Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index (ARI)

## Practical libraries & tools

- NumPy for from-scratch k-means
- `sklearn.cluster` (KMeans, AgglomerativeClustering, DBSCAN)
- `sklearn.mixture.GaussianMixture`
- `scipy.cluster.hierarchy` (linkage, dendrogram)
- `sklearn.metrics` (silhouette_score, davies_bouldin_score, adjusted_rand_score)
- Matplotlib/Seaborn for cluster plots and dendrograms

## Datasets & examples

- `sklearn.datasets.make_blobs` — well-separated, convex clusters (k-means ideal)
- `sklearn.datasets.make_moons` — non-convex, crescent-shaped clusters (DBSCAN ideal)
- `sklearn.datasets.make_circles` — concentric rings (demonstrates k-means failure)
- `sklearn.datasets.load_iris` — 4-D real data with known labels (ARI evaluation)
- `digits` PCA projections — visualising cluster structure in 2D

## Exercises

1. **K-means from scratch** — implement `kmeans_from_scratch(X, k, max_iter=300, tol=1e-4)` using Lloyd's algorithm; verify that centroid recovery matches centroid injection on `make_blobs` (tolerance 0.5).

2. **Elbow + Silhouette** — run k from 2 to 10 on `make_blobs`; plot inertia (elbow) and silhouette score; identify the best k from both.

3. **DBSCAN vs k-means** — apply both to `make_moons` and `make_circles`; compare cluster visualisations and silhouette scores.

4. **Hierarchical clustering** — cluster iris with Ward linkage; plot dendrogram; cut at 3 clusters and report ARI against true labels.

5. **GMM** — fit `GaussianMixture(n_components=3)` on iris; compare ARI with k-means.

## Code hints

```python
# Lloyd's k-means — one iteration
def _assign(X, centroids):
    dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)  # (n, k)
    return np.argmin(dists, axis=1)

def _update(X, labels, k):
    return np.vstack([X[labels == j].mean(axis=0) for j in range(k)])
```

## Deliverables

- [ ] `kmeans_from_scratch` implementation and centroid-recovery test.
- [ ] Elbow and silhouette plots.
- [ ] Side-by-side DBSCAN vs k-means on `make_moons` / `make_circles`.
- [ ] Dendrogram with marked cut.
- [ ] ARI comparison table: k-means, hierarchical, GMM on iris.

## What comes next

- **Week 05 (likelihood)** introduces GMMs as a probabilistic clustering model, justified by MLE.
- **Week 06 (uncertainty)** treats cluster membership as a latent variable (EM / variational inference).
- **Week 11 (CNNs)** — feature maps can be clustered to analyse learned representations.
