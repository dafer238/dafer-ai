# Clustering: Unsupervised Structure Discovery

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [The Clustering Problem](#2-the-clustering-problem)
    - 2.1 [What Is a Cluster?](#21-what-is-a-cluster)
    - 2.2 [Distance and Similarity](#22-distance-and-similarity)
    - 2.3 [Taxonomy of Clustering Algorithms](#23-taxonomy-of-clustering-algorithms)
3. [K-Means Clustering](#3-k-means-clustering)
    - 3.1 [The Objective: Within-Cluster Sum of Squares](#31-the-objective-within-cluster-sum-of-squares)
    - 3.2 [Lloyd's Algorithm](#32-lloyds-algorithm)
    - 3.3 [Convergence Analysis](#33-convergence-analysis)
    - 3.4 [Initialisation: K-Means++](#34-initialisation-k-means)
    - 3.5 [Choosing $K$: Elbow Method and Silhouette Analysis](#35-choosing-k-elbow-method-and-silhouette-analysis)
    - 3.6 [Limitations of K-Means](#36-limitations-of-k-means)
4. [Hierarchical Clustering](#4-hierarchical-clustering)
    - 4.1 [Agglomerative (Bottom-Up) Clustering](#41-agglomerative-bottom-up-clustering)
    - 4.2 [Linkage Criteria](#42-linkage-criteria)
    - 4.3 [The Dendrogram](#43-the-dendrogram)
    - 4.4 [Choosing the Number of Clusters](#44-choosing-the-number-of-clusters)
5. [DBSCAN: Density-Based Clustering](#5-dbscan-density-based-clustering)
    - 5.1 [Core, Border, and Noise Points](#51-core-border-and-noise-points)
    - 5.2 [The Algorithm](#52-the-algorithm)
    - 5.3 [Choosing $\varepsilon$ and MinPts](#53-choosing-varepsilon-and-minpts)
    - 5.4 [Strengths and Limitations](#54-strengths-and-limitations)
6. [Gaussian Mixture Models: Soft Clustering](#6-gaussian-mixture-models-soft-clustering)
    - 6.1 [The Model](#61-the-model)
    - 6.2 [The EM Algorithm (Intuition)](#62-the-em-algorithm-intuition)
    - 6.3 [GMM vs. K-Means](#63-gmm-vs-k-means)
    - 6.4 [Model Selection: BIC and AIC](#64-model-selection-bic-and-aic)
7. [Cluster Evaluation](#7-cluster-evaluation)
    - 7.1 [External Metrics (When Labels Are Available)](#71-external-metrics-when-labels-are-available)
    - 7.2 [Internal Metrics (No Labels)](#72-internal-metrics-no-labels)
    - 7.3 [Metric Comparison](#73-metric-comparison)
8. [Algorithm Selection Guide](#8-algorithm-selection-guide)
9. [Connections to the Rest of the Course](#9-connections-to-the-rest-of-the-course)
10. [Notebook Reference Guide](#10-notebook-reference-guide)
11. [Symbol Reference](#11-symbol-reference)
12. [References](#12-references)

---

## 1. Scope and Purpose

All models in [Weeks 01](../week01_optimization/theory.md#4-gradient-descent)–[04](../week04_dimensionality_reduction/theory.md) were **supervised**: they learned from labelled data $\{(\mathbf{x}_i, y_i)\}$. This week introduces **unsupervised learning**: discovering structure in unlabelled data $\{\mathbf{x}_i\}$.

**Clustering** is the task of partitioning a dataset into groups (_clusters_) such that points within a cluster are "similar" and points in different clusters are "dissimilar." There is no single correct answer — different algorithms, distance metrics, and assumptions can yield different valid clusterings of the same data.

Clustering is ubiquitous:
- **Data exploration** — understanding structure before building models.
- **Customer segmentation** — grouping users by behaviour.
- **Image quantisation** — reducing colour palettes.
- **Anomaly detection** — points that belong to no cluster are outliers.
- **Latent-space analysis** — understanding what deep models learn ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#4-fully-connected-networks-mlps), [Week 15](../../05_deep_learning/week15_cnn_representations/theory.md#9-visualising-filters-and-activations)).

This week covers the four main algorithmic families: centroid-based (K-means), linkage-based (hierarchical), density-based (DBSCAN), and model-based (Gaussian Mixture Models), along with the evaluation machinery needed to assess results without ground-truth labels.

**Prerequisites.** [Week 00b](../../01_intro/week00b_math_and_data/theory.md#26-norms) (distance metrics, linear algebra), [Week 04](../week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view) (PCA for visualisation of clusters in 2D).

---

## 2. The Clustering Problem

### 2.1 What Is a Cluster?

There is no single universal definition. Each algorithm implicitly defines what it considers a "cluster":

| Algorithm    | Implicit cluster definition                                                         |
| ------------ | ----------------------------------------------------------------------------------- |
| K-means      | A set of points closer to their centroid than to any other centroid (Voronoi cells) |
| Hierarchical | Groups of points connected at a given distance threshold                            |
| DBSCAN       | A maximal set of density-connected points                                           |
| GMM          | Points drawn from the same Gaussian component                                       |

> **Key insight.** The choice of algorithm encodes an assumption about cluster shape. K-means assumes **spherical** clusters. Hierarchical clustering can produce arbitrary shapes depending on the linkage. DBSCAN assumes **density-connected** regions. GMM assumes **elliptical** (Gaussian) clusters. Matching the algorithm to the data geometry is the most important practical decision.

---

### 2.2 Distance and Similarity

All clustering algorithms depend on a notion of distance or similarity between points.

**Euclidean distance** (the default for all algorithms in this week):

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{j=1}^{d}(x_j - y_j)^2}$$

**Other distance metrics (for reference):**

| Metric                   | Formula                                                                 | When to use                                                 |
| ------------------------ | ----------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Manhattan** ($\ell_1$) | $\sum_j \|x_j - y_j\|$                                                  | Sparse data, high dimensions                                |
| **Cosine distance**      | $1 - \frac{\mathbf{x}^\top\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$    | Text data, when magnitude is irrelevant                     |
| **Mahalanobis**          | $\sqrt{(\mathbf{x}-\mathbf{y})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{y})}$ | When feature correlations matter (GMM uses this internally) |

> **Scaling matters.** Euclidean distance is sensitive to feature scales. If features have different units (e.g., metres vs. kilogrammes), always standardise before clustering:
> ```python
> from sklearn.preprocessing import StandardScaler
> X_scaled = StandardScaler().fit_transform(X)
> ```
> This is the same issue as PCA ([Week 04](../week04_dimensionality_reduction/theory.md#81-centering-and-scaling), Section 8.1).

---

### 2.3 Taxonomy of Clustering Algorithms

| Family             | Examples                | Cluster shape                  | Requires $K$?       | Handles noise?                      |
| ------------------ | ----------------------- | ------------------------------ | ------------------- | ----------------------------------- |
| **Centroid-based** | K-means, K-medoids      | Spherical (convex)             | Yes                 | No                                  |
| **Linkage-based**  | Agglomerative, BIRCH    | Arbitrary (depends on linkage) | Yes (cut threshold) | No                                  |
| **Density-based**  | DBSCAN, HDBSCAN, OPTICS | Arbitrary (non-convex)         | No                  | Yes (noise points)                  |
| **Model-based**    | GMM, Bayesian GMM       | Elliptical                     | Yes                 | Indirectly (low-probability points) |

---

## 3. K-Means Clustering

### 3.1 The Objective: Within-Cluster Sum of Squares

K-means seeks a partition of $n$ points into $K$ clusters $C_1, C_2, \ldots, C_K$ and $K$ centroids $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K$ that minimise the **Within-Cluster Sum of Squares (WCSS)**, also called **inertia**:

$$\boxed{J = \sum_{k=1}^{K}\sum_{\mathbf{x}_i \in C_k}\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2}$$

| Symbol                          | Meaning                                                                 |
| ------------------------------- | ----------------------------------------------------------------------- |
| $K$                             | Number of clusters (specified by the user)                              |
| $C_k$                           | Set of points assigned to cluster $k$                                   |
| $\boldsymbol{\mu}_k = \frac{1}{ | C_k                                                                     | }\sum_{\mathbf{x}_i \in C_k}\mathbf{x}_i$ | Centroid (mean) of cluster $k$ |
| $J$                             | WCSS / inertia — total squared distance from each point to its centroid |

**Geometric interpretation.** $J$ measures the total "compactness" of the clusters. Each cluster's contribution is the sum of squared distances from its members to its centre — this is proportional to the cluster's variance. Minimising $J$ yields the tightest possible grouping around $K$ centres.

> **Hardness.** Finding the globally optimal partition is NP-hard (even for $K = 2$). K-means uses a greedy heuristic (Lloyd's algorithm) that finds a local minimum.

---

### 3.2 Lloyd's Algorithm

The standard K-means algorithm (Lloyd, 1982) alternates two steps:

**Initialise:** choose $K$ initial centroids $\boldsymbol{\mu}_1^{(0)}, \ldots, \boldsymbol{\mu}_K^{(0)}$.

**Repeat until convergence:**

1. **Assign step.** Assign each point to the nearest centroid:

$$C_k^{(t)} = \left\{\mathbf{x}_i : k = \arg\min_j \|\mathbf{x}_i - \boldsymbol{\mu}_j^{(t)}\|^2\right\}$$

2. **Update step.** Recompute each centroid as the mean of its assigned points:

$$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|C_k^{(t)}|}\sum_{\mathbf{x}_i \in C_k^{(t)}}\mathbf{x}_i$$

**Convergence criterion:** stop when the centroids change by less than a tolerance $\varepsilon$ (e.g., $\|\boldsymbol{\mu}^{(t+1)} - \boldsymbol{\mu}^{(t)}\| < 10^{-4}$) or when labels do not change.

In NumPy, the assign step is:
```python
dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)  # (n, K)
labels = np.argmin(dists, axis=1)
```

And the update step:
```python
centroids = np.vstack([X[labels == k].mean(axis=0) for k in range(K)])
```

> **Notebook reference.** Cell 5 implements `kmeans_from_scratch` with K-means++ initialisation. The code runs on `make_blobs` (4 well-separated clusters) and reports ARI ≈ 1.0.

---

### 3.3 Convergence Analysis

**Theorem.** Lloyd's algorithm converges in a finite number of iterations.

*Proof sketch.* Each step (assign or update) either decreases $J$ or leaves it unchanged:
- **Assign step:** reassigning a point to a closer centroid strictly decreases the point's contribution to $J$.
- **Update step:** setting $\boldsymbol{\mu}_k$ to the mean of $C_k$ minimises $\sum_{\mathbf{x}_i \in C_k}\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$ (the mean is the least-squares optimal centre — this is the same argument as the normal equations in [Week 03](../week03_linear_models/theory.md#33-the-normal-equations-closed-form-solution), but for a single cluster).

Since $J$ is bounded below by 0 and strictly decreases, and there are finitely many possible partitions of $n$ points into $K$ clusters, the algorithm must terminate.

**Complexity per iteration:** $\mathcal{O}(nKd)$ — computing all $n \times K$ distances in $d$ dimensions.

**Total complexity:** $\mathcal{O}(nKdT)$, where $T$ is the number of iterations (typically $T \ll n$; empirically, convergence is fast on most datasets).

> **Caveat.** Convergence is to a **local minimum**, not the global minimum. The result depends on the initialisation. Running K-means multiple times with different random seeds (`n_init=10` in sklearn) and taking the best (lowest $J$) is standard practice.

---

### 3.4 Initialisation: K-Means++

Random initialisation can lead to poor local minima (e.g., two centroids near the same cluster, leaving another cluster uncovered). **K-means++** (Arthur & Vassilvitskii, 2007) provides a smarter initialisation:

1. Choose the first centroid $\boldsymbol{\mu}_1$ uniformly at random from the data.
2. For each subsequent centroid $\boldsymbol{\mu}_j$:
   - Compute $D(\mathbf{x}_i)$ = the distance from $\mathbf{x}_i$ to the **nearest already-chosen centroid**.
   - Choose the next centroid with probability proportional to $D(\mathbf{x}_i)^2$.
3. Repeat until $K$ centroids are chosen.

**Why it works.** Points far from existing centroids have high $D(\mathbf{x}_i)^2$ and are more likely to be chosen. This spreads the initial centroids across the dataset, reducing the chance of two centroids starting in the same cluster.

**Guarantee.** K-means++ initialisation ensures that the expected value of $J$ is at most $\mathcal{O}(\log K)$ times the optimal $J$. This is a competitive ratio guarantee — much better than random initialisation, which has no such bound.

> **Notebook reference.** The `kmeans_from_scratch` in Cell 5 implements K-means++ initialisation: the probability of selecting the next centroid is proportional to the squared distance to the nearest existing centroid.

---

### 3.5 Choosing $K$: Elbow Method and Silhouette Analysis

K-means requires the user to specify $K$ in advance. Two standard heuristics help choose $K$:

#### The Elbow Method

Plot $J$ (inertia) vs. $K$ for $K = 1, 2, \ldots, K_{\max}$. The curve always decreases (more clusters = lower inertia), but the rate of decrease slows. The "elbow" — the point where the curve transitions from steep to flat — suggests the natural $K$.

**Mathematical motivation.** When $K$ is less than the true number of clusters, adding one more cluster captures a large amount of variance (big drop in $J$). When $K$ exceeds the true number, additional clusters split existing clusters and yield diminishing returns (small drop in $J$).

**Formalisation (gap statistic).** Tibshirani et al. (2001) formalised the elbow heuristic by comparing $\log J$ to its expected value under a null reference distribution (uniform random data). The optimal $K$ maximises the "gap":

$$\text{Gap}(K) = \mathbb{E}_{\text{null}}[\log J_K^*] - \log J_K$$

#### Silhouette Analysis

The **silhouette score** for a single point $\mathbf{x}_i$ measures how well it fits its assigned cluster compared to the next-best cluster:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

| Symbol            | Meaning                                                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| $a_i$             | **Intra-cluster distance:** mean distance from $\mathbf{x}_i$ to all other points in its cluster                  |
| $b_i$             | **Nearest-cluster distance:** mean distance from $\mathbf{x}_i$ to all points in the nearest neighbouring cluster |
| $s_i \in [-1, 1]$ | $+1$: perfectly clustered; $0$: on the boundary; $-1$: probably misassigned                                       |

The **overall silhouette score** is $\bar{s} = \frac{1}{n}\sum_i s_i$. Choose $K$ that maximises $\bar{s}$.

| $\bar{s}$ range | Interpretation              |
| --------------- | --------------------------- |
| $0.71 - 1.0$    | Strong structure            |
| $0.51 - 0.70$   | Reasonable structure        |
| $0.26 - 0.50$   | Weak / overlapping clusters |
| $< 0.25$        | No substantial structure    |

> **Notebook reference.** Cell 7 plots both the elbow curve and silhouette score for $K \in \{2, \ldots, 10\}$ on `make_blobs`. Both methods should correctly identify $K = 4$.
>
> **Suggested experiment.** Repeat the analysis on `make_moons` ($K = 2$). The silhouette score peaks at $K = 2$, but the visual result is poor because K-means cannot capture non-convex shapes. This motivates DBSCAN (Section 5).

---

### 3.6 Limitations of K-Means

| Limitation                 | Root cause                                                               | Remedy                                           |
| -------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------ |
| Assumes spherical clusters | Objective minimises squared Euclidean distance = isotropic Voronoi cells | GMM (elliptical), DBSCAN (arbitrary shape)       |
| Sensitive to outliers      | Outliers pull centroids away from cluster centres                        | K-medoids (uses medians), DBSCAN (labels noise)  |
| Requires specifying $K$    | No principled built-in model selection                                   | Elbow/silhouette, BIC (GMM), HDBSCAN (automatic) |
| Finds local minima         | NP-hard globally                                                         | K-means++, multiple random restarts              |
| Equal-sized clusters       | WCSS is biased toward equal partition sizes                              | GMM (varying covariances), spectral clustering   |

> **Notebook reference.** Cell 9 demonstrates K-means failing on `make_moons` and `make_circles` — non-convex shapes that K-means cannot capture. DBSCAN recovers the correct structure.

---

## 4. Hierarchical Clustering

### 4.1 Agglomerative (Bottom-Up) Clustering

Agglomerative clustering builds a hierarchy of clusters from the bottom up:

1. **Start:** every point is its own cluster ($n$ singleton clusters).
2. **Merge:** at each step, merge the two closest clusters.
3. **Repeat** until only one cluster remains.

This produces a binary tree of $n - 1$ merges. To obtain $K$ clusters, cut the tree at the appropriate level.

**Complexity:** $\mathcal{O}(n^2d)$ for distance matrix computation + $\mathcal{O}(n^2 \log n)$ for the linkage = $\mathcal{O}(n^2 d + n^2 \log n)$ overall. The $\mathcal{O}(n^2)$ distance matrix makes this impractical for very large datasets.

---

### 4.2 Linkage Criteria

The "distance between two clusters" can be defined in several ways. Let $C_a$ and $C_b$ be two clusters:

| Linkage      | Formula                                                                      | Character                                                                                                                                |
| ------------ | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Single**   | $\min\{d(\mathbf{x}, \mathbf{y}) : \mathbf{x} \in C_a, \mathbf{y} \in C_b\}$ | Distance between closest pair. Can find non-convex shapes; susceptible to **chaining** (thin bridges of points merge unrelated clusters) |
| **Complete** | $\max\{d(\mathbf{x}, \mathbf{y}) : \mathbf{x} \in C_a, \mathbf{y} \in C_b\}$ | Distance between farthest pair. Produces compact, roughly equal-diameter clusters                                                        |
| **Average**  | $\frac{1}{                                                                   | C_a                                                                                                                                      |  | C_b | }\sum_{\mathbf{x} \in C_a}\sum_{\mathbf{y} \in C_b}d(\mathbf{x}, \mathbf{y})$ | Mean pairwise distance. Compromise between single and complete |
| **Ward**     | Increase in total WCSS caused by merging $C_a$ and $C_b$                     | Minimises inertia at each merge. Most similar to K-means; tends to produce spherical, equal-sized clusters                               |

**Ward linkage in detail.** The distance between $C_a$ and $C_b$ under Ward's criterion is:

$$d_{\text{Ward}}(C_a, C_b) = \frac{|C_a| \cdot |C_b|}{|C_a| + |C_b|}\|\boldsymbol{\mu}_a - \boldsymbol{\mu}_b\|^2$$

This measures how much the total WCSS would increase if the two clusters were merged. Ward linkage is the default in most implementations and often produces the most interpretable results.

> **Notebook reference.** Cell 11 uses Ward linkage on the Iris dataset. Exercise 4 (Cell 18) asks you to compare all four linkage methods and report ARI for each.

---

### 4.3 The Dendrogram

A **dendrogram** is a tree diagram that displays the merge history:

- **Horizontal axis:** individual data points (leaves) or sub-clusters.
- **Vertical axis:** the distance (or dissimilarity) at which each merge occurs.
- **Horizontal lines:** connect the two clusters that were merged; their height indicates the merge distance.

**Reading the dendrogram:**
- Long vertical gaps between merge levels suggest well-separated clusters.
- A horizontal cut at a given height produces a specific number of clusters.
- The number of intersections of a horizontal line with the tree gives $K$.

```python
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(X, method='ward')
dendrogram(Z)
```

The linkage matrix $Z$ has shape $(n-1, 4)$: each row records `[cluster_a, cluster_b, distance, size]`.

> **Notebook reference.** Cell 11 produces a truncated dendrogram (showing the last 12 merges for clarity). The tallest gap between merge levels should correspond to cutting into 3 clusters (matching the 3 Iris species).

---

### 4.4 Choosing the Number of Clusters

| Method                  | Approach                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| **Visual (dendrogram)** | Cut where the longest vertical gap occurs                                                |
| **Inconsistency**       | Flag merges where the distance is significantly larger than the average of recent merges |
| **External validation** | If labels are available, maximise ARI at various cut levels                              |
| **Internal validation** | Maximise silhouette score across different cuts                                          |

> **Suggested experiment.** Plot the dendrogram for `make_moons`. Single linkage should correctly identify 2 non-convex clusters (connected through density), while Ward linkage will fail (it assumes spherical shapes). This demonstrates how linkage choice interacts with cluster geometry.

---

## 5. DBSCAN: Density-Based Clustering

### 5.1 Core, Border, and Noise Points

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise; Ester et al., 1996) defines clusters as regions of high density separated by regions of low density.

Two parameters govern the algorithm:

| Parameter     | Name                 | Meaning                                                                     |
| ------------- | -------------------- | --------------------------------------------------------------------------- |
| $\varepsilon$ | Neighbourhood radius | Maximum distance for two points to be considered neighbours                 |
| MinPts        | Minimum points       | Minimum number of points within $\varepsilon$-radius to form a dense region |

The **$\varepsilon$-neighbourhood** of a point $\mathbf{x}$ is:

$$N_\varepsilon(\mathbf{x}) = \{\mathbf{y} \in \mathcal{D} : d(\mathbf{x}, \mathbf{y}) \leq \varepsilon\}$$

Points are classified into three types:

| Type             | Condition                                               | Interpretation                 |
| ---------------- | ------------------------------------------------------- | ------------------------------ |
| **Core point**   | $                                                       | N_\varepsilon(\mathbf{x})      | \geq \text{MinPts}$                                       | In a dense region; part of a cluster's interior |
| **Border point** | $                                                       | N_\varepsilon(\mathbf{x})      | < \text{MinPts}$ but within $\varepsilon$ of a core point | On the edge of a cluster                        |
| **Noise point**  | Not core and not within $\varepsilon$ of any core point | Outlier; belongs to no cluster |

**Density-reachability.** A point $\mathbf{y}$ is **directly density-reachable** from $\mathbf{x}$ if $\mathbf{x}$ is a core point and $\mathbf{y} \in N_\varepsilon(\mathbf{x})$. A point is **density-reachable** if there is a chain of directly density-reachable points connecting them.

A **cluster** in DBSCAN is a maximal set of density-connected points: every pair of core points is connected through a chain of core points within $\varepsilon$ of each other.

---

### 5.2 The Algorithm

```
1. For each unvisited point p:
   a. Mark p as visited
   b. Find N_ε(p) — all points within ε of p
   c. If |N_ε(p)| < MinPts → label p as noise (may later become a border point)
   d. Else (p is a core point):
      i.   Create a new cluster C
      ii.  Add p to C
      iii. For each point q in N_ε(p):
           - If q is unvisited: mark visited, find N_ε(q)
             If |N_ε(q)| ≥ MinPts: add N_ε(q) to the neighbourhood to expand
           - If q is not yet assigned to any cluster: add q to C
```

**Key properties:**
- **No $K$ needed.** The number of clusters is determined by the data and the parameters $\varepsilon$, MinPts.
- **Noise handling.** Points that do not belong to any cluster are labelled as noise ($-1$ in sklearn).
- **Arbitrary shape.** Clusters can have any shape, as long as they are connected through density.

**Complexity:** $\mathcal{O}(n^2)$ for naive implementation (computing all pairwise distances). With spatial indexing (KD-tree or ball tree), this reduces to $\mathcal{O}(n \log n)$ in low dimensions.

---

### 5.3 Choosing $\varepsilon$ and MinPts

**MinPts rule of thumb** (Sander et al., 1998):
- For 2D data: $\text{MinPts} \geq 4$.
- General: $\text{MinPts} \approx 2d$ (twice the dimensionality).

**Choosing $\varepsilon$ — the k-distance plot:**

1. For each point, compute the distance to its $k$-th nearest neighbour ($k = \text{MinPts}$).
2. Sort these distances in descending order and plot them.
3. The "elbow" in the plot (where distances jump sharply) suggests a good $\varepsilon$.

**Intuition.** Points inside dense clusters have small $k$-distances (many neighbours nearby). Noise points have large $k$-distances (isolated). The elbow separates the two regimes.

```python
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X)
distances, _ = nn.kneighbors(X)
k_dists = np.sort(distances[:, -1])[::-1]  # k-th nearest neighbour distance
plt.plot(k_dists)
plt.ylabel('5-NN distance'); plt.xlabel('Points (sorted)')
plt.title('k-distance plot'); plt.show()
```

> **Notebook reference.** Exercise 3 (Cell 17) asks you to sweep $\varepsilon \in \{0.1, 0.2, \ldots, 0.6\}$ on `make_moons` and report noise fraction vs. ARI.
>
> **Suggested experiment.** Generate `make_moons` with increasing noise levels (e.g., `noise` $\in \{0.05, 0.10, 0.20\}$). Observe how the optimal $\varepsilon$ shifts — higher noise requires larger $\varepsilon$ (more tolerant neighbourhoods), but at the cost of potentially merging the two crescents.

---

### 5.4 Strengths and Limitations

| Strength                            | Limitation                                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| No need to specify $K$              | Sensitive to $\varepsilon$ and MinPts                                                                         |
| Discovers arbitrary-shaped clusters | Struggles with varying-density clusters (a single $\varepsilon$ cannot capture both dense and sparse regions) |
| Identifies noise/outliers           | Degrades in high dimensions (distances concentrate, Section 2 of [Week 04](../week04_dimensionality_reduction/theory.md#2-the-curse-of-dimensionality))                                     |
| Deterministic (given parameters)    | Border points may be assigned to different clusters depending on visit order                                  |

**HDBSCAN** (Campello et al., 2013) addresses the varying-density limitation by running DBSCAN across all $\varepsilon$ values simultaneously and extracting stable clusters from the resulting hierarchy. It requires only MinPts and is generally the recommended density-based method for production use.

---

## 6. Gaussian Mixture Models: Soft Clustering

### 6.1 The Model

All algorithms so far assign each point to exactly one cluster (**hard clustering**). **Gaussian Mixture Models (GMMs)** perform **soft (probabilistic) clustering**: each point has a probability of belonging to each cluster.

A GMM assumes the data is generated by a mixture of $K$ Gaussian distributions:

$$p(\mathbf{x}) = \sum_{k=1}^{K}\pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

| Symbol                                                                   | Meaning                                                                                     |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| $K$                                                                      | Number of mixture components                                                                |
| $\pi_k$                                                                  | Mixing coefficient (prior probability of component $k$): $\pi_k \geq 0$, $\sum_k \pi_k = 1$ |
| $\boldsymbol{\mu}_k \in \mathbb{R}^d$                                    | Mean of component $k$                                                                       |
| $\boldsymbol{\Sigma}_k \in \mathbb{R}^{d \times d}$                      | Covariance matrix of component $k$                                                          |
| $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ | Multivariate Gaussian density                                                               |

The multivariate Gaussian density is:

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

The exponent $(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$ is the **Mahalanobis distance** from $\mathbf{x}$ to $\boldsymbol{\mu}$ — it accounts for the covariance structure. Points that are far in Mahalanobis distance have low density, even if they are close in Euclidean distance.

**Covariance types in sklearn:**

| `covariance_type` | $\boldsymbol{\Sigma}_k$                                    | Parameters per component | Flexibility                     |
| ----------------- | ---------------------------------------------------------- | ------------------------ | ------------------------------- |
| `'full'`          | Arbitrary $d \times d$ PSD matrix                          | $\mathcal{O}(d^2)$       | Full ellipsoids (most flexible) |
| `'diag'`          | Diagonal matrix                                            | $\mathcal{O}(d)$         | Axis-aligned ellipsoids         |
| `'spherical'`     | $\sigma_k^2 I$                                             | $\mathcal{O}(1)$         | Spheres (like K-means)          |
| `'tied'`          | All $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$ (shared) | $\mathcal{O}(d^2)$ total | Same shape, different means     |

---

### 6.2 The EM Algorithm (Intuition)

GMM parameters cannot be estimated in closed form (unlike linear regression). They are fit using the **Expectation-Maximisation (EM)** algorithm, which alternates two steps analogous to Lloyd's algorithm for K-means:

**E-step (Expectation — analogous to K-means assign).** Compute the **responsibility** $\gamma_{ik}$ — the posterior probability that point $\mathbf{x}_i$ was generated by component $k$:

$$\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K}\pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

This is a **soft assignment**: $\gamma_{ik} \in [0, 1]$ and $\sum_k \gamma_{ik} = 1$.

**M-step (Maximisation — analogous to K-means update).** Update each component's parameters using the responsibilities as weights:

$$N_k = \sum_{i=1}^{n}\gamma_{ik} \qquad \text{(effective number of points in component } k\text{)}$$

$$\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k}\sum_{i=1}^{n}\gamma_{ik}\mathbf{x}_i$$

$$\boldsymbol{\Sigma}_k^{\text{new}} = \frac{1}{N_k}\sum_{i=1}^{n}\gamma_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^\top$$

$$\pi_k^{\text{new}} = \frac{N_k}{n}$$

**Convergence.** EM provably increases the log-likelihood at every step (or leaves it unchanged) and converges to a local maximum. Like K-means, it is sensitive to initialisation.

> **K-means is a special case of EM for GMMs.** If all $\boldsymbol{\Sigma}_k = \sigma^2 I$ (spherical, equal variance) and $\sigma^2 \to 0$, the soft assignments $\gamma_{ik}$ collapse to hard assignments (0 or 1), and EM reduces to Lloyd's algorithm.

> **Forward pointer.** The EM algorithm is the topic of [Week 07](../../03_probability/week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle) (Likelihood) and [Week 08](../../03_probability/week08_uncertainty/theory.md#7-bayesian-inference) (Uncertainty), where it is derived rigorously from the perspective of maximum likelihood estimation with latent variables.

---

### 6.3 GMM vs. K-Means

| Property            | K-Means                                      | GMM                                               |
| ------------------- | -------------------------------------------- | ------------------------------------------------- |
| **Assignment**      | Hard ($\mathbf{x}_i$ belongs to one cluster) | Soft (probability distribution over clusters)     |
| **Cluster shape**   | Spherical (Voronoi cells)                    | Elliptical (covariance matrices)                  |
| **Cluster size**    | Biased toward equal-size clusters            | Different sizes via $\pi_k$                       |
| **Objective**       | Minimise WCSS (inertia)                      | Maximise log-likelihood                           |
| **Algorithm**       | Lloyd's (assign/update)                      | EM (E-step/M-step)                                |
| **Speed**           | Faster ($\mathcal{O}(nKd)$ per iteration)    | Slower ($\mathcal{O}(nKd^2)$ for full covariance) |
| **Model selection** | Elbow, silhouette                            | BIC, AIC (principled)                             |

> **Notebook reference.** Cell 13 fits GMM with $K = 3$ on Iris and compares ARI to K-means and hierarchical clustering. GMM often achieves the highest ARI because Iris clusters are elliptical (different species have different feature variances).

---

### 6.4 Model Selection: BIC and AIC

GMMs have a principled model selection criterion based on **information criteria**:

**Bayesian Information Criterion (BIC):**

$$\text{BIC} = -2\ell(\hat{\theta}) + p\log n$$

**Akaike Information Criterion (AIC):**

$$\text{AIC} = -2\ell(\hat{\theta}) + 2p$$

| Symbol               | Meaning                                |
| -------------------- | -------------------------------------- |
| $\ell(\hat{\theta})$ | Maximised log-likelihood               |
| $p$                  | Number of free parameters in the model |
| $n$                  | Number of data points                  |

Both criteria balance **goodness-of-fit** ($-2\ell$) against **model complexity** ($p$). BIC penalises complexity more heavily (via $\log n > 2$ for $n > 8$).

**Choosing $K$:** fit GMMs for $K = 1, 2, \ldots, K_{\max}$, compute BIC for each, and choose the $K$ that **minimises** BIC.

The number of parameters for a full-covariance GMM with $K$ components in $d$ dimensions:

$$p = K\left(d + \frac{d(d+1)}{2}\right) + K - 1 = K \cdot \frac{d^2 + 3d}{2} + K - 1$$

(Each component has $d$ mean parameters, $d(d+1)/2$ covariance parameters, and there are $K - 1$ free mixing coefficients since they sum to 1.)

> **Notebook reference.** Exercise 5 (Cell 19) asks you to fit GMMs with $K \in \{1, \ldots, 8\}$ on Iris and plot BIC vs. $K$. The minimum should occur at $K = 3$.
>
> **Suggested experiment.** Compare BIC model selection to the elbow method on the same dataset. BIC provides a single number per $K$ (no visual interpretation needed), making it more reproducible. On well-separated Gaussian clusters, both should agree; on non-Gaussian data, BIC may overfit.

---

## 7. Cluster Evaluation

Evaluating cluster quality is challenging because there is typically no ground truth. Two families of metrics exist:

### 7.1 External Metrics (When Labels Are Available)

These compare the clustering to known labels. They are used in research (to benchmark algorithms) but are not available in practice (where you cluster because you _don't_ have labels).

**Adjusted Rand Index (ARI):**

The Rand Index counts the fraction of point-pairs that are in agreement (both in the same cluster and same class, or both in different clusters and different classes). The **Adjusted** version corrects for chance:

$$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$

| Value | Interpretation                         |
| ----- | -------------------------------------- |
| $1.0$ | Perfect agreement with ground truth    |
| $0.0$ | Random labelling (agreement by chance) |
| $< 0$ | Worse than random                      |

**Other external metrics:** Normalised Mutual Information (NMI), Fowlkes-Mallows Index, V-measure.

---

### 7.2 Internal Metrics (No Labels)

These evaluate the clustering using only the data itself.

**Silhouette Score** (Section 3.5):

$$\bar{s} = \frac{1}{n}\sum_{i=1}^{n}\frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1]$$

Higher is better. Measures separation between clusters relative to within-cluster cohesion.

**Davies-Bouldin Index:**

$$\text{DB} = \frac{1}{K}\sum_{k=1}^{K}\max_{j \neq k}\left(\frac{S_k + S_j}{d_{kj}}\right)$$

where $S_k = \frac{1}{|C_k|}\sum_{\mathbf{x} \in C_k}\|\mathbf{x} - \boldsymbol{\mu}_k\|$ is the average intra-cluster distance and $d_{kj} = \|\boldsymbol{\mu}_k - \boldsymbol{\mu}_j\|$ is the inter-centroid distance.

| Symbol   | Meaning                                                                   |
| -------- | ------------------------------------------------------------------------- |
| $S_k$    | Average distance from points to centroid within cluster $k$ (compactness) |
| $d_{kj}$ | Distance between centroids $k$ and $j$ (separation)                       |

**Lower DB is better.** The index measures the worst-case ratio of (compactness) to (separation) for each cluster. If clusters are compact and well-separated, the ratio is small.

**Calinski-Harabasz Index (variance ratio):**

$$\text{CH} = \frac{\text{between-cluster variance} / (K-1)}{\text{within-cluster variance} / (n-K)}$$

Higher is better. Like an "F-statistic" for clustering.

---

### 7.3 Metric Comparison

| Metric                | Range               | Optimal | Needs labels? | Sensitive to                             |
| --------------------- | ------------------- | ------- | ------------- | ---------------------------------------- |
| **ARI**               | $[-1, 1]$           | Higher  | Yes           | Number of clusters                       |
| **Silhouette**        | $[-1, 1]$           | Higher  | No            | Cluster shape (biased toward convex)     |
| **Davies-Bouldin**    | $[0, \infty)$       | Lower   | No            | Cluster shape (biased toward spherical)  |
| **Calinski-Harabasz** | $[0, \infty)$       | Higher  | No            | Cluster size (biased toward equal-sized) |
| **BIC (GMM)**         | $(-\infty, \infty)$ | Lower   | No            | Model assumptions (Gaussian)             |

> **Practical advice.** Use silhouette as the primary internal metric (it is the most widely used and interpretable). Use ARI when labels exist (benchmarking). Use BIC for GMM model selection. Always visualise the clusters (in 2D via PCA) — no single number captures cluster quality fully.

> **Notebook reference.** Cell 13 reports ARI, silhouette, and Davies-Bouldin for K-means, hierarchical, and GMM on the Iris dataset.

---

## 8. Algorithm Selection Guide

| Data characteristic                           | Best algorithm                               | Why                                     |
| --------------------------------------------- | -------------------------------------------- | --------------------------------------- |
| Spherical, well-separated clusters            | **K-means**                                  | Fast, simple, optimal for this case     |
| Elliptical clusters, different sizes          | **GMM**                                      | Models covariance per cluster           |
| Non-convex clusters (crescents, spirals)      | **DBSCAN / HDBSCAN**                         | Density-based; follows arbitrary shapes |
| Hierarchical structure (groups within groups) | **Agglomerative**                            | Dendrogram reveals nested structure     |
| Large dataset ($n > 10^5$)                    | **Mini-batch K-means** or **HDBSCAN**        | Scalable implementations                |
| Need to detect outliers                       | **DBSCAN / HDBSCAN**                         | Noise labelling built in                |
| Don't know $K$                                | **HDBSCAN** or **GMM + BIC**                 | Automatic $K$ selection                 |
| Need probabilistic assignments                | **GMM**                                      | Soft clustering with responsibilities   |
| High-dimensional data                         | **PCA + K-means** or **spectral clustering** | Reduce dimesionality first ([Week 04](../week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view))    |

> **Default recommendation.** Start with **K-means** (fast, interpretable baseline). If clusters are non-convex → try **DBSCAN**. If you need uncertainty estimates → use **GMM**. Always visualise with PCA or t-SNE before choosing.

---

## 9. Connections to the Rest of the Course

| Week                         | Connection                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **[Week 04](../week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view) (PCA)**            | PCA projections used to visualise clusters in 2D; PCA whitening improves K-means (Section 7 of [Week 04](../week04_dimensionality_reduction/theory.md#7-pca-whitening))                  |
| **[Week 06](../week06_regularization/theory.md#3-ridge-regression-l2-regularisation) (Regularisation)** | Regularised GMMs (adding a small diagonal to $\Sigma_k$) prevent singular covariance — same idea as Ridge regularisation |
| **[Week 07](../../03_probability/week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle) (Likelihood)**     | GMM fitting is MLE with latent variables — the EM algorithm is derived formally                                          |
| **[Week 08](../../03_probability/week08_uncertainty/theory.md#7-bayesian-inference) (Uncertainty)**    | Bayesian GMMs with Dirichlet priors; variational inference as generalised EM                                             |
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#4-fully-connected-networks-mlps) (NNs)**            | Neural network latent representations can be clustered to understand what the network has learned                        |
| **[Week 15](../../05_deep_learning/week15_cnn_representations/theory.md#9-visualising-filters-and-activations) (CNNs)**           | Feature map activations clustered to discover semantic concepts                                                          |
| **[Week 18](../../06_sequence_models/week18_transformers/theory.md#11-attention-patterns-across-layers) (Transformers)**   | Attention heads exhibit cluster-like specialisation across sequence positions                                            |

> **The unifying principle.** Clustering is the simplest form of **latent variable discovery** — finding hidden structure that explains the observed data. Every generative model in the course (VAEs, GMMs, latent diffusion) is a more sophisticated version of this idea.

---

## 10. Notebook Reference Guide

| Cell                      | Section                       | What to observe                                                                            | Theory reference |
| ------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------ | ---------------- |
| 5 (K-means from scratch)  | Lloyd's algorithm + K-means++ | Correct cluster recovery (ARI ≈ 1.0); centroid positions                                   | Section 3.2, 3.4 |
| 7 (Elbow + Silhouette)    | Two diagnostic plots          | Elbow at $K = 4$; silhouette peak at $K = 4$                                               | Section 3.5      |
| 9 (DBSCAN vs K-means)     | Side-by-side on moons/circles | K-means fails; DBSCAN captures non-convex shapes                                           | Section 3.6, 5   |
| 11 (Dendrogram)           | Ward linkage on Iris          | Tallest gap suggests 3 clusters; ARI reported                                              | Section 4.3      |
| 13 (GMM)                  | Soft clustering on Iris       | ARI comparison table: K-means vs hierarchical vs GMM                                       | Section 6.3      |
| Ex.1 (Centroid recovery)  | K-means accuracy              | Centroid positions within 0.5 of known ground truth                                        | Section 3.2      |
| Ex.3 (DBSCAN sweep)       | Parameter sensitivity         | Noise fraction increases with small $\varepsilon$; ARI peaks at intermediate $\varepsilon$ | Section 5.3      |
| Ex.4 (Linkage comparison) | Different linkage methods     | Ward typically best ARI on Iris; single may chain                                          | Section 4.2      |
| Ex.5 (BIC)                | Model selection               | BIC minimised at $K = 3$                                                                   | Section 6.4      |

**Suggested modifications across exercises:**

| Modification                                                   | What it reveals                                                          |
| -------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Run K-means with `n_init=1` (single random init, no K-means++) | Much worse results — illustrates initialisation sensitivity              |
| Add 10% random outliers to `make_blobs`                        | K-means centroids shift; DBSCAN correctly labels noise                   |
| Use `covariance_type='spherical'` in GMM                       | GMM reduces to K-means-like behaviour; ARI may drop on Iris              |
| Scale features before hierarchical clustering                  | Results change significantly for datasets with mixed-scale features      |
| Apply PCA (2 components) before DBSCAN on Iris                 | DBSCAN may outperform K-means — Iris in 2D has a density-based structure |
| Try HDBSCAN (`pip install hdbscan`) on all datasets            | Automatic $K$ selection; handles varying density                         |

---

## 11. Symbol Reference

| Symbol                          | Name                           | Meaning                                                                 |
| ------------------------------- | ------------------------------ | ----------------------------------------------------------------------- |
| $\mathbf{x}_i \in \mathbb{R}^d$ | Data point                     | Feature vector for example $i$                                          |
| $n$                             | Number of data points          | Size of the dataset                                                     |
| $d$                             | Number of features             | Dimensionality                                                          |
| $K$                             | Number of clusters             | Specified by user (K-means, GMM) or determined by algorithm (DBSCAN)    |
| $C_k$                           | Cluster $k$                    | Set of points assigned to cluster $k$                                   |
| $\boldsymbol{\mu}_k$            | Centroid of cluster $k$        | Mean of points in $C_k$                                                 |
| $J$                             | WCSS / Inertia                 | $\sum_k \sum_{\mathbf{x} \in C_k}\|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ |
| $a_i$                           | Intra-cluster distance         | Mean distance from $\mathbf{x}_i$ to its cluster-mates                  |
| $b_i$                           | Nearest-cluster distance       | Mean distance from $\mathbf{x}_i$ to nearest other cluster              |
| $s_i$                           | Silhouette coefficient         | $(b_i - a_i)/\max(a_i, b_i) \in [-1, 1]$                                |
| $\bar{s}$                       | Mean silhouette score          | Average over all points                                                 |
| $S_k$                           | Intra-cluster dispersion       | Average distance from points to centroid in $C_k$                       |
| $d_{kj}$                        | Inter-centroid distance        | $\|\boldsymbol{\mu}_k - \boldsymbol{\mu}_j\|$                           |
| $\text{DB}$                     | Davies-Bouldin Index           | $\frac{1}{K}\sum_k \max_{j \neq k}(S_k + S_j)/d_{kj}$                   |
| $\text{ARI}$                    | Adjusted Rand Index            | Chance-corrected agreement with labels                                  |
| $\varepsilon$                   | Neighbourhood radius           | DBSCAN parameter                                                        |
| MinPts                          | Minimum points                 | DBSCAN parameter                                                        |
| $N_\varepsilon(\mathbf{x})$     | $\varepsilon$-neighbourhood    | $\{\mathbf{y}: d(\mathbf{x},\mathbf{y}) \leq \varepsilon\}$             |
| $\pi_k$                         | Mixing coefficient             | Prior probability of GMM component $k$                                  |
| $\boldsymbol{\Sigma}_k$         | Component covariance           | Covariance matrix of GMM component $k$                                  |
| $\gamma_{ik}$                   | Responsibility                 | $p(z_i = k \mid \mathbf{x}_i)$ — posterior probability of component $k$ |
| $N_k$                           | Effective count                | $\sum_i \gamma_{ik}$ — soft count of points in component $k$            |
| $\ell(\hat{\theta})$            | Log-likelihood                 | $\sum_i \log p(\mathbf{x}_i \mid \hat{\theta})$                         |
| $\text{BIC}$                    | Bayesian Information Criterion | $-2\ell + p\log n$                                                      |
| $\text{AIC}$                    | Akaike Information Criterion   | $-2\ell + 2p$                                                           |
| $p$ (in BIC/AIC)                | Number of parameters           | Free parameters in the model                                            |
| $Z$                             | Linkage matrix                 | $(n-1) \times 4$ matrix encoding merge history                          |

---

## 12. References

1. Lloyd, S. P. (1982). "Least squares quantization in PCM." *IEEE Transactions on Information Theory*, 28(2), 129–137. — The original K-means algorithm (written in 1957, published in 1982).
2. Arthur, D. & Vassilvitskii, S. (2007). "K-means++: The Advantages of Careful Seeding." *SODA*, 1027–1035. — K-means++ initialisation with competitive ratio guarantee.
3. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." *KDD*, 226–231. — DBSCAN.
4. Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates." *PAKDD*, 160–172. — HDBSCAN.
5. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." *JRSS Series B*, 39(1), 1–38. — The EM algorithm.
6. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 9. Springer. — GMMs and EM; detailed derivations.
7. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Chapters 13–14. Springer. — Clustering, hierarchical methods, self-organising maps.
8. Tibshirani, R., Walther, G., & Hastie, T. (2001). "Estimating the Number of Clusters in a Data Set via the Gap Statistic." *JRSS Series B*, 63(2), 411–423. — Gap statistic for choosing $K$.
9. Rousseeuw, P. J. (1987). "Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis." *Journal of Computational and Applied Mathematics*, 20, 53–65. — Silhouette score.
10. Schwarz, G. (1978). "Estimating the Dimension of a Model." *Annals of Statistics*, 6(2), 461–464. — BIC.
11. Scikit-learn User Guide: Clustering. [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html).
12. van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using t-SNE." *JMLR*, 9, 2579–2605. — Useful for visualising cluster structure.
