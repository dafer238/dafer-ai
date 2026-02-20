# Week 07 — Surrogate Models & Gaussian Processes

## Prerequisites

- **Week 05 (likelihood)** — MLE; probabilistic view of regression.
- **Week 06 (uncertainty)** — Bayesian linear regression, posterior distributions.
- **Week 03 (linear models)** — kernel trick intuition from polynomial features.

## What this week delivers

A **surrogate model** is a cheap-to-evaluate approximation of an expensive black-box function — think computational simulation, physical experiment, or deep-network training run. Gaussian Processes (GPs) are the Bayesian workhorse for this task: they give not just a prediction but a calibrated uncertainty estimate at every point, which is exploited by acquisition functions (such as Expected Improvement) to guide efficient search.

GPs also connect directly to kernel methods (SVMs, kernel ridge regression) and to attention mechanisms in transformers, where the attention matrix is a kernel Gram matrix.

## Overview

Derive GP regression from first principles, implement it with both NumPy and sklearn, explore kernel selection, and build a simple Bayesian optimisation loop using Expected Improvement.

## Study

- Gaussian Process prior: mean function, covariance (kernel) function
- GP posterior: closed-form posterior mean and variance
- Common kernels: RBF / squared-exponential, Matérn, periodic, linear
- Hyperparameter optimisation (marginal likelihood maximisation)
- Surrogate-based optimisation: acquisition functions (EI, UCB)
- Expected Improvement (EI) derivation and implementation

## Practical libraries & tools

- NumPy for from-scratch GP
- `sklearn.gaussian_process.GaussianProcessRegressor`, `RBF`, `Matern`, `WhiteKernel`
- Matplotlib for mean ± confidence bands
- SciPy (`scipy.stats.norm`) for EI closed-form

## Datasets & examples

- **1D noisy sine** — intuition: GP prior/posterior, confidence bands, interpolation
- **2D Branin function** — benchmark for surrogate-based optimisation

## Exercises

1. **GP from scratch** — implement `gp_regression(X_train, y_train, X_test, kernel_fn, noise)` using the closed-form posterior equations; verify that the posterior mean matches sklearn's `GaussianProcessRegressor`.

2. **Kernel comparison** — fit GP with RBF, Matérn(ν=1.5), and periodic kernels on a seasonal 1D function; compare log-marginal-likelihoods and visual fits.

3. **Coverage check** — generate 1000 test points; report the fraction that falls inside the 95% credible interval (should be ≈ 0.95).

4. **Expected Improvement** — implement `expected_improvement(X_cand, X_obs, y_obs, gp_model, xi=0.01)`; demonstrate that it balances exploration and exploitation on a 1D example.

5. **Bayesian optimisation** — use GP + EI to minimise the Branin function in ≤ 30 evaluations; compare to random search; plot the optimisation trace.

## Code hints

```python
# GP posterior (noise-free observations by default)
def gp_posterior(X_train, y_train, X_test, kernel_fn, sigma_n=1e-4):
    K    = kernel_fn(X_train, X_train) + sigma_n**2 * np.eye(len(X_train))
    K_s  = kernel_fn(X_train, X_test)
    K_ss = kernel_fn(X_test, X_test)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu    = K_s.T @ alpha
    v     = np.linalg.solve(L, K_s)
    sigma2 = np.diag(K_ss) - np.einsum('ij,ij->j', v, v)
    return mu, np.sqrt(np.maximum(sigma2, 0))

# Expected Improvement
from scipy.stats import norm
def expected_improvement(X_cand, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_cand, return_std=True)
    Z = (mu - y_best - xi) / (sigma + 1e-9)
    return (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
```

## Deliverables

- [ ] GP from scratch implementation matching sklearn (exercise 1).
- [ ] Kernel comparison table (log-marginal-likelihood) + visual fits.
- [ ] Coverage plot: credible interval calibration.
- [ ] EI acquisition function plot (exploration vs exploitation).
- [ ] Bayesian optimisation trace on Branin: GP+EI vs random search.

## What comes next

- **Week 06 (uncertainty)** — GPs are a special case of Bayesian non-parametric models; compare to the Bayesian linear regression posterior.
- **Week 14 (transformers)** — the attention matrix is a scaled dot-product kernel; GP intuition helps interpret attention as kernel regression.
- **Week 16 (deployment)** — surrogate models are used in hyperparameter tuning (Bayesian HPO), a key deployment concern.
