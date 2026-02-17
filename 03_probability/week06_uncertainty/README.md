Week 06 — Uncertainty & Statistics

Overview
This week focuses on quantifying uncertainty in predictions, distinguishing aleatoric (data) vs epistemic (model) uncertainty, and learning practical diagnostics (calibration, credible/confidence intervals, posterior predictive checks).

Study (concepts)
- Aleatoric vs epistemic uncertainty — when each matters and how to estimate them
- Bayesian vs frequentist perspectives; credible vs confidence intervals
- Calibration, sharpness, and predictive intervals

Practical libraries & tools
- NumPy / SciPy for simulations and bootstrapping
- scikit-learn (calibration_curve, `calibration` tools)
- PyMC3 or NumPyro for Bayesian inference and MCMC / SVI workflows
- Prophet / statsmodels for simple time-series baselines

Datasets & synthetic examples
- Simulated heteroscedastic regression: generate x, noise var = f(x)
- UCI / Kaggle tabular datasets (add synthetic noise) to test calibration
- Any industrial sensor logs you have — add noise and test interval coverage

Exercises (step-by-step)
1) Monte Carlo basics
   - Task: estimate E[f(X)] via Monte Carlo and compute empirical intervals.
   - Code hint: draw many samples, compute mean and percentiles (2.5%, 97.5%).

2) Bootstrap
   - Task: implement nonparametric bootstrap to get confidence intervals for a statistic (mean, regression slope).
   - Practice: compare bootstrap intervals vs analytic intervals on Gaussian data.

3) Bayesian linear regression (small)
   - Task: implement Bayesian linear regression with PyMC3 or NumPyro; obtain posterior of weights and posterior predictive distribution.
   - Practice: compute 95% credible intervals and posterior predictive checks (PPC).

4) Calibration & reliability diagrams
   - Task: train a simple regressor/classifier, then plot calibration curve and reliability diagram.
   - Tools: `sklearn.calibration.calibration_curve` (for probabilistic classifiers) and custom bins for regression prediction intervals.

5) Heteroscedastic modeling
   - Task: build a model that predicts both mean and variance (e.g., predict log-variance), and evaluate whether prediction intervals have correct coverage.

Advanced topics / further practice
- Conformal prediction for distribution-free prediction intervals
- Bayesian model averaging and marginal likelihood approximations
- Variational inference (SVI) for scalable Bayesian methods using NumPyro or Pyro

Concrete example snippets
- Monte Carlo mean + percentile intervals (Python):

  import numpy as np
  samples = np.random.normal(loc=0.0, scale=1.0, size=10000)
  mean = samples.mean()
  lower, upper = np.percentile(samples, [2.5, 97.5])

- Simple bootstrap (Python):

  def bootstrap_stat(x, stat_fn, n_boot=1000):
      n = len(x)
      stats = []
      for _ in range(n_boot):
          sample = np.random.choice(x, size=n, replace=True)
          stats.append(stat_fn(sample))
      return np.percentile(stats, [2.5, 97.5])

Recommended readings & tutorials
- Chapter on Uncertainty and Bayesian inference in Murphy's "Machine Learning: A Probabilistic Perspective"
- "Bayesian Methods for Hackers" (practical PyMC examples)
- Scikit-learn docs: calibration and model evaluation
- NumPyro / PyMC3 tutorials for hands-on Bayesian modeling

Deliverable suggestions
- Notebook 1: Monte Carlo and bootstrap exercises with visualizations and a short write-up of results.
- Notebook 2: Bayesian linear regression on a toy dataset, including PPCs and calibration assessment.
- Mini-project: take a small real dataset, add synthetic noise or corruptions, and produce a short report comparing methods for uncertainty quantification.

Notes
- Emphasize reproducible notebooks, clear visual diagnostics (coverage plots, calibration curves), and short writeups explaining when methods succeed or fail.
