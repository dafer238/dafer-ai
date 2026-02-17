Week 05 — Probability & Noise (Likelihood)

Overview
Tie loss-based ML to probabilistic modeling. Understand how likelihood assumptions lead to common losses and how to evaluate model fit.

Study
- Likelihood vs loss; MLE principles
- Gaussian noise assumptions and robust alternatives

Practical libraries & tools
- SciPy and statsmodels for likelihood-based estimators
- scikit-learn for baseline models

Datasets & examples
- Simulated regression with Gaussian and heavy-tailed noise
- UCI tabular datasets for real-data MLE experiments

Exercises
1) Derive and implement MLE for Gaussian linear regression.

2) Robust alternatives
   - Fit using Laplace/Student-t noise assumptions and compare residuals.

3) Likelihood surfaces
   - Plot likelihood as a function of parameters to understand multimodality.

Code hint
  # log-likelihood for Gaussian linear model
  def log_lik(params, X, y):
      w, sigma = params[:-1], params[-1]
      mu = X @ w
      return -0.5*np.sum(((y-mu)/sigma)**2) - len(y)*np.log(sigma)

Reading
- Murphy (ML: A Probabilistic Perspective) chapters on likelihood and MLE

Deliverable
- Notebook with MLE derivations, implementations, and comparison to least-squares.
