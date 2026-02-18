Week 05 — Probability & Noise (Likelihood)

## Prerequisites

- **Week 00b** — Gaussian distribution definition; the preview of MLE in the "From probability to loss functions" section.
- **Week 03** — linear regression, so you can see _why_ MSE is the natural loss under Gaussian assumptions.

## What this week delivers

This week closes the loop opened in Week 00b: MSE is not arbitrary — it is the correct loss when noise is Gaussian. Understanding this lets you choose different loss functions when the noise model is different (heavy-tailed data, count data, binary outcomes). Everything learned here reappears in Week 06 (uncertainty quantification) and beyond.

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

## What comes next

- **Week 06** extends this to uncertainty quantification: credible intervals, calibration, and Bayesian inference.
- Every loss function you encounter from Week 07 onwards is a negative log-likelihood under some noise model.
