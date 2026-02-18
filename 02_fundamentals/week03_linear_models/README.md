Week 03 — Classical ML Foundations

## Prerequisites

- **Week 00a** — supervised learning problem setup; the concept of features, labels, training/validation/test split.
- **Week 00b** — matrix multiply (the forward pass of linear regression is literally `X @ w`), derivatives, and EDA.
- **Week 01–02** — gradient descent, because you will compare gradient-based and closed-form solutions.

## What this week delivers

Linear and logistic regression are not "simple" models — they are the foundation every more complex model is built on. A neural network is just stacked linear models with nonlinearities. Understanding these models deeply, including the bias-variance tradeoff, is a prerequisite to diagnosing any model in the course.

Overview
Solidify foundations in linear and logistic models, understand closed-form vs iterative solutions, and connect statistical interpretations to ML practice.

Study
- Ordinary least squares, normal equations, bias–variance tradeoff
- Logistic regression and probabilistic interpretation

Practical libraries & tools
- NumPy, SciPy, scikit-learn for baseline implementations
- Statsmodels for statistical summaries

Datasets & examples
- Synthetic linear data with controlled noise
- UCI datasets (e.g., Wine, Boston) for small regression/classification

Exercises
1) Derive and implement normal equations and compare to gradient-based fit.

2) Implement logistic regression and compare to `sklearn.linear_model.LogisticRegression`.

3) Bias–variance study
   - Vary model complexity (polynomial features) and plot training vs validation error.

Code hint
  # closed-form linear regression
  XTX = X.T @ X
  w = np.linalg.solve(XTX + lam*np.eye(X.shape[1]), X.T @ y)

Readings
- Classic stats references for OLS and logistic regression; scikit-learn user guide.

Deliverable
- Notebook with derivations, code, plots, and a short mini-project applying linear models to a small real dataset.

## What comes next

- **Week 04** adds regularization to the models built here (Ridge = L2 on the weights of linear regression).
- **Week 05** provides the probabilistic justification for least-squares loss (Gaussian MLE).
- **Week 07** re-interprets linear regression as a single-layer neural network without activation function.
