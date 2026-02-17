Week 03 — Classical ML Foundations

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
