Week 04 — Regularization & Validation

## Prerequisites

- **Week 03** — linear/logistic regression. Regularization is a modification of those models.
- **Week 00a** — overfitting/underfitting vocabulary; train/validation/test split concept.

## What this week delivers

Regularization is the primary tool for making models generalize. The ideas here (penalizing complexity, using held-out data to select hyperparameters) recur in every subsequent week — dropout (Week 12), BatchNorm (Week 12), early stopping (Week 09), and fine-tuning frozen layers (Week 15) are all forms of regularization.

Overview
Learn techniques to control overfitting and select models robustly. Emphasize practical cross-validation and visualization.

Study
- L1 vs L2 regularization, elastic net, dropout concepts
- Cross-validation strategies: k-fold, stratified, time-series-aware CV

Practical libraries & tools
- scikit-learn for regularized models and CV tools
- Matplotlib/Seaborn for validation curves

Datasets & examples
- Synthetic data with correlated features to demonstrate L1 feature selection
- Financial returns toy data for stability tests

Exercises
1) Implement Ridge and Lasso (coordinate descent or use sklearn) and compare.

2) Cross-validation experiments
   - Use k-fold to select lambda and plot validation curves.

3) Time-series CV
   - Implement walk-forward validation for time-aware problems.

Code hint
  from sklearn.linear_model import RidgeCV
  clf = RidgeCV(alphas=[0.1,1.0,10.0], cv=5)
  clf.fit(X, y)

Reading
- Papers and chapters about regularization; sklearn docs on linear models and CV.

Deliverable
- Notebook with experiments showing effects of regularization and how to choose hyperparameters defensibly.

## What comes next

- **Week 12** revisits regularization for deep learning: Dropout and BatchNorm replace L1/L2 as the primary tools.
- **Week 15** applies the same concept to transfer learning: freezing layers is regularization by parameter constraint.
