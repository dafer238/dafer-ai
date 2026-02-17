Week 12 — Regularization at Scale

Overview
Study regularization techniques specific to deep models and how to design experiments (ablations) to measure their effects.

Study
- Dropout, BatchNorm, weight decay, data augmentation, ensembles

Practical libraries & tools
- PyTorch, Albumentations for augmentation, tensorboard for tracking experiments

Exercises
1) Implement dropout and weight decay in a CNN and measure validation curves.

2) Data augmentation study
   - Apply augmentations and compare generalization across holdout sets.

3) Ensemble methods
   - Train multiple seeds and average predictions; compare to single-model uncertainty.

Reading
- Papers on dropout/batchnorm and practical guides to augmentation.

Deliverable
- Notebook with ablation tables and clear recommendations for regularization in your experiments.
