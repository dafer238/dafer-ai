Week 12 — Regularization at Scale

## Prerequisites

- **Week 04** — regularization concepts for linear models (L1/L2, the idea of penalising complexity). Week 12 is the deep-learning counterpart: the goal is identical, the tools are different.
- **Week 11** — the CNN you built is the model you will regularize here.
- **Week 08** — BatchNorm was introduced as a training-stability fix; this week examines it as a regularizer and studies its interaction with Dropout.

## What this week delivers

Deep models overfit aggressively. This week makes you systematic about controlling that: Dropout, BatchNorm, weight decay, augmentation, and ensembles are not independent tricks — they interact, and ablation experiments reveal how. The habit of running controlled ablations is the central skill of this week and carries through to every subsequent experiment.

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

## What comes next

- **Week 13** moves to sequential data and attention; the regularization habits from this week (dropout in attention layers, weight decay) transfer directly.
- **Week 15** — fine-tuning a pretrained model is itself a form of regularization (parameter constraint). The ablation methodology from this week informs how to evaluate fine-tuning strategies.
