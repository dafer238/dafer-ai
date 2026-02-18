# dafer-ai

An 18-week deep dive into Artificial Intelligence, Machine Learning, and Deep Learning,
with an emphasis on **first principles**, **physical intuition**, and **real-world systems**.

## Focus Areas
- AI landscape and problem taxonomy
- Math and data foundations (linear algebra, calculus, probability, EDA)
- Optimization & learning dynamics
- Machine learning from scratch
- Deep learning with PyTorch
- Transformers and attention
- Applications to:
  - Quantitative finance
  - Energy systems
  - Thermodynamics
  - Industrial engineering

## Structure

```
01_intro/
    week00_ai_landscape/        ← What is ML? Paradigms, vocabulary, the training loop
    week00b_math_and_data/      ← NumPy, derivatives, probability, EDA
02_fundamentals/
    week01_optimization/        ← Gradient descent, loss landscapes
    week02_advanced_optimizers/ ← Momentum, Adam, LR schedules
    week03_linear_models/       ← Linear & logistic regression from scratch
    week04_regularization/      ← L1/L2, cross-validation
03_probability/
    week05_likelihood/          ← MLE, loss functions as negative log-likelihoods
    week06_uncertainty/         ← Monte Carlo, Bayes, calibration
04_neural_networks/
    week07_nn_from_scratch/     ← Fully connected NN, backprop
    week08_training_pathologies/← Vanishing gradients, BatchNorm, diagnostics
05_deep_learning/
    week09_pytorch_basics/      ← Autograd, nn.Module, training loops
    week10_training_at_scale/   ← DataLoaders, schedulers, GPU
    week11_cnn_representations/ ← CNNs, inductive bias, representation learning
    week12_regularization_dl/   ← Dropout, BatchNorm at scale
06_sequence_models/
    week13_attention/           ← Attention from scratch
    week14_transformers/        ← Full transformer block
07_transfer_learning/
    week15_finetuning/          ← Fine-tuning, adapters, feature extraction
08_deployment/
    week16_deployment/          ← FastAPI, Docker, monitoring
capstone/                       ← End-to-end project
```

Each week has:
- A `README.md` with: prerequisites, study material, exercises, code hints, readings, and forward links to the next week
- A `starter.ipynb` Jupyter notebook for hands-on work

## Philosophy
AI is treated as:
> Optimization under uncertainty in complex systems

No copy-paste. No magic. Everything explained.

Start with `01_intro/week00_ai_landscape/README.md` before anything else.

## Capstone
The capstone project integrates forecasting, uncertainty estimation,
and deployment in a real-world-inspired domain.
