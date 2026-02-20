# dafer-ai

A 20-week deep dive into Artificial Intelligence, Machine Learning, and Deep Learning,
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

## Serving the repository via Uvicorn

A minimal FastAPI application (`server.py`) lives at the top level and
will render any of the markdown files in the workspace as HTML.  This
lets you browse the `README.md`, `theory.md` and other notes using a web
browser over HTTP.

### Quick start

1. install the dependencies in your Python environment:

   ```sh
   pip install fastapi uvicorn markdown
   ```

2. run the service (bind to `0.0.0.0` on a headless Orange Pi if you
   intend to route it elsewhere):

   ```sh
   uvicorn server:app --host 127.0.0.1 --port 8003
   ```

3. visit `http://localhost:8003/` to see the top‑level `README.md`.  Any
   relative links to other `.md` files will work automatically, and links
   to directories will show their `README.md`.

Feel free to add a `static/` directory and uncomment the mount line in
`server.py` if you need to serve images or CSS assets.



```
01_intro/
    week00_ai_landscape/             ← What is ML? Paradigms, vocabulary, the training loop
    week00b_math_and_data/           ← NumPy, derivatives, probability, EDA
02_fundamentals/
    week01_optimization/             ← Gradient descent, loss landscapes
    week02_advanced_optimizers/      ← Momentum, Adam, LR schedules
    week03_linear_models/            ← Linear & logistic regression from scratch
    week04_dimensionality_reduction/ ← PCA, SVD, whitening
    week05_clustering/               ← k-means, DBSCAN, hierarchical, GMM
    week06_regularization/           ← L1/L2, cross-validation
03_probability/
    week07_likelihood/               ← MLE, loss functions as negative log-likelihoods
    week08_uncertainty/              ← Monte Carlo, Bayes, calibration
    week09_time_series/              ← Decomposition, ARIMA, SARIMA, walk-forward CV
    week10_surrogate_models/         ← GP regression, kernels, EI, Bayesian optimisation
04_neural_networks/
    week11_nn_from_scratch/          ← Fully connected NN, backprop
    week12_training_pathologies/     ← Vanishing gradients, BatchNorm, diagnostics
05_deep_learning/
    week13_pytorch_basics/           ← Autograd, nn.Module, training loops
    week14_training_at_scale/        ← DataLoaders, schedulers, GPU
    week15_cnn_representations/      ← CNNs, inductive bias, representation learning
    week16_regularization_dl/        ← Dropout, BatchNorm at scale
06_sequence_models/
    week17_attention/                ← Attention from scratch
    week18_transformers/             ← Full transformer block
07_transfer_learning/
    week19_finetuning/               ← Fine-tuning, adapters, feature extraction
08_deployment/
    week20_deployment/               ← FastAPI, Docker, monitoring
capstone/                            ← End-to-end project
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
