# 18-Week AI Study Program – Printable Weekly Checklist

Use this as a **tick-box execution plan**. Each week assumes ~10–12 focused hours. Do not rush checkpoints; understanding beats speed.

---

## Week 0a – The AI Landscape

**Study**

* ☐ The three learning paradigms: supervised, unsupervised, reinforcement
* ☐ Problem taxonomy: regression, classification, clustering, dimensionality reduction, generation
* ☐ The training loop (forward → loss → backward → update)
* ☐ Vocabulary: features, labels, parameters, hyperparameters, loss, gradient, overfitting, underfitting

**Read / Watch**

* ☐ Goodfellow et al. _Deep Learning_ Chapter 1 (free online)
* ☐ 3Blue1Brown "Neural Networks" series (first two videos for visual intuition)
* ☐ Burkov _The Hundred-Page ML Book_ Chapters 1–3

**Checkpoint**

* ☐ Can categorize any common ML problem into the right paradigm
* ☐ Can name the four design choices every ML practitioner makes (architecture, loss, optimizer, regularization)
* ☐ Can draw the training loop without looking at notes

**Deliverable**

* ☐ One-page diagram: ML taxonomy + training loop annotated with terms

---

## Week 0b – Math and Data Foundations

**Study**

* ☐ Vectors, matrices, dot product, matrix multiply (numpy fluency)
* ☐ Derivatives: geometric meaning, partial derivatives, chain rule
* ☐ Probability: Gaussian distribution, mean, variance, likelihood (preview)
* ☐ EDA: shape, missing values, distributions, correlations, target inspection

**Build**

* ☐ Matrix operations and shape management (NumPy exercises)
* ☐ Finite-difference derivative check
* ☐ Gaussian sampling + fitting + plot
* ☐ Full EDA on California Housing dataset

**Checkpoint**

* ☐ Can implement MSE vectorized in NumPy from scratch
* ☐ Can explain what a derivative means geometrically
* ☐ Can explain the Gaussian distribution and why MSE comes from it
* ☐ Can run EDA and state 3 meaningful observations about a dataset

**Read**

* ☐ NumPy Quickstart tutorial
* ☐ 3Blue1Brown Essence of Linear Algebra (chapters 1–5)
* ☐ 3Blue1Brown Essence of Calculus (chapters 1–4)

**Deliverable**

* ☐ Notebook: NumPy ops, derivative check, Gaussian fit, EDA with written observations

---

## Week 1 – Optimization Intuition (Loss as Energy)

> **Prerequisites:** Week 0a + 0b — you need the training loop, derivatives, and NumPy fluency before this week makes sense.

**Study**

* ☐ Loss functions as energy landscapes — visualize minima and basins
* ☐ Gradients as forces; relation to physics intuition
* ☐ Convex vs non-convex geometry and saddle points

**Build**

* ☐ Implement vanilla gradient descent in NumPy (from scratch, no sklearn)
* ☐ Visualize loss surfaces (2D contour and 3D surface)
* ☐ Plot parameter trajectories during optimization

**Checkpoint**

* ☐ Can explain why large learning rates diverge
* ☐ Can explain momentum without equations
* ☐ Can implement the update rule $w \leftarrow w - \eta \nabla L$ from memory

**Mini-project**

* ☐ Estimate thermal parameters of a simple building heat model

---

## Week 2 – Advanced Optimization

> **Prerequisites:** Week 1 — gradient descent and loss landscapes.

**Study**

* ☐ SGD vs Momentum vs RMSProp vs Adam; what each fixes
* ☐ Learning rate schedules: step, cosine, warmup

**Build**

* ☐ Implement Momentum and Adam from scratch in NumPy
* ☐ Compare optimizers on same loss surface; plot trajectories

**Checkpoint**

* ☐ Can write the Adam update rule from memory
* ☐ Understand why Adam converges fast but may generalize worse

**Mini-project**

* ☐ Energy system parameter fitting with noisy data

---

## Week 3 – Classical ML Foundations

> **Prerequisites:** Week 0a (supervised learning concept), Week 0b (matrix multiply, derivatives), Week 1–2 (gradient descent).

**Study**

* ☐ Supervised learning: regression and classification problem setup
* ☐ Ordinary least squares and the normal equations (closed-form solution)
* ☐ Logistic regression and probabilistic interpretation
* ☐ Bias–variance tradeoff

**Build**

* ☐ Linear regression: closed-form (normal equations) and gradient-based — compare both
* ☐ Logistic regression from scratch; compare to sklearn

**Checkpoint**

* ☐ Can diagnose underfitting vs overfitting numerically
* ☐ Can derive the normal equations on paper

**Mini-project**

* ☐ Simple return prediction model (linear factors)

---

## Week 4 – Regularization & Validation

> **Prerequisites:** Week 3 (linear models, bias-variance).

**Study**

* ☐ L1 (Lasso) vs L2 (Ridge) regularization — penalty, geometry, sparsity effects
* ☐ Cross-validation: k-fold, stratified, time-series walk-forward

**Build**

* ☐ Ridge and Lasso implementations; compare with sklearn baselines
* ☐ Walk-forward validation for time-series data

**Checkpoint**

* ☐ Can explain regularization as a constraint (geometry) and as entropy control
* ☐ Can choose between L1 and L2 given a problem description

**Mini-project**

* ☐ Stable factor selection for financial returns

---

## Week 5 – Probability & Noise

> **Prerequisites:** Week 0b (Gaussian distribution, likelihood preview), Week 3 (linear regression).

**Study**

* ☐ Likelihood vs loss — the probabilistic motivation for every loss function
* ☐ Maximum likelihood estimation (MLE) principle
* ☐ Gaussian assumptions and robust alternatives (Laplace, Student-t)

**Build**

* ☐ MLE for Gaussian linear regression — derive and implement
* ☐ Fit Laplace and Student-t noise models; compare residuals

**Checkpoint**

* ☐ Can derive squared-error loss from Gaussian MLE on paper
* ☐ Can explain why heavy-tailed noise calls for a different loss

**Mini-project**

* ☐ Sensor noise modeling with uncertainty bounds

---

## Week 6 – Uncertainty & Statistics

> **Prerequisites:** Week 5 (probability, distributions).

**Study**

* ☐ Aleatoric (data) vs epistemic (model) uncertainty
* ☐ Bayesian vs frequentist perspectives; credible vs confidence intervals
* ☐ Calibration, sharpness, and predictive intervals

**Build**

* ☐ Monte Carlo simulations for expectation and intervals
* ☐ Bootstrap confidence intervals
* ☐ Bayesian linear regression with PyMC (posterior + credible intervals)

**Checkpoint**

* ☐ Can explain the difference between aleatoric and epistemic uncertainty
* ☐ Can explain when probabilistic models fail

**Mini-project**

* ☐ Industrial measurement uncertainty analysis

---

## Week 7 – Neural Networks From Scratch

> **Prerequisites:** Week 0b (chain rule), Week 3 (linear models as single-layer NNs), Week 1–2 (gradient descent).

**Study**

* ☐ Neurons, activation functions (ReLU, tanh, sigmoid), weight initialization
* ☐ Chain rule unrolled as backpropagation — derive on paper

**Build**

* ☐ Fully connected NN in NumPy (forward + backward + update loop)
* ☐ Gradient checking via finite differences

**Checkpoint**

* ☐ Can derive backprop on paper for a 2-layer network

**Mini-project**

* ☐ Nonlinear energy demand model

---

## Week 8 – Training Pathologies

> **Prerequisites:** Week 7 (neural networks, gradients through layers).

**Study**

* ☐ Vanishing/exploding gradients — why they happen and layer-by-layer diagnosis
* ☐ Activation saturation (sigmoid/tanh vs ReLU)
* ☐ BatchNorm and LayerNorm

**Build**

* ☐ Log gradient L2 norms per layer during training and plot
* ☐ Compare sigmoid / tanh / ReLU / LeakyReLU on deep nets

**Checkpoint**

* ☐ Can debug unstable training by reading gradient norm plots
* ☐ Can explain why ReLU largely solved vanishing gradients

**Mini-project**

* ☐ Demand forecasting stress tests

---

## Week 9 – PyTorch Fundamentals

> **Prerequisites:** Week 7–8 (NN from scratch) — you are now re-implementing what you already understand.

**Study**

* ☐ Tensors, broadcasting, autograd internals, computational graphs
* ☐ `torch.nn`, `DataLoader`, optimizers, checkpointing

**Build**

* ☐ Re-implement your NumPy NN in PyTorch; verify identical outputs on small data
* ☐ Compare autograd gradients to your finite-difference checks from Week 7

**Checkpoint**

* ☐ Understand what autograd stores in the computational graph
* ☐ Can explain what `.backward()` does without looking at docs

**Mini-project**

* ☐ Volatility regime classifier

---

## Week 10 – Efficient Training

> **Prerequisites:** Week 9 (PyTorch basics).

**Study**

* ☐ Batching and stochastic vs full-batch gradient descent tradeoffs
* ☐ GPU utilization, mixed-precision basics
* ☐ DataLoaders, learning rate schedulers

**Build**

* ☐ DataLoader pipeline with custom dataset; compare batch sizes
* ☐ Learning rate scheduler (cosine, step) and effect on convergence

**Checkpoint**

* ☐ Can explain batch size tradeoffs (gradient noise, generalization, memory)

**Mini-project**

* ☐ Improved regime detection model

---

## Week 11 – Representation Learning

> **Prerequisites:** Week 9–10 (PyTorch). Note: CNNs are one example of representation learning — the idea applies to all deep models.

**Study**

* ☐ Inductive bias: why architecture choice encodes assumptions about data structure
* ☐ Feature hierarchies in deep models
* ☐ CNN basics: convolution, pooling, receptive field

**Build**

* ☐ Implement a small CNN in PyTorch; inspect learned filters
* ☐ Compare CNN vs MLP on a structured input task

**Checkpoint**

* ☐ Can explain inductive bias with a concrete example (translation invariance in CNNs)
* ☐ Can explain why deep models overfit more easily than shallow ones

**Mini-project**

* ☐ Fault detection from sensor windows

---

## Week 12 – Regularization at Scale

> **Prerequisites:** Week 4 (regularization concepts), Week 11 (deep models context).

**Study**

* ☐ Dropout: mechanism, effect on implicit ensemble, when to use
* ☐ BatchNorm: normalization, training vs inference mode differences
* ☐ Data augmentation as a regularizer

**Build**

* ☐ CNN with Dropout and BatchNorm; ablation to show contribution of each

**Checkpoint**

* ☐ Know when regularization hurts (small datasets, distribution shift)
* ☐ Can explain BatchNorm training vs inference mode difference

**Mini-project**

* ☐ Robust industrial fault classifier

---

## Week 13 – Sequential Models & Attention

> **Prerequisites:** Week 9–10 (PyTorch). Attention is introduced here as a standalone concept before transformers — read Week 0a's "sequence modeling" entry first.

**Study**

* ☐ Temporal dependencies and why RNNs struggle with long-range dependencies
* ☐ Attention concept: queries, keys, values, and scaled dot-product attention

**Build**

* ☐ Implement scaled dot-product attention from scratch in NumPy/PyTorch
* ☐ Visualize attention weight matrices

**Checkpoint**

* ☐ Can explain attention intuitively (which inputs does the model focus on?)
* ☐ Can implement the attention formula $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ from memory

**Mini-project**

* ☐ Time-series energy price forecasting

---

## Week 14 – Transformers

> **Prerequisites:** Week 13 (attention). Transformers stack attention with positional encoding and feed-forward blocks.

**Study**

* ☐ Positional encoding: why transformers need it (unlike RNNs)
* ☐ Multi-head attention and the full transformer block (pre/post-norm variants)
* ☐ Scaling laws and emergent behavior

**Build**

* ☐ Implement a small transformer (encoder only) from scratch in PyTorch

**Checkpoint**

* ☐ Can explain why transformers parallelize better than RNNs
* ☐ Understand scaling limits (compute, data, context length)

**Mini-project**

* ☐ Transformer vs LSTM comparison on a time-series task

---

## Week 15 – Fine-Tuning & Transfer Learning

> **Prerequisites:** Week 14 (transformers for NLP transfer) or Week 11 (CNNs for vision transfer).

**Study**

* ☐ Transfer learning: why pretrained features transfer across domains
* ☐ Feature extraction vs full fine-tuning — when to use each
* ☐ Parameter-efficient methods: adapters, LoRA, prefix tuning

**Build**

* ☐ Fine-tune a pretrained model on a small downstream task
* ☐ Compare feature-extraction vs full fine-tuning across data-size regimes

**Checkpoint**

* ☐ Know when NOT to fine-tune (small target data + large domain gap → feature extraction)
* ☐ Can explain what covariate shift is and how it affects transfer

**Mini-project**

* ☐ Domain-adapted forecasting model

---

## Week 16 – Deployment & Capstone

> **Prerequisites:** All previous weeks. This week ties the course together.

**Study**

* ☐ Inference constraints: latency, memory, throughput — and why training metrics are insufficient
* ☐ Model serving: FastAPI, TorchServe, containerization
* ☐ Monitoring: metrics, logs, health checks, data drift detection

**Build**

* ☐ Package trained model as a FastAPI endpoint
* ☐ Write a `Dockerfile` for reproducible deployment
* ☐ Add `/health` endpoint and request latency logging

**Checkpoint**

* ☐ Can explain the full ML lifecycle: data → train → evaluate → deploy → monitor
* ☐ End-to-end system understanding

**Capstone (choose one)**

* ☐ Energy optimization system
* ☐ Quantitative trading signal with uncertainty
* ☐ Industrial fault prediction pipeline

---

## Final Validation

* ☐ Can explain the training loop (forward → loss → backward → update) without notes
* ☐ Can categorize any ML problem into supervised / unsupervised / reinforcement
* ☐ Can explain training dynamics (gradient flow, vanishing gradients, batch size effects) without slides
* ☐ Can debug models without guessing: check gradient norms, validation curves, calibration
* ☐ Can connect AI behavior to physical/financial intuition (loss as energy, noise as uncertainty)
* ☐ Can implement linear regression, a 2-layer NN, and attention from scratch in NumPy
* ☐ Can deploy a model as an API in a Docker container

**If all boxes are checked: you are no longer a user of AI — you are a builder.**
