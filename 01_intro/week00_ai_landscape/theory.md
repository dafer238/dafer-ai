# The AI Landscape: A Rigorous Introduction

## Table of Contents

- [The AI Landscape: A Rigorous Introduction](#the-ai-landscape-a-rigorous-introduction)
  - [Table of Contents](#table-of-contents)
  - [1. Notation Conventions](#1-notation-conventions)
  - [2. From Classical Programming to Machine Learning](#2-from-classical-programming-to-machine-learning)
  - [3. The Three Learning Paradigms](#3-the-three-learning-paradigms)
    - [3.1 Supervised Learning](#31-supervised-learning)
    - [3.2 Unsupervised Learning](#32-unsupervised-learning)
    - [3.3 Reinforcement Learning](#33-reinforcement-learning)
  - [4. The Supervised Learning Framework](#4-the-supervised-learning-framework)
    - [4.1 Data](#41-data)
    - [4.2 Model](#42-model)
    - [4.3 Loss Function](#43-loss-function)
    - [4.4 Optimisation](#44-optimisation)
    - [4.5 Generalisation](#45-generalisation)
  - [5. Unsupervised Learning](#5-unsupervised-learning)
    - [5.1 Clustering](#51-clustering)
    - [5.2 Dimensionality Reduction](#52-dimensionality-reduction)
    - [5.3 Density Estimation](#53-density-estimation)
    - [5.4 Connections Between Paradigms](#54-connections-between-paradigms)
  - [6. Reinforcement Learning](#6-reinforcement-learning)
  - [7. The Training Loop](#7-the-training-loop)
  - [8. What Is a Model?](#8-what-is-a-model)
  - [9. Hypothesis Spaces and Inductive Bias](#9-hypothesis-spaces-and-inductive-bias)
    - [9.1 The Hypothesis Space](#91-the-hypothesis-space)
    - [9.2 Inductive Bias](#92-inductive-bias)
    - [9.3 Capacity, Underfitting, and Overfitting](#93-capacity-underfitting-and-overfitting)
  - [10. The Four Practitioner Choices](#10-the-four-practitioner-choices)
    - [10.1 Architecture](#101-architecture)
    - [10.2 Loss Function](#102-loss-function)
    - [10.3 Optimiser](#103-optimiser)
    - [10.4 Regularisation](#104-regularisation)
  - [11. The Evaluation Mindset](#11-the-evaluation-mindset)
    - [Data Splits](#data-splits)
    - [Key Vocabulary](#key-vocabulary)
  - [12. References](#12-references)

---

## 1. Notation Conventions

The following notation is used consistently throughout this course. Familiarising oneself with these conventions before proceeding will reduce cognitive load in every subsequent week.

| Symbol                               | Meaning                                                                      |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| $x, y, z$                            | Scalars (lowercase italic)                                                   |
| $\mathbf{x}, \mathbf{w}, \mathbf{b}$ | Vectors (lowercase bold). Vectors are column vectors unless stated otherwise |
| $X, W, A$                            | Matrices (uppercase italic)                                                  |
| $\mathbf{x}_i$                       | The $i$-th data point (a vector)                                             |
| $x_{ij}$                             | Element in row $i$, column $j$ of matrix $X$                                 |
| $\theta$                             | Generic parameter vector; $\theta = (\mathbf{w}, b)$ when context is clear   |
| $n$                                  | Number of training samples                                                   |
| $d$                                  | Number of input features (dimensionality)                                    |
| $K$                                  | Number of classes (classification) or clusters                               |
| $\eta$                               | Learning rate                                                                |
| $\lambda$                            | Regularisation strength                                                      |
| $\mathcal{L}(\theta)$                | Loss function (a scalar function of parameters)                              |
| $\nabla_\theta \mathcal{L}$          | Gradient of the loss with respect to $\theta$                                |
| $\hat{y}$                            | Model prediction (the "hat" denotes an estimate)                             |
| $\mathcal{D}$                        | A dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$                                |
| $p(\cdot)$                           | Probability density or mass function                                         |
| $\mathbb{E}[\cdot]$                  | Expected value                                                               |
| $\sim$                               | "Distributed as" (e.g., $\epsilon \sim \mathcal{N}(0, \sigma^2)$)            |
| $\mathbb{R}^d$                       | The $d$-dimensional real coordinate space                                    |
| $\|\mathbf{x}\|$                     | The $L_2$ (Euclidean) norm of $\mathbf{x}$, unless otherwise subscripted     |
| $\odot$                              | Element-wise (Hadamard) product                                              |
| $\circ$                              | Function composition: $(g \circ h)(\mathbf{x}) = g(h(\mathbf{x}))$           |

**Index conventions.** Subscript $i$ indexes samples ($i = 1, \ldots, n$), subscript $j$ indexes features ($j = 1, \ldots, d$), and subscript $k$ indexes classes or clusters ($k = 1, \ldots, K$). Subscript $t$ is reserved for iteration steps (optimisation) or time steps (time series).

---

## 2. From Classical Programming to Machine Learning

In classical programming, a human writes explicit rules that transform data into answers:

$$\text{rules} + \text{data} \longrightarrow \text{answers}$$

In machine learning, the paradigm is inverted. Given data and corresponding answers, the system discovers rules automatically:

$$\text{data} + \text{answers} \longrightarrow \text{rules (learned)}$$

Formally, the "rules" are encoded in a parameterised function $f_\theta$, where $\theta \in \mathbb{R}^p$ is a vector of $p$ learnable parameters. The system adjusts $\theta$ by minimising a **loss function** $\mathcal{L}(\theta)$ — a scalar that measures prediction error. The adjustment procedure is called **optimisation** (the subject of [Weeks 01](../../02_fundamentals/week01_optimization/theory.md#3-loss-functions)–02).

**Example.** Suppose one wishes to predict energy demand $y$ from temperature $x$. A classical program might hard-code a lookup table. A machine learning approach defines $f_\theta(x) = \theta_1 x + \theta_0$, collects historical $(x_i, y_i)$ pairs, and finds $\theta^*$ such that:

$$\theta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

This simple example already contains every element of the ML pipeline: a model ($f_\theta$), a loss (mean squared error), and an optimisation target ($\arg\min$).

> **Intuition.** Think of the parameters $\theta$ as knobs on a control panel. Each knob setting produces a different input–output mapping. The loss function is a meter that reads "how wrong you are." Optimisation is the systematic process of turning the knobs to make the meter read as close to zero as possible. The remarkable insight of ML is that this knob-turning can be automated via calculus — the gradient tells you *which direction* to turn each knob, and *by how much*.

> **Suggested experiment.** Before [Week 01](../../02_fundamentals/week01_optimization/theory.md#9-convergence-diagnostics), try the following in Python: define $f(w) = (w - 3)^2$, plot it, and manually perform gradient descent starting at $w = 0$ with learning rate $\eta = 0.1$. Observe how the updates $w \leftarrow w - \eta \cdot 2(w - 3)$ move $w$ toward $3$. Vary $\eta$ to see overshooting ($\eta = 1.5$) and slow convergence ($\eta = 0.01$).

---

## 3. The Three Learning Paradigms

Machine learning is conventionally divided into three paradigms, distinguished by the nature of the supervision signal.

### 3.1 Supervised Learning

The learner has access to a training set of labelled pairs $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$, where $\mathbf{x}_i \in \mathbb{R}^d$ is an input (feature vector) and $y_i$ is the corresponding target. The goal is to learn a function $f_\theta : \mathbb{R}^d \to \mathcal{Y}$ that generalises to unseen inputs.

| Task                           | Target Space $\mathcal{Y}$ | Loss (typical)            | Example                  |
| ------------------------------ | -------------------------- | ------------------------- | ------------------------ |
| **Regression**                 | $\mathbb{R}$               | Mean Squared Error        | Energy demand prediction |
| **Binary Classification**      | $\{0, 1\}$                 | Binary Cross-Entropy      | Fault detection          |
| **Multi-class Classification** | $\{1, \ldots, K\}$         | Categorical Cross-Entropy | Digit recognition        |
| **Structured Prediction**      | Sequence of tokens         | Sequence loss             | Machine translation      |

**Analogy.** The model is a student; the labelled data is a teacher who marks each answer as correct or incorrect. The loss function quantifies how badly the student performed.

> **When to use supervised learning.** Whenever labelled data is available and the task is clearly defined as "predict $y$ from $\mathbf{x}$." The key constraint is that labels must exist and be trustworthy. In practice, obtaining high-quality labels is often the bottleneck — this motivates unsupervised and semi-supervised approaches.

### 3.2 Unsupervised Learning

The learner has access only to inputs $\{\mathbf{x}_i\}_{i=1}^{n}$ — no labels. The goal is to discover structure: groups, compressed representations, or probability distributions.

| Task                         | Objective                                  | Example Algorithm            |
| ---------------------------- | ------------------------------------------ | ---------------------------- |
| **Clustering**               | Partition data into groups                 | K-Means, DBSCAN, GMM         |
| **Dimensionality Reduction** | Find low-dimensional structure             | PCA, t-SNE, autoencoders     |
| **Density Estimation**       | Estimate $p(\mathbf{x})$                   | Kernel density estimation    |
| **Anomaly Detection**        | Identify outliers                          | Isolation Forest             |
| **Generative Modelling**     | Sample new $\mathbf{x} \sim p(\mathbf{x})$ | VAEs, GANs, diffusion models |

**Analogy.** The model is a scientist given a pile of unlabelled specimens; it must discover categories by itself.

> **When to use unsupervised learning.** When labels are absent or the goal is exploratory — discovering groups in customer data, compressing high-dimensional sensor readings for visualisation, or detecting anomalies without prior examples of failures. Unsupervised methods are also used as **preprocessing** for supervised tasks (e.g., PCA before regression, or embeddings from autoencoders as features).

### 3.3 Reinforcement Learning

An **agent** interacts with an **environment** over discrete time steps $t = 0, 1, 2, \ldots$. At each step, the agent observes a state $s_t$, takes an action $a_t$, receives a reward $r_t$, and transitions to a new state $s_{t+1}$. The agent's goal is to learn a **policy** $\pi(a \mid s)$ that maximises the expected cumulative (discounted) reward:

$$J(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t\right], \quad \gamma \in [0, 1)$$

Reinforcement learning is not covered in this course, but the concepts of function approximation (neural networks) and optimisation that are developed here are prerequisites for RL.

---

## 4. The Supervised Learning Framework

Since the majority of this course concerns supervised learning, the framework deserves careful formal treatment.

### 4.1 Data

A dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ is drawn independently and identically distributed (i.i.d.) from an unknown joint distribution $p(\mathbf{x}, y)$. The matrix form is:

$$X = \begin{bmatrix} \mathbf{x}_1^\top \\ \vdots \\ \mathbf{x}_n^\top \end{bmatrix} \in \mathbb{R}^{n \times d}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ \vdots \\ y_n \end{bmatrix} \in \mathbb{R}^n$$

where $n$ is the number of samples and $d$ is the number of features.

### 4.2 Model

A model is a parameterised family of functions $\{f_\theta : \theta \in \Theta\}$. Examples:

- **Linear regression:** $f_\theta(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$, where $\theta = (\mathbf{w}, b) \in \mathbb{R}^{d+1}$.
- **Neural network:** $f_\theta = g_L \circ g_{L-1} \circ \cdots \circ g_1$, a composition of affine transformations and nonlinearities.

### 4.3 Loss Function

The loss $\mathcal{L}(\theta; \mathcal{D})$ quantifies prediction error. Common choices:

$$\text{MSE:} \quad \mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n}(y_i - f_\theta(\mathbf{x}_i))^2$$

$$\text{Cross-Entropy:} \quad \mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \log \hat{y}_{ik}$$

The probabilistic justification for these losses is developed in [Week 07](../../03_probability/week07_likelihood/theory.md#3-likelihood-from-data-to-models).

### 4.4 Optimisation

Find $\theta^* = \arg\min_\theta \mathcal{L}(\theta)$. The dominant method is **gradient descent**:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta > 0$ is the **learning rate**. This is the subject of [Weeks 01](../../02_fundamentals/week01_optimization/theory.md#8-learning-rate-selection)–02.

### 4.5 Generalisation

The ultimate objective is to minimise the **expected risk** (error on the true distribution), not just the **empirical risk** (error on the training set). Formally:

$$R(\theta) = \mathbb{E}_{(\mathbf{x}, y) \sim p(\mathbf{x}, y)}\left[\ell(y, f_\theta(\mathbf{x}))\right]$$

Since $p(\mathbf{x}, y)$ is unknown, one approximates $R(\theta)$ using held-out data (validation and test sets). The gap between training and validation error reveals whether the model **overfits** (memorises training data) or **underfits** (fails to capture the underlying pattern).

> **Intuition.** Imagine memorising the answers to a practice exam word for word. You score perfectly on that specific exam (zero training error), but when the real exam presents different questions on the same topics, you fail (high test error). A model that generalises has learned the *topics*, not the specific questions. Regularisation ([Week 06](../../02_fundamentals/week06_regularization/theory.md#2-why-regularisation-the-overfitting-problem-revisited)) is the mathematical machinery for encouraging this behaviour.

> **The three error regimes:**
> | Regime | Training error | Validation error | Diagnosis | Remedy |
> |---|---|---|---|---|
> | Underfitting | High | High | Model too simple | Increase capacity, train longer |
> | Good fit | Low | Low | Appropriate complexity | Deploy and monitor |
> | Overfitting | Very low | High (or rising) | Model memorises noise | Regularise, add data, reduce capacity |

---

## 5. Unsupervised Learning

### 5.1 Clustering

Given $\{\mathbf{x}_i\}_{i=1}^{n}$, assign each point to one of $K$ clusters. K-Means minimises the within-cluster sum of squares:

$$\mathcal{J} = \sum_{k=1}^{K}\sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

where $\boldsymbol{\mu}_k$ is the centroid of cluster $C_k$. This is covered in [Week 05](../../02_fundamentals/week05_clustering/theory.md#3-k-means-clustering).

### 5.2 Dimensionality Reduction

Find a mapping $\mathbf{z}_i = g(\mathbf{x}_i) \in \mathbb{R}^k$ with $k \ll d$ that preserves important structure. **Principal Component Analysis (PCA)** finds the $k$ directions of maximum variance by eigendecomposition of the covariance matrix:

$$\Sigma = \frac{1}{n-1}(X - \bar{X})^\top(X - \bar{X})$$

This is the subject of [Week 04](../../02_fundamentals/week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view).

> **Intuition.** Imagine a photograph of $1000 \times 1000$ pixels — that is a point in $\mathbb{R}^{1{,}000{,}000}$. Yet natural images occupy a tiny manifold within that vast space (most random pixel arrangements are noise). PCA and autoencoders find coordinate systems aligned with this manifold, discarding the dimensions that carry only noise. The result: a compact representation that preserves the information that matters.

### 5.3 Density Estimation

Estimate the probability density function $p(\mathbf{x})$ from samples. Gaussian Mixture Models combine clustering and density estimation, and connect to the MLE framework developed in [Week 07](../../03_probability/week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle).

> **Concrete example.** Given height measurements from a population that includes adults and children, a single Gaussian is a poor fit (bimodal data). A mixture of two Gaussians captures the two subpopulations, simultaneously clustering the data and estimating the density.

### 5.4 Connections Between Paradigms

The boundaries between supervised and unsupervised learning are not rigid:

- **Semi-supervised learning**: a small number of labelled examples combined with a large unlabelled set. The unsupervised component discovers structure; the supervised component leverages labels.
- **Self-supervised learning**: the model creates its own labels from the data (e.g., predicting masked words in a sentence). This is the paradigm behind modern language models and is technically supervised, but requires no human annotation.
- **Representation learning**: unsupervised methods (autoencoders, contrastive learning) produce features that improve downstream supervised tasks.

These hybrid paradigms are increasingly dominant in practice, but the foundational tools remain the same: loss functions, gradients, and optimisation.

---

## 6. Reinforcement Learning

Although not treated in depth in this course, RL completes the map. The key mathematical objects are:

- **State space** $\mathcal{S}$, **action space** $\mathcal{A}$
- **Transition dynamics** $p(s_{t+1} \mid s_t, a_t)$
- **Reward function** $r(s_t, a_t)$
- **Policy** $\pi(a \mid s)$
- **Value function** $V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$

The Bellman equation provides a recursive relationship:

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \left[r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) V^\pi(s')\right]$$

Deep RL approximates $V^\pi$ or $\pi$ with neural networks — which requires all the optimisation and network-building skills from this course.

> **Legend for RL notation:**
> | Symbol | Name | Meaning |
> |---|---|---|
> | $s_t$ | State | The agent's observation of the environment at time $t$ |
> | $a_t$ | Action | The decision taken by the agent at time $t$ |
> | $r_t$ | Reward | Scalar feedback signal received after taking $a_t$ in $s_t$ |
> | $\gamma$ | Discount factor | How much the agent values future vs. immediate rewards ($0$: myopic; $\to 1$: far-sighted) |
> | $\pi(a \mid s)$ | Policy | Probability of choosing action $a$ in state $s$ |
> | $V^\pi(s)$ | Value function | Expected total discounted reward starting from state $s$ under policy $\pi$ |

> **Intuition.** RL is like training a dog: you cannot show it the "correct" action at each moment (no labels), but you can reward good behaviour and discourage bad behaviour. Over many trials, the dog learns a policy — a mapping from situations to actions — that maximises treats.

---

## 7. The Training Loop

Every supervised algorithm in this course follows the same iterative structure:

1. **Forward pass** — compute predictions: $\hat{\mathbf{y}} = f_\theta(X)$
2. **Loss computation** — evaluate the scalar loss: $\mathcal{L} = \mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$
3. **Backward pass** — compute the gradient: $\nabla_\theta \mathcal{L}$
4. **Parameter update** — adjust parameters: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$
5. **Repeat** until convergence

This loop is invariant to model complexity. Whether fitting a 2-parameter linear model or a 70-billion-parameter language model, the structure is identical; only the scale changes.

> **Notebook reference.** The starter notebooks from [Week 01](../../02_fundamentals/week01_optimization/theory.md#11-notebook-reference-guide) onward implement this exact loop. In `starter.ipynb` ([Week 01](../../02_fundamentals/week01_optimization/theory.md#11-notebook-reference-guide)), vanilla gradient descent is implemented in NumPy following this structure. Observe how the loop components map to specific lines of code.

> **Pseudocode for the training loop:**
> ```
> initialise θ randomly
> for epoch in 1, 2, ..., max_epochs:
>     ŷ = f_θ(X)                    # forward pass
>     L = loss(y, ŷ)                # loss computation
>     g = ∇_θ L                     # backward pass (gradient)
>     θ = θ - η * g                 # parameter update
>     if validate(θ) stops improving:
>         break                     # early stopping (regularisation)
> ```

> **Suggested experiment.** When you reach [Week 01](../../02_fundamentals/week01_optimization/theory.md#11-notebook-reference-guide)'s notebook, add `print(f"Epoch {epoch}: loss = {L:.4f}")` inside the training loop and plot the loss over epochs. A healthy training curve decreases rapidly at first (high-error parameters are easy to fix), then flattens (diminishing returns). If the curve oscillates, the learning rate $\eta$ is too large; if it barely moves, $\eta$ is too small.

---

## 8. What Is a Model?

A model is a **parameterised function**. The choice of function family determines what patterns the model can express — its **hypothesis space**.

**Linear model:**

$$f_\theta(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b, \quad \theta = (\mathbf{w}, b)$$

This can only represent linear relationships. If the true relationship is nonlinear, the model will underfit regardless of the optimisation procedure.

**Neural network (multi-layer perceptron):**

$$f_\theta(\mathbf{x}) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 \mathbf{x} + \mathbf{b}_1) \cdots) + \mathbf{b}_{L-1}) + \mathbf{b}_L$$

where $\sigma$ is a nonlinear activation function (e.g., ReLU: $\sigma(z) = \max(0, z)$). The composition of affine transformations with nonlinearities allows neural networks to approximate arbitrary continuous functions (the Universal Approximation Theorem).

> **Key insight.** A neural network is stacked linear models with nonlinearities in between. Understanding linear models deeply ([Week 03](../../02_fundamentals/week03_linear_models/theory.md#3-linear-regression)) is therefore a prerequisite for understanding deep networks.

> **Why nonlinearities matter.** Without $\sigma$, composing affine transformations yields another affine transformation: $W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2 \mathbf{b}_1 + \mathbf{b}_2) = W'\mathbf{x} + \mathbf{b}'$. The entire network collapses to a single linear layer — depth adds no representational power. The activation function $\sigma$ breaks this linearity, and each layer can carve the input space into increasingly complex regions. This is proven in [Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#74-he-kaiming-initialisation) from scratch.

---

## 9. Hypothesis Spaces and Inductive Bias

### 9.1 The Hypothesis Space

The choice of model family defines a **hypothesis space** $\mathcal{H}$ — the set of all functions $f_\theta$ that the model can represent as $\theta$ ranges over the parameter space $\Theta$. Learning is a search over $\mathcal{H}$ for the function that best fits the data according to the loss.

**Example.** For a linear model $f_\theta(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ with $\mathbf{x} \in \mathbb{R}^2$, the hypothesis space is the set of all planes in $\mathbb{R}^3$ (the input–output space). No matter how much data is provided or how long optimisation runs, the model cannot represent a curved surface.

For a neural network with ReLU activations and sufficient width, the hypothesis space is dramatically larger — in principle, any continuous function on a compact domain can be approximated to arbitrary precision (the **Universal Approximation Theorem**, Cybenko 1989; Hornik et al. 1989).

### 9.2 Inductive Bias

An **inductive bias** is any assumption built into the learning algorithm that constrains or prioritises certain hypotheses over others. Without inductive bias, learning from finite data is impossible — this is the mathematical content of the **No Free Lunch Theorem** (Wolpert, 1996), which states that no learning algorithm is universally superior; every algorithm that performs well on some class of problems must perform poorly on others.

Examples of inductive bias:

| Model / Technique | Inductive Bias                                                                  |
| ----------------- | ------------------------------------------------------------------------------- |
| Linear regression | The true function is linear in inputs                                           |
| CNN               | Spatially local patterns are important; the function is translation-equivariant |
| RNN               | Sequential order matters; recent context is most relevant                       |
| Transformer       | All pairwise position interactions may be relevant (no locality assumption)     |
| L2 regularisation | Smaller weights are preferable (smoother functions)                             |
| Data augmentation | The function should be invariant to certain transformations                     |

The art of machine learning is selecting inductive biases that match the structure of the problem. A convolutional network applied to tabular data encodes an inappropriate bias (spatial locality in a feature table is meaningless), and a linear model applied to image pixels encodes too weak a bias (it cannot capture edges, textures, or shapes).

### 9.3 Capacity, Underfitting, and Overfitting

The **capacity** of a model is the richness of its hypothesis space — informally, how complex a function it can represent. There is a fundamental tension:

- **Too little capacity** → underfitting. The hypothesis space does not contain a good approximation of the true function.
- **Too much capacity** → overfitting. The hypothesis space is so large that the model can memorise noise in the training data, and it selects a function that fits training points perfectly but generalises poorly.

This tension is formalised in the **bias-variance decomposition** (developed in [Week 06](../../02_fundamentals/week06_regularization/theory.md#21-biasvariance-recap)). For now, the intuition suffices: a good model is one whose hypothesis space is rich enough to contain the truth, but constrained enough that finite data can identify it. Regularisation (Section 10.4) is the primary tool for managing this trade-off.

> **Suggested experiment.** In [Week 03](../../02_fundamentals/week03_linear_models/theory.md#5-polynomial-regression-and-feature-expansion)'s notebook, fit polynomial functions of degree 1, 3, 10, and 20 to a small dataset and observe how training error decreases monotonically while validation error first decreases and then increases. This U-shaped validation curve is the empirical signature of the bias-variance trade-off.

> **Analogy: the Goldilocks principle.** A model with too few parameters is like trying to describe a city with a single sentence — the description is not *wrong*, but it omits everything interesting (underfitting). A model with too many parameters is like memorising the phone book — it captures every detail of the training set, including typos, and fails on any new entry (overfitting). The goal is a model that captures the *generating rules* behind the data, not the data itself.

---

## 10. The Four Practitioner Choices

Every ML project requires four fundamental decisions:

### 10.1 Architecture

The choice of function family: linear model, decision tree, convolutional neural network, transformer, etc. This choice encodes **inductive biases** — assumptions about data structure.

- **CNN:** assumes spatial locality (nearby pixels are related) → suited for images.
- **Transformer:** assumes pairwise dependencies between positions → suited for sequences.
- **Linear model:** assumes a linear relationship → suited when features are well-engineered.

### 10.2 Loss Function

What "wrong" means. The loss must be differentiable (for gradient-based optimisation) and aligned with the task:

| Task                  | Loss                      | Probabilistic Interpretation |
| --------------------- | ------------------------- | ---------------------------- |
| Regression            | MSE                       | Gaussian noise model         |
| Regression (robust)   | MAE                       | Laplace noise model          |
| Binary classification | Binary cross-entropy      | Bernoulli likelihood         |
| Multi-class           | Categorical cross-entropy | Categorical likelihood       |

### 10.3 Optimiser

How to adjust parameters. Gradient descent is the foundation; Adam is the default in practice. The choice of optimiser and its hyperparameters (learning rate, momentum) affect convergence speed and final quality.

### 10.4 Regularisation

How to prevent over-memorising training data. Techniques include:

- **L2 penalty (Ridge):** $\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\mathbf{w}\|^2$
- **L1 penalty (Lasso):** $\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\mathbf{w}\|_1$
- **Dropout:** randomly zero activations during training
- **Early stopping:** halt training when validation error increases
- **Data augmentation:** artificially expand the training set

These techniques form the subject of [Weeks 06](../../02_fundamentals/week06_regularization/theory.md) and 16.

---

## 11. The Evaluation Mindset

For any trained model, three questions must always be answered:

1. **Does it fit the training data well?** If not → **underfitting**. The model is too simple or the optimisation has not converged. Remedies: increase model capacity, train longer, reduce regularisation.

2. **Does it generalise to unseen data?** If not → **overfitting**. The model has memorised noise in the training data. Remedies: add regularisation, collect more data, reduce model capacity.

3. **Is the metric aligned with the actual problem?** A model with 99% accuracy on an imbalanced dataset (1% positive class) may predict the majority class exclusively and be entirely useless.

### Data Splits

To operationalise evaluation, the data is divided:

| Split                        | Purpose                         | Used to                                                     |
| ---------------------------- | ------------------------------- | ----------------------------------------------------------- |
| **Training set** (~60–80%)   | Learn parameters $\theta$       | Compute gradient updates                                    |
| **Validation set** (~10–20%) | Tune hyperparameters            | Select learning rate, regularisation strength, architecture |
| **Test set** (~10–20%)       | Estimate real-world performance | Report final metrics (used **once**)                        |

> **Important.** The test set must never influence any modelling decision. Doing so invalidates the performance estimate.

### Key Vocabulary

| Term                   | Definition                                                             |
| ---------------------- | ---------------------------------------------------------------------- |
| **Feature**            | A measurable property of an input example                              |
| **Label / Target**     | The value to be predicted                                              |
| **Parameter / Weight** | A number learned from data                                             |
| **Hyperparameter**     | A setting chosen by the practitioner (learning rate, number of layers) |
| **Loss / Cost**        | Scalar measuring prediction error                                      |
| **Gradient**           | Vector of partial derivatives of loss w.r.t. parameters                |
| **Epoch**              | One full pass through the training dataset                             |
| **Batch**              | A subset of training data used for one gradient update                 |
| **Inference**          | Using a trained model to make predictions on new data                  |

---

## 12. References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 1. Springer.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 1. MIT Press. Available at [deeplearningbook.org](https://www.deeplearningbook.org).
3. Burkov, A. (2019). *The Hundred-Page Machine Learning Book*, Chapters 1–3.
4. 3Blue1Brown. "Neural Networks" series. YouTube. For visual intuition on forward passes and gradient descent.
5. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
6. Cybenko, G. (1989). "Approximation by Superpositions of a Sigmoidal Function." *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
7. Hornik, K., Stinchcombe, M., & White, H. (1989). "Multilayer Feedforward Networks Are Universal Approximators." *Neural Networks*, 2(5), 359–366.
8. Wolpert, D. H. (1996). "The Lack of A Priori Distinctions Between Learning Algorithms." *Neural Computation*, 8(7), 1341–1390.
