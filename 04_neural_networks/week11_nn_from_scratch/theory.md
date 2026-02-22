# Neural Networks From Scratch

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [From Linear Models to Neural Networks](#2-from-linear-models-to-neural-networks)
    - 2.1 [The Limitation of Linearity](#21-the-limitation-of-linearity)
    - 2.2 [The Universal Approximation Idea](#22-the-universal-approximation-idea)
3. [The Neuron](#3-the-neuron)
    - 3.1 [Mathematical Definition](#31-mathematical-definition)
    - 3.2 [Activation Functions](#32-activation-functions)
    - 3.3 [Choosing an Activation](#33-choosing-an-activation)
4. [Fully Connected Networks (MLPs)](#4-fully-connected-networks-mlps)
    - 4.1 [Architecture](#41-architecture)
    - 4.2 [Forward Pass in Matrix Form](#42-forward-pass-in-matrix-form)
    - 4.3 [Output Layer and Loss Functions](#43-output-layer-and-loss-functions)
5. [Backpropagation](#5-backpropagation)
    - 5.1 [The Chain Rule — Single Variable](#51-the-chain-rule--single-variable)
    - 5.2 [The Chain Rule — Vectors and Matrices](#52-the-chain-rule--vectors-and-matrices)
    - 5.3 [Backprop Through a Two-Layer Network](#53-backprop-through-a-two-layer-network)
    - 5.4 [General Backpropagation Algorithm](#54-general-backpropagation-algorithm)
    - 5.5 [Computational Graph Perspective](#55-computational-graph-perspective)
6. [Gradient Checking](#6-gradient-checking)
7. [Weight Initialisation](#7-weight-initialisation)
    - 7.1 [Why Initialisation Matters](#71-why-initialisation-matters)
    - 7.2 [Small Random Initialisation](#72-small-random-initialisation)
    - 7.3 [Xavier / Glorot Initialisation](#73-xavier--glorot-initialisation)
    - 7.4 [He (Kaiming) Initialisation](#74-he-kaiming-initialisation)
    - 7.5 [Summary of Initialisation Rules](#75-summary-of-initialisation-rules)
8. [The Training Loop](#8-the-training-loop)
    - 8.1 [Full-Batch vs. Mini-Batch](#81-full-batch-vs-mini-batch)
    - 8.2 [Hyperparameters for a First Network](#82-hyperparameters-for-a-first-network)
9. [Regularisation in Neural Networks](#9-regularisation-in-neural-networks)
10. [Connections to the Rest of the Course](#10-connections-to-the-rest-of-the-course)
11. [Notebook Reference Guide](#11-notebook-reference-guide)
12. [Symbol Reference](#12-symbol-reference)
13. [References](#13-references)

---

## 1. Scope and Purpose

[[Weeks 01](../../02_fundamentals/week01_optimization/theory.md)](../../02_fundamentals/week01_optimization/theory.md)–[10](../../03_probability/week10_surrogate_models/theory.md) built models from the probabilistic toolkit: linear regression, regularisation, MLE, Bayesian inference, GPs. All of these are powerful but fundamentally limited in the functions they can represent (linear in features, or requiring hand-crafted kernels).

This week crosses the threshold into **neural networks** — models that learn their own features. We implement every component from scratch in NumPy:

1. **Forward pass** — compute the output of the network given inputs.
2. **Loss** — measure how far off the prediction is.
3. **Backward pass (backpropagation)** — compute the gradient of the loss with respect to every parameter.
4. **Parameter update** — gradient descent step.

After this week, when you use `loss.backward()` in PyTorch ([Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md)), you will know exactly what it computes.

**Prerequisites.** [Week 00b](../../01_intro/week00b_math_and_data/theory.md) (chain rule), [[Weeks 01](../../02_fundamentals/week01_optimization/theory.md)](../../02_fundamentals/week01_optimization/theory.md)–[02](../../02_fundamentals/week02_advanced_optimizers/theory.md) (gradient descent, optimisers), [Week 03](../../02_fundamentals/week03_linear_models/theory.md) (linear regression as a single neuron).

---

## 2. From Linear Models to Neural Networks

### 2.1 The Limitation of Linearity

A linear model computes:

$$\hat{y} = \mathbf{w}^\top\mathbf{x} + b$$

No matter how many linear layers you stack:

$$\mathbf{z}_1 = W_1\mathbf{x} + \mathbf{b}_1, \quad \mathbf{z}_2 = W_2\mathbf{z}_1 + \mathbf{b}_2$$

the result collapses to a single linear transform:

$$\mathbf{z}_2 = W_2 W_1\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2) = W'\mathbf{x} + \mathbf{b}'$$

**Stacking linear layers without nonlinearities gains nothing.** The model can only represent linear decision boundaries.

---

### 2.2 The Universal Approximation Idea

Insert a **nonlinear activation function** $\sigma(\cdot)$ between layers:

$$\mathbf{z}_1 = W_1\mathbf{x} + \mathbf{b}_1, \quad \mathbf{a}_1 = \sigma(\mathbf{z}_1), \quad \hat{y} = W_2\mathbf{a}_1 + \mathbf{b}_2$$

**Universal Approximation Theorem** (Cybenko 1989, Hornik 1991): a single hidden layer with a sufficient number of neurons and a non-polynomial activation can approximate any continuous function on a compact set to arbitrary accuracy.

> **Caveat.** The theorem guarantees _existence_ but says nothing about how to _find_ the approximation (optimisation) or how many neurons are needed (could be exponentially many). In practice, deeper networks with fewer neurons per layer are far more efficient than shallow wide ones.

---

## 3. The Neuron

### 3.1 Mathematical Definition

A single neuron computes:

$$\boxed{a = \sigma\!\left(\mathbf{w}^\top\mathbf{x} + b\right) = \sigma(z)}$$

where:
- $\mathbf{x} \in \mathbb{R}^{n_{\text{in}}}$ — input vector.
- $\mathbf{w} \in \mathbb{R}^{n_{\text{in}}}$ — weight vector (learnable).
- $b \in \mathbb{R}$ — bias (learnable).
- $z = \mathbf{w}^\top\mathbf{x} + b$ — pre-activation (linear combination).
- $\sigma(\cdot)$ — activation function (nonlinearity).
- $a$ — activation (output of the neuron).

This is exactly logistic regression ([Week 03](../../02_fundamentals/week03_linear_models/theory.md)) when $\sigma = \text{sigmoid}$ and the loss is binary cross-entropy — so a **logistic regression unit is a single neuron**.

---

### 3.2 Activation Functions

| Name           | $\sigma(z)$                                              | $\sigma'(z)$                                           | Range               | Properties                                                      |
| -------------- | -------------------------------------------------------- | ------------------------------------------------------ | ------------------- | --------------------------------------------------------------- |
| **Sigmoid**    | $\frac{1}{1+e^{-z}}$                                     | $\sigma(z)(1 - \sigma(z))$                             | $(0, 1)$            | Squashes to probability; **saturates** for $                    | z | \gg 0$ |
| **Tanh**       | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$                      | $1 - \tanh^2(z)$                                       | $(-1, 1)$           | Zero-centred; still saturates                                   |
| **ReLU**       | $\max(0, z)$                                             | $\begin{cases}1 & z > 0\\0 & z \leq 0\end{cases}$      | $[0, \infty)$       | Sparse; no saturation for $z > 0$; **dead neurons** for $z < 0$ |
| **Leaky ReLU** | $\begin{cases}z & z > 0\\\alpha z & z \leq 0\end{cases}$ | $\begin{cases}1 & z > 0\\\alpha & z \leq 0\end{cases}$ | $(-\infty, \infty)$ | Fixes dead neuron problem; $\alpha \approx 0.01$–$0.1$          |
| **Softmax**    | $\frac{e^{z_k}}{\sum_j e^{z_j}}$                         | (Jacobian matrix)                                      | $(0, 1)$; sums to 1 | Multi-class output layer                                        |

**Implementations from the notebook:**

```python
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

> **The `max` trick in softmax.** Subtracting $\max_j z_j$ before exponentiating prevents overflow ($e^{1000} = \infty$) without changing the result (the subtraction cancels in the ratio).

---

### 3.3 Choosing an Activation

| Guideline                                     | Recommendation                         |
| --------------------------------------------- | -------------------------------------- |
| Hidden layers (default)                       | **ReLU** — fast, sparse, no saturation |
| Hidden layers (if dead neurons are a problem) | Leaky ReLU or ELU                      |
| Binary classification output                  | Sigmoid                                |
| Multi-class classification output             | Softmax                                |
| Regression output                             | No activation (identity)               |
| Recurrent networks (LSTM gates)               | Sigmoid (gates) + tanh (cell state)    |

---

## 4. Fully Connected Networks (MLPs)

### 4.1 Architecture

A **multi-layer perceptron (MLP)** or fully connected network with $L$ layers:

$$\text{Input} \xrightarrow{W_1, b_1} \sigma_1 \xrightarrow{W_2, b_2} \sigma_2 \xrightarrow{} \cdots \xrightarrow{W_L, b_L} \text{Output}$$

**Notation for layer $l$:**

| Symbol             | Shape                      | Description                           |
| ------------------ | -------------------------- | ------------------------------------- |
| $W^{[l]}$          | $n^{[l-1]} \times n^{[l]}$ | Weight matrix                         |
| $\mathbf{b}^{[l]}$ | $1 \times n^{[l]}$         | Bias vector                           |
| $Z^{[l]}$          | $m \times n^{[l]}$         | Pre-activations (before nonlinearity) |
| $A^{[l]}$          | $m \times n^{[l]}$         | Activations (after nonlinearity)      |
| $n^{[l]}$          | scalar                     | Number of neurons in layer $l$        |
| $m$                | scalar                     | Batch size (number of samples)        |

Convention: $A^{[0]} = X$ (the input is "layer 0").

---

### 4.2 Forward Pass in Matrix Form

For a two-layer network (one hidden layer + output):

$$Z^{[1]} = X W^{[1]} + \mathbf{b}^{[1]} \qquad A^{[1]} = \text{ReLU}(Z^{[1]})$$

$$Z^{[2]} = A^{[1]} W^{[2]} + \mathbf{b}^{[2]} \qquad A^{[2]} = \text{softmax}(Z^{[2]})$$

In NumPy:

```python
z1 = X @ W1 + b1       # (m, n_hidden)
a1 = np.maximum(0, z1) # ReLU
z2 = a1 @ W2 + b2      # (m, n_out)
a2 = softmax(z2)        # (m, n_out)
```

Each operation is a matrix multiply followed by a pointwise nonlinearity. The entire forward pass for $m$ samples is batched — no loops over individual samples.

> **Notebook reference.** Cell 4 implements `TwoLayerNet` with `forward()` performing this exact computation on `make_moons` data.

---

### 4.3 Output Layer and Loss Functions

| Task                           | Output activation | Loss function                  | Formula                                                                                 |
| ------------------------------ | ----------------- | ------------------------------ | --------------------------------------------------------------------------------------- |
| **Binary classification**      | Sigmoid           | Binary cross-entropy (BCE)     | $\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ |
| **Multi-class classification** | Softmax           | Categorical cross-entropy (CE) | $\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\log\hat{y}_{i, c_i}$                          |
| **Regression**                 | Identity          | MSE                            | $\mathcal{L} = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2$                            |

where $c_i$ is the true class index for sample $i$ and $\hat{y}_{i, c_i}$ is the predicted probability for the correct class.

**Connection to likelihood ([Week 07](../../03_probability/week07_likelihood/theory.md)):** cross-entropy loss = negative log-likelihood of the Bernoulli (BCE) or Categorical (CE) distribution. Minimising CE = maximising likelihood.

---

## 5. Backpropagation

### 5.1 The Chain Rule — Single Variable

If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{df}{dg}\cdot\frac{dg}{dx}$$

Applied repeatedly through a chain of $L$ composed functions:

$$\frac{d\mathcal{L}}{dx} = \frac{d\mathcal{L}}{df_L}\cdot\frac{df_L}{df_{L-1}}\cdots\frac{df_2}{df_1}\cdot\frac{df_1}{dx}$$

This is all that backpropagation is: **the chain rule applied systematically from the loss back through the network**.

---

### 5.2 The Chain Rule — Vectors and Matrices

For vector-valued functions, the chain rule involves **Jacobian matrices**:

If $\mathbf{z} = g(\mathbf{x})$ and $\mathcal{L} = f(\mathbf{z})$, then:

$$\frac{\partial\mathcal{L}}{\partial\mathbf{x}} = \frac{\partial\mathcal{L}}{\partial\mathbf{z}}\cdot\frac{\partial\mathbf{z}}{\partial\mathbf{x}}$$

where $\frac{\partial\mathbf{z}}{\partial\mathbf{x}}$ is the Jacobian (matrix of partial derivatives).

**Key matrix calculus identities used in backprop:**

| Forward                       | Backward ($\frac{\partial\mathcal{L}}{\partial\text{input}}$)                                                                                                                                                                                                                           |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $Z = XW + \mathbf{b}$         | $\frac{\partial\mathcal{L}}{\partial X} = \frac{\partial\mathcal{L}}{\partial Z}W^\top$, $\;\frac{\partial\mathcal{L}}{\partial W} = X^\top\frac{\partial\mathcal{L}}{\partial Z}$, $\;\frac{\partial\mathcal{L}}{\partial\mathbf{b}} = \sum_i\frac{\partial\mathcal{L}}{\partial Z_i}$ |
| $A = \sigma(Z)$ (elementwise) | $\frac{\partial\mathcal{L}}{\partial Z} = \frac{\partial\mathcal{L}}{\partial A}\odot\sigma'(Z)$                                                                                                                                                                                        |

where $\odot$ denotes elementwise (Hadamard) product.

> **Key insight.** In the backward pass, the weight matrix appears **transposed**: forward uses $W$, backward uses $W^\top$. This makes intuitive sense — the gradient flows backward through the same connections, just in reverse.

---

### 5.3 Backprop Through a Two-Layer Network

Consider the network: $X \to Z^{[1]} \to A^{[1]} \to Z^{[2]} \to A^{[2]} \to \mathcal{L}$.

**Step 1. Output gradient (softmax + cross-entropy):**

The combined gradient of softmax + cross-entropy has a remarkably clean form:

$$\frac{\partial\mathcal{L}}{\partial Z^{[2]}} = A^{[2]} - Y$$

where $Y$ is the one-hot encoded label matrix. This is the same as the logistic regression gradient ([Week 03](../../02_fundamentals/week03_linear_models/theory.md)) — no coincidence; the derivation is identical.

**Derivation.** For sample $i$ with true class $c_i$:

$$\mathcal{L}_i = -\log\hat{y}_{i,c_i} = -\log\frac{e^{z_{c_i}}}{\sum_k e^{z_k}} = -z_{c_i} + \log\sum_k e^{z_k}$$

$$\frac{\partial\mathcal{L}_i}{\partial z_j} = -\mathbb{1}[j = c_i] + \frac{e^{z_j}}{\sum_k e^{z_k}} = \hat{y}_{i,j} - y_{i,j}$$

**Step 2. Gradients for $W^{[2]}$ and $\mathbf{b}^{[2]}$:**

$$\frac{\partial\mathcal{L}}{\partial W^{[2]}} = \frac{1}{m}(A^{[1]})^\top\frac{\partial\mathcal{L}}{\partial Z^{[2]}}$$

$$\frac{\partial\mathcal{L}}{\partial\mathbf{b}^{[2]}} = \frac{1}{m}\sum_{i=1}^{m}\frac{\partial\mathcal{L}}{\partial Z^{[2]}_i}$$

**Step 3. Propagate gradient to hidden layer:**

$$\frac{\partial\mathcal{L}}{\partial A^{[1]}} = \frac{\partial\mathcal{L}}{\partial Z^{[2]}}(W^{[2]})^\top$$

$$\frac{\partial\mathcal{L}}{\partial Z^{[1]}} = \frac{\partial\mathcal{L}}{\partial A^{[1]}}\odot\text{ReLU}'(Z^{[1]})$$

The ReLU derivative $\text{ReLU}'(z) = \mathbb{1}[z > 0]$ acts as a **gate**: gradients pass through where neurons are active and are blocked where neurons are inactive (the gradient is zero).

**Step 4. Gradients for $W^{[1]}$ and $\mathbf{b}^{[1]}$:**

$$\frac{\partial\mathcal{L}}{\partial W^{[1]}} = \frac{1}{m}X^\top\frac{\partial\mathcal{L}}{\partial Z^{[1]}}$$

$$\frac{\partial\mathcal{L}}{\partial\mathbf{b}^{[1]}} = \frac{1}{m}\sum_{i=1}^{m}\frac{\partial\mathcal{L}}{\partial Z^{[1]}_i}$$

> **Notebook reference.** Cell 4 implements `backward()` with exactly these equations. The cached values (`X, z1, a1, z2, a2`) from the forward pass are reused.

---

### 5.4 General Backpropagation Algorithm

For an $L$-layer network:

**Forward pass** (compute and cache):
$$\text{For } l = 1, \ldots, L: \quad Z^{[l]} = A^{[l-1]}W^{[l]} + \mathbf{b}^{[l]}, \quad A^{[l]} = \sigma_l(Z^{[l]})$$

**Backward pass** (propagate gradients):
$$dZ^{[L]} = A^{[L]} - Y \quad \text{(softmax + CE)}$$

$$\text{For } l = L, L{-}1, \ldots, 1:$$

$$dW^{[l]} = \frac{1}{m}(A^{[l-1]})^\top\,dZ^{[l]}$$

$$d\mathbf{b}^{[l]} = \frac{1}{m}\mathbf{1}^\top\,dZ^{[l]}$$

$$dA^{[l-1]} = dZ^{[l]}(W^{[l]})^\top$$

$$dZ^{[l-1]} = dA^{[l-1]}\odot\sigma'_{l-1}(Z^{[l-1]})$$

**Parameter update:**
$$W^{[l]} \leftarrow W^{[l]} - \eta\,dW^{[l]}, \quad \mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta\,d\mathbf{b}^{[l]}$$

The algorithm is $O(n)$ in the number of parameters — the same cost as computing the forward pass. This is why neural networks are trainable despite having millions of parameters.

---

### 5.5 Computational Graph Perspective

Every neural network can be drawn as a **directed acyclic graph (DAG)** where:
- **Nodes** are operations (multiply, add, ReLU, softmax, loss).
- **Edges** carry tensors (data flows forward, gradients flow backward).

```
X ──► [@ W1+b1] ──► Z1 ──► [ReLU] ──► A1 ──► [@ W2+b2] ──► Z2 ──► [softmax] ──► Ŷ ──► [CE] ──► L
           ▲                                         ▲                                       |
          W1,b1                                    W2,b2                                     Y
```

The backward pass traverses this graph in **reverse topological order**, applying the chain rule at each node. This is exactly what PyTorch's `autograd` does automatically ([Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md)).

> **Why cache the forward pass?** Each backward step needs values from the forward pass (e.g., $A^{[1]}$ is needed to compute $dW^{[2]}$, $Z^{[1]}$ is needed for $\text{ReLU}'$). Storing these avoids recomputation at the cost of memory.

---

## 6. Gradient Checking

Analytic gradients from backprop can have subtle bugs (wrong shapes, missing transposes, sign errors). **Gradient checking** validates them against numerical gradients.

**Numerical gradient** via centred finite differences:

$$\frac{\partial\mathcal{L}}{\partial\theta_j} \approx \frac{\mathcal{L}(\theta_j + \epsilon) - \mathcal{L}(\theta_j - \epsilon)}{2\epsilon}$$

with $\epsilon \approx 10^{-5}$.

**Comparison metric (relative error):**

$$\text{rel\_error} = \frac{\|\mathbf{g}_{\text{analytic}} - \mathbf{g}_{\text{numerical}}\|}{\|\mathbf{g}_{\text{analytic}}\| + \|\mathbf{g}_{\text{numerical}}\| + \delta}$$

| Relative error         | Verdict                                                 |
| ---------------------- | ------------------------------------------------------- |
| $< 10^{-7}$            | Excellent — implementation is almost certainly correct  |
| $< 10^{-5}$            | Good — acceptable for most purposes                     |
| $10^{-5}$ to $10^{-3}$ | Suspicious — check for edge cases (e.g., kinks in ReLU) |
| $> 10^{-3}$            | Bug — backprop implementation is wrong                  |

**Important notes:**
- Use a **small batch** (e.g., 10 samples) — gradient checking is $O(\text{params})$ forward passes, very slow on large data.
- **Turn off regularisation** during checking (or include it in the numerical loss too).
- **Check each parameter group** separately ($W_1, b_1, W_2, b_2$).
- ReLU has a non-differentiable point at $z = 0$; this can cause slightly higher relative errors at isolated points.

> **Notebook reference.** Cell 7 implements `gradient_check` with centred finite differences and reports the relative error per parameter group.

---

## 7. Weight Initialisation

### 7.1 Why Initialisation Matters

The loss landscape of a neural network is non-convex. The initial point determines which local minimum (or plateau) gradient descent converges to. Poor initialisation causes:

| Problem                   | Symptom                                       | Cause                                            |
| ------------------------- | --------------------------------------------- | ------------------------------------------------ |
| **Vanishing activations** | All hidden activations collapse to zero       | Weights too small                                |
| **Exploding activations** | Activations overflow to NaN                   | Weights too large                                |
| **Vanishing gradients**   | Gradients shrink exponentially through layers | Saturated activations (sigmoid/tanh with large $ | z | $) |
| **Symmetry**              | All neurons learn the same thing              | All weights identical (never use $W = 0$!)       |

**The goal of initialisation:** keep the variance of activations and gradients approximately constant across layers.

---

### 7.2 Small Random Initialisation

$$W^{[l]}_{ij} \sim \mathcal{N}(0, \sigma^2_{\text{init}})$$

with $\sigma_{\text{init}} = 0.01$. This works for shallow networks but fails for deep ones: if $\sigma$ is too small, activations shrink layer by layer; if too large, they explode.

---

### 7.3 Xavier / Glorot Initialisation

**Glorot & Bengio (2010).** For layers with symmetric activations (tanh, sigmoid):

$$\boxed{W^{[l]}_{ij} \sim \mathcal{N}\!\left(0, \frac{2}{n^{[l-1]} + n^{[l]}}\right)}$$

or the uniform variant:

$$W^{[l]}_{ij} \sim \text{Uniform}\!\left(-\sqrt{\frac{6}{n^{[l-1]} + n^{[l]}}}, \sqrt{\frac{6}{n^{[l-1]} + n^{[l]}}}\right)$$

**Derivation sketch.** Suppose $z = \sum_{j=1}^{n_{\text{in}}} w_j x_j$ where $w_j$ and $x_j$ are independent, zero-mean. Then:

$$\text{Var}(z) = n_{\text{in}}\cdot\text{Var}(w)\cdot\text{Var}(x)$$

To preserve variance ($\text{Var}(z) = \text{Var}(x)$), we need $\text{Var}(w) = 1/n_{\text{in}}$. The same argument applied backward gives $\text{Var}(w) = 1/n_{\text{out}}$. Xavier compromises with $\text{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$.

---

### 7.4 He (Kaiming) Initialisation

**He et al. (2015).** For layers with **ReLU** activations:

$$\boxed{W^{[l]}_{ij} \sim \mathcal{N}\!\left(0, \frac{2}{n^{[l-1]}}\right)}$$

The factor of 2 accounts for the fact that ReLU zeros out half the activations (on average), halving the variance.

**Derivation.** With ReLU, $\text{Var}(a) = \frac{1}{2}\text{Var}(z)$ (half the pre-activations are zeroed). To compensate: $\text{Var}(w) = 2/n_{\text{in}}$.

---

### 7.5 Summary of Initialisation Rules

| Activation            | Initialisation | $\text{Var}(W)$                            | Rule name |
| --------------------- | -------------- | ------------------------------------------ | --------- |
| **Sigmoid / Tanh**    | Xavier         | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | Glorot    |
| **ReLU / Leaky ReLU** | He             | $\frac{2}{n_{\text{in}}}$                  | Kaiming   |
| **SELU**              | LeCun          | $\frac{1}{n_{\text{in}}}$                  | LeCun     |

**Biases** are always initialised to **zero**: $\mathbf{b}^{[l]} = \mathbf{0}$.

> **Notebook reference.** Cell 9 compares three initialisations on `make_moons`: small random ($\sigma = 0.01$), Xavier ($\sigma = \sqrt{2/(n_{\text{in}} + n_{\text{out}})}$), and He ($\sigma = \sqrt{2/n_{\text{in}}}$). The loss curves show that small random converges slowly (or gets stuck), while Xavier and He converge quickly.

---

## 8. The Training Loop

### 8.1 Full-Batch vs. Mini-Batch

The notebook's `train_step` performs **full-batch gradient descent** (all $m$ samples at once):

```python
def train_step(self, X, y, lr=0.01):
    y_pred = self.forward(X)
    loss = self.compute_loss(y_pred, y)
    grads = self.backward(y)
    self.W1 -= lr * grads['dW1']
    self.b1 -= lr * grads['db1']
    self.W2 -= lr * grads['dW2']
    self.b2 -= lr * grads['db2']
    return loss
```

For the `make_moons` dataset ($m = 210$ training samples), this is fine. For larger datasets, use **mini-batch SGD** ([[Weeks 01](../../02_fundamentals/week01_optimization/theory.md)](../../02_fundamentals/week01_optimization/theory.md)–[02](../../02_fundamentals/week02_advanced_optimizers/theory.md)):

1. Shuffle the training data.
2. Split into batches of size $B$ (32, 64, 128, or 256).
3. Run `train_step` on each batch.
4. One pass through all batches = one **epoch**.

Mini-batch SGD adds noise to the gradient, which:
- Acts as implicit regularisation ([Week 06](../../02_fundamentals/week06_regularization/theory.md)).
- Helps escape sharp local minima.
- Enables training on datasets that don't fit in memory.

---

### 8.2 Hyperparameters for a First Network

| Hyperparameter   | Typical starting value | Notes                                             |
| ---------------- | ---------------------- | ------------------------------------------------- |
| Hidden size      | 10–128                 | Start small; increase if underfitting             |
| Number of layers | 1–3 hidden             | More layers = more expressive but harder to train |
| Learning rate    | 0.01–1.0               | The notebook uses 0.5 for the small dataset       |
| Epochs           | 500–2000               | Until loss plateaus                               |
| Activation       | ReLU                   | Default for hidden layers                         |
| Initialisation   | He (for ReLU)          | Xavier for tanh/sigmoid                           |

> **Notebook reference.** Cell 6 trains `TwoLayerNet(input_size=2, hidden_size=10, output_size=2)` on `make_moons` for 1000 epochs with $\eta = 0.5$, achieving ~98% test accuracy.

---

## 9. Regularisation in Neural Networks

Neural networks are powerful enough to memorise the training set. Regularisation is essential.

**L2 regularisation (weight decay):**

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda\sum_l\|W^{[l]}\|_F^2$$

The gradient modification is:

$$\frac{\partial\mathcal{L}_{\text{reg}}}{\partial W^{[l]}} = \frac{\partial\mathcal{L}}{\partial W^{[l]}} + 2\lambda W^{[l]}$$

**Connection to Bayesian inference ([Week 08](../../03_probability/week08_uncertainty/theory.md)):** L2 regularisation = Gaussian prior on weights with precision $\alpha = 2\lambda/\sigma_n^2$. The regularised loss = negative log-posterior (MAP estimate).

**Other forms** (covered in later weeks):
- **Dropout** ([Week 16](../../05_deep_learning/week16_regularization_dl/theory.md)) — randomly zero out neurons during training.
- **Early stopping** ([Week 06](../../02_fundamentals/week06_regularization/theory.md)) — stop training when validation loss increases.
- **Batch normalisation** ([Week 12](../week12_training_pathologies/theory.md)) — normalises activations; has implicit regularising effect.
- **Data augmentation** ([Week 15](../../05_deep_learning/week15_cnn_representations/theory.md)) — artificially increase training set.

> **Notebook reference.** Exercise 4 asks you to add L2 regularisation and sweep $\lambda \in \{0, 0.001, 0.01, 0.1\}$.

---

## 10. Connections to the Rest of the Course

| Week                               | Connection                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **[[Week 01](../../02_fundamentals/week01_optimization/theory.md)](../../02_fundamentals/week01_optimization/theory.md)–[02](../../02_fundamentals/week02_advanced_optimizers/theory.md) (Optimisation)**      | Gradient descent and advanced optimisers (Adam, momentum) are used unchanged; only the gradient computation is new |
| **[Week 03](../../02_fundamentals/week03_linear_models/theory.md) (Linear Models)**        | A single neuron with sigmoid = logistic regression; a single neuron with identity = linear regression              |
| **[Week 06](../../02_fundamentals/week06_regularization/theory.md) (Regularisation)**       | L2 regularisation = weight decay; early stopping applies directly                                                  |
| **[Week 07](../../03_probability/week07_likelihood/theory.md) (Likelihood)**           | Cross-entropy loss = NLL; MSE loss = Gaussian NLL                                                                  |
| **[Week 08](../../03_probability/week08_uncertainty/theory.md) (Uncertainty)**          | Weight decay = Gaussian prior (MAP); Bayesian NNs place a full posterior on $W$                                    |
| **[Week 10](../../03_probability/week10_surrogate_models/theory.md) (GPs)**                  | Infinite-width single-hidden-layer network → GP (Neal, 1996)                                                       |
| **[Week 12](../week12_training_pathologies/theory.md) (Training Pathologies)** | Vanishing/exploding gradients; batch norm; residual connections                                                    |
| **[Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md) (PyTorch)**              | PyTorch autograd computes exactly what `backward()` computes here, but automatically                               |
| **[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md) (Training at Scale)**    | Mini-batch SGD, learning rate schedules, distributed training                                                      |
| **[Week 15](../../05_deep_learning/week15_cnn_representations/theory.md) (CNNs)**                 | Convolutions are a specialised layer type; backprop extends naturally                                              |

---

## 11. Notebook Reference Guide

| Cell                | Section                | What it demonstrates                                                     | Theory reference |
| ------------------- | ---------------------- | ------------------------------------------------------------------------ | ---------------- |
| 4 (TwoLayerNet)     | Network implementation | Full `forward()`, `backward()`, `compute_loss()`, `train_step()`         | Sections 4–5     |
| 6 (Training)        | Train on make_moons    | 1000 epochs, $\eta = 0.5$; loss curve; train/test accuracy               | Section 8        |
| 7 (Gradient check)  | Validation             | Finite-difference vs. analytic gradients; relative error per param group | Section 6        |
| 9 (Initialisation)  | Init comparison        | Small random vs. Xavier vs. He; loss curves (log scale)                  | Section 7        |
| Ex. 1 (3-layer)     | Deeper network         | Extend to `ThreeLayerNet`; compare decision boundaries                   | Section 4.1      |
| Ex. 2 (Activations) | tanh, Leaky ReLU       | Train with different activations; compare loss curves                    | Section 3.2      |
| Ex. 4 (L2 reg)      | Weight decay           | Add L2 penalty; sweep $\lambda$; plot test accuracy                      | Section 9        |

**Suggested modifications:**

| Modification                                  | What it reveals                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------- |
| Set all weights to zero ($W = 0$)             | All neurons learn identical features (symmetry problem); network fails    |
| Use $\sigma_{\text{init}} = 5.0$              | Activations explode; loss becomes NaN within a few epochs                 |
| Replace ReLU with sigmoid in hidden layers    | Slower convergence; vanishing gradients in deeper networks                |
| Increase hidden size from 10 to 200           | Overfits make_moons easily; test accuracy may drop without regularisation |
| Add a third hidden layer with small init      | Gradients in layer 1 become tiny (vanishing); first layers barely learn   |
| Change learning rate from 0.5 to 0.001        | Very slow convergence — demonstrates the importance of $\eta$ tuning      |
| Plot decision boundary at epoch 10, 100, 1000 | Watch the boundary evolve from linear to curved to sharp                  |

---

## 12. Symbol Reference

| Symbol                          | Name                    | Meaning                                                         |
| ------------------------------- | ----------------------- | --------------------------------------------------------------- |
| $L$                             | Number of layers        | Total layers (including output)                                 |
| $l$                             | Layer index             | $l = 1, \ldots, L$                                              |
| $n^{[l]}$                       | Layer width             | Number of neurons in layer $l$                                  |
| $m$                             | Batch size              | Number of training samples in a batch                           |
| $W^{[l]}$                       | Weight matrix           | Shape $n^{[l-1]} \times n^{[l]}$                                |
| $\mathbf{b}^{[l]}$              | Bias vector             | Shape $1 \times n^{[l]}$                                        |
| $Z^{[l]}$                       | Pre-activations         | $A^{[l-1]}W^{[l]} + \mathbf{b}^{[l]}$; shape $m \times n^{[l]}$ |
| $A^{[l]}$                       | Activations             | $\sigma_l(Z^{[l]})$; shape $m \times n^{[l]}$                   |
| $A^{[0]}$                       | Input                   | $= X$; shape $m \times n^{[0]}$                                 |
| $\sigma(\cdot)$                 | Activation function     | ReLU, sigmoid, tanh, softmax, etc.                              |
| $\sigma'(\cdot)$                | Activation derivative   | Used in backward pass                                           |
| $\odot$                         | Hadamard product        | Elementwise multiplication                                      |
| $\mathcal{L}$                   | Loss                    | Cross-entropy or MSE                                            |
| $dZ^{[l]}$                      | Pre-activation gradient | $\partial\mathcal{L}/\partial Z^{[l]}$                          |
| $dW^{[l]}$                      | Weight gradient         | $\partial\mathcal{L}/\partial W^{[l]}$                          |
| $d\mathbf{b}^{[l]}$             | Bias gradient           | $\partial\mathcal{L}/\partial\mathbf{b}^{[l]}$                  |
| $\eta$                          | Learning rate           | Step size for gradient descent                                  |
| $\lambda$                       | Regularisation strength | L2 penalty coefficient                                          |
| $\epsilon$                      | Finite-difference step  | $\approx 10^{-5}$ for gradient checking                         |
| $n_{\text{in}}, n_{\text{out}}$ | Fan-in, fan-out         | Input and output dimensions of a layer                          |
| $\|\cdot\|_F$                   | Frobenius norm          | $\sqrt{\sum_{ij}w_{ij}^2}$                                      |

---

## 13. References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapters 6–8. MIT Press. Available at [deeplearningbook.org](https://www.deeplearningbook.org/). — The standard reference for feedforward networks, backpropagation, regularisation, and optimisation.
2. Nielsen, M. (2015). *Neural Networks and Deep Learning*. Available at [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/). — Excellent visual explanations; builds networks from scratch in Python.
3. Karpathy, A. (2016). "CS231n: Convolutional Neural Networks for Visual Recognition." Stanford. — Backpropagation notes; computational graph perspective; gradient checking.
4. Glorot, X. & Bengio, Y. (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." *AISTATS*. — Xavier initialisation; analysis of activation variance through layers.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV*. — He/Kaiming initialisation for ReLU networks; PReLU.
6. Cybenko, G. (1989). "Approximation by Superpositions of a Sigmoidal Function." *Mathematics of Control, Signals and Systems*, 2(4), 303–314. — Universal approximation theorem (sigmoid).
7. Hornik, K. (1991). "Approximation Capabilities of Multilayer Feedforward Networks." *Neural Networks*, 4(2), 251–257. — Generalisation of universal approximation to arbitrary activations.
8. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323, 533–536. — The seminal backpropagation paper.
9. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). "Efficient BackProp." In *Neural Networks: Tricks of the Trade*, Springer. — Practical advice on learning rates, initialisation, preprocessing.
10. Neal, R. M. (1996). *Bayesian Learning for Neural Networks*. Springer. — Infinite-width networks → GPs; foundation for understanding the prior that initialisation implies.
