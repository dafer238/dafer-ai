# Training Pathologies

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [The Gradient Flow Problem](#2-the-gradient-flow-problem)
   - 2.1 [Forward Pass: Activation Magnitudes](#21-forward-pass-activation-magnitudes)
   - 2.2 [Backward Pass: Gradient Magnitudes](#22-backward-pass-gradient-magnitudes)
   - 2.3 [The Exponential Effect of Depth](#23-the-exponential-effect-of-depth)
3. [Vanishing Gradients](#3-vanishing-gradients)
   - 3.1 [Mechanism](#31-mechanism)
   - 3.2 [Diagnosis](#32-diagnosis)
   - 3.3 [Consequences](#33-consequences)
4. [Exploding Gradients](#4-exploding-gradients)
   - 4.1 [Mechanism](#41-mechanism)
   - 4.2 [Diagnosis](#42-diagnosis)
   - 4.3 [Consequences](#43-consequences)
5. [Activation Saturation and Dead Neurons](#5-activation-saturation-and-dead-neurons)
   - 5.1 [Sigmoid and Tanh Saturation](#51-sigmoid-and-tanh-saturation)
   - 5.2 [Dead ReLU Neurons](#52-dead-relu-neurons)
6. [Fix 1: Proper Initialisation](#6-fix-1-proper-initialisation)
7. [Fix 2: Batch Normalisation](#7-fix-2-batch-normalisation)
   - 7.1 [The Idea](#71-the-idea)
   - 7.2 [The Algorithm](#72-the-algorithm)
   - 7.3 [Training vs. Inference](#73-training-vs-inference)
   - 7.4 [Why BatchNorm Works](#74-why-batchnorm-works)
   - 7.5 [Where to Place BatchNorm](#75-where-to-place-batchnorm)
8. [Fix 3: Layer Normalisation](#8-fix-3-layer-normalisation)
9. [Fix 4: Gradient Clipping](#9-fix-4-gradient-clipping)
   - 9.1 [Clip-by-Norm](#91-clip-by-norm)
   - 9.2 [Clip-by-Value](#92-clip-by-value)
   - 9.3 [When to Use Gradient Clipping](#93-when-to-use-gradient-clipping)
10. [Fix 5: Residual (Skip) Connections](#10-fix-5-residual-skip-connections)
    - 10.1 [The Identity Shortcut](#101-the-identity-shortcut)
    - 10.2 [Why Residual Connections Fix Gradient Flow](#102-why-residual-connections-fix-gradient-flow)
11. [Diagnostic Toolkit Summary](#11-diagnostic-toolkit-summary)
12. [Connections to the Rest of the Course](#12-connections-to-the-rest-of-the-course)
13. [Notebook Reference Guide](#13-notebook-reference-guide)
14. [Symbol Reference](#14-symbol-reference)
15. [References](#15-references)

---

## 1. Scope and Purpose

[Week 11](../week11_nn_from_scratch/theory.md) built a neural network from scratch. This week asks: **what goes wrong when we make it deeper?**

A 2-layer network on `make_moons` trains easily. A 10-layer network on the same task often fails completely — the loss barely moves, or it diverges. The problem is not the architecture's expressive power (universal approximation guarantees that) but the **dynamics of gradient flow** through many layers.

This week provides:
1. **Diagnostic skills** — how to detect vanishing/exploding gradients and dead neurons.
2. **A toolbox of fixes** — initialisation (review from [Week 11](../week11_nn_from_scratch/theory.md)), batch normalisation, layer normalisation, gradient clipping, residual connections.
3. **The intuition** for why each fix works, grounded in the mathematics of gradient flow.

These diagnostic techniques apply to every deep-learning week that follows.

**Prerequisites.** [Week 11](../week11_nn_from_scratch/theory.md) (backpropagation, chain rule through layers, initialisation).

---

## 2. The Gradient Flow Problem

### 2.1 Forward Pass: Activation Magnitudes

Consider a deep network with $L$ layers and identical weight matrices (a simplification for analysis):

$$\mathbf{a}^{[l]} = \sigma(W^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})$$

If we ignore biases and activations for a moment:

$$\mathbf{a}^{[L]} = W^{[L]}W^{[L-1]}\cdots W^{[1]}\mathbf{x}$$

The magnitude of the output depends on the product of $L$ matrices. If the weight matrices consistently scale their inputs by a factor $s$:

$$\|\mathbf{a}^{[L]}\| \approx s^L\|\mathbf{x}\|$$

| $s$       | Effect after $L = 50$ layers                  |
| --------- | --------------------------------------------- |
| $s = 0.9$ | $0.9^{50} \approx 0.005$ — activations vanish |
| $s = 1.0$ | $1.0^{50} = 1$ — stable                       |
| $s = 1.1$ | $1.1^{50} \approx 117$ — activations explode  |

---

### 2.2 Backward Pass: Gradient Magnitudes

The gradient of the loss with respect to the weights at layer $l$ involves the chain of Jacobians from layer $L$ back to layer $l$ ([Week 11](../week11_nn_from_scratch/theory.md)):

$$\frac{\partial\mathcal{L}}{\partial W^{[l]}} = \frac{\partial\mathcal{L}}{\partial Z^{[L]}}\cdot\prod_{k=l+1}^{L}\left(\text{diag}(\sigma'(Z^{[k-1]}))\cdot W^{[k]}\right)\cdot (A^{[l-1]})^\top$$

Each factor in the product contributes:

$$\text{diag}(\sigma'(Z^{[k]}))\cdot W^{[k+1]}$$

The gradient magnitude at layer $l$ depends on the product of $(L - l)$ such factors. This product suffers the same exponential scaling problem as the forward pass.

---

### 2.3 The Exponential Effect of Depth

$$\boxed{\left\|\frac{\partial\mathcal{L}}{\partial W^{[l]}}\right\| \approx c^{L-l}\left\|\frac{\partial\mathcal{L}}{\partial W^{[L]}}\right\|}$$

where $c$ is the effective scaling factor per layer (depends on weights and activation derivatives).

| $c$     | Gradient behaviour                                 | Name                    |
| ------- | -------------------------------------------------- | ----------------------- |
| $c < 1$ | Gradients shrink exponentially toward early layers | **Vanishing gradients** |
| $c = 1$ | Gradients maintain magnitude                       | **Ideal**               |
| $c > 1$ | Gradients grow exponentially toward early layers   | **Exploding gradients** |

> **The fundamental challenge of deep learning is maintaining $c \approx 1$ across all layers.** Every fix in this week's toolbox addresses this single goal.

---

## 3. Vanishing Gradients

### 3.1 Mechanism

**Sigmoid and tanh.** The derivative of sigmoid is:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) \leq 0.25$$

The maximum value (at $z = 0$) is 0.25. Each layer multiplies gradients by at most 0.25 from the activation alone. Over $L$ layers:

$$\prod_{l=1}^{L}\sigma'(z^{[l]}) \leq 0.25^L$$

For $L = 10$: $0.25^{10} \approx 10^{-6}$. Gradients in the first layer are a million times smaller than in the last layer.

**Tanh** is better but not immune: $\tanh'(z) \leq 1$ (at $z = 0$), but for most activations $|\tanh'(z)| \ll 1$.

---

### 3.2 Diagnosis

1. **Plot gradient norms per layer.** Vanishing gradients show a steep exponential drop from the output layer to the input layer on a log-scale plot.

```python
def compute_gradient_norms(model):
    return [layer.weight.grad.norm().item()
            for layer in model.layers
            if hasattr(layer, 'weight') and layer.weight.grad is not None]
```

2. **Check activation distributions.** If activations pile up near 0 and 1 (sigmoid) or $\pm 1$ (tanh), the neurons are saturated and their derivatives are near zero.

3. **Monitor training loss.** Vanishing gradients cause the loss to plateau early — the model appears to "not learn."

> **Notebook reference.** Cell 4 builds a 10-layer `DeepNet` and plots gradient norms for sigmoid, tanh, and ReLU. Sigmoid shows gradient norms dropping by orders of magnitude from layer 10 to layer 1.

---

### 3.3 Consequences

| Symptom                           | Explanation                                                               |
| --------------------------------- | ------------------------------------------------------------------------- |
| **Early layers don't learn**      | Gradient is too small to meaningfully update $W^{[1]}, W^{[2]}$           |
| **Slow convergence**              | Only the last few layers learn; most parameters are wasted                |
| **Feature extraction fails**      | The first layers are supposed to learn useful representations; they can't |
| **Increasing width doesn't help** | The problem is depth-dependent, not width-dependent                       |

---

## 4. Exploding Gradients

### 4.1 Mechanism

If weight magnitudes are slightly too large ($\|W^{[l]}\| > 1/\max\sigma'$), each layer amplifies the gradient:

$$\left\|\frac{\partial\mathcal{L}}{\partial W^{[l]}}\right\| \sim c^{L-l}, \quad c > 1$$

With ReLU ($\sigma'(z) = 1$ for $z > 0$), the activation derivative doesn't attenuate — the gradient passes through unchanged. If $\|W^{[l]}\| > 1$ for all layers, gradients grow exponentially.

---

### 4.2 Diagnosis

1. **Gradient norms grow toward early layers** on a log-scale plot (opposite of vanishing).
2. **Loss spikes or becomes NaN.** A single large update can push the parameters into a catastrophic region.
3. **Weights grow unboundedly.** Inspect $\|W^{[l]}\|$ over training epochs.

---

### 4.3 Consequences

| Symptom                          | Explanation                                                |
| -------------------------------- | ---------------------------------------------------------- |
| **NaN loss**                     | Overflow in activations or loss computation                |
| **Oscillating / diverging loss** | Gradient updates overshoot by a huge margin                |
| **Numerical instability**        | $\exp(z)$ in softmax overflows; $\log(0)$ in cross-entropy |

> **Vanishing gradients are insidious (silent failure); exploding gradients are loud (NaN/divergence).** Both are solved by the same set of tools.

---

## 5. Activation Saturation and Dead Neurons

### 5.1 Sigmoid and Tanh Saturation

A neuron is **saturated** when its pre-activation $z$ has a large magnitude:

$$|z| \gg 0 \implies \sigma'(z) \approx 0$$

**For sigmoid:** $z > 5 \implies \sigma(z) \approx 1, \sigma'(z) \approx 0$. The neuron outputs a near-constant value regardless of input changes, and the gradient through it is essentially zero.

**For tanh:** $|z| > 3 \implies \tanh(z) \approx \pm 1, \tanh'(z) \approx 0$.

**Causes of saturation:**
- Weights too large (poor initialisation).
- Inputs not centred (all positive or all negative).
- During training, weights grow and push pre-activations into saturation regions.

---

### 5.2 Dead ReLU Neurons

ReLU doesn't saturate for $z > 0$, but it has a different problem: **dead neurons**.

$$\text{ReLU}(z) = \begin{cases}z & z > 0\\0 & z \leq 0\end{cases} \implies \text{ReLU}'(z) = \begin{cases}1 & z > 0\\0 & z \leq 0\end{cases}$$

If a neuron's pre-activation $z$ is negative for all training samples, then:
- Its activation is always 0.
- Its gradient is always 0.
- Its weights never receive any gradient — the neuron is **permanently dead**.

**Causes:**
- Large learning rate causes a big update that pushes $W$ into a region where $z < 0$ for all inputs.
- Bias initialised too negative.
- Poor initialisation combined with all-positive or all-negative input distributions.

**Fixes:**
- **Leaky ReLU:** $\text{LeakyReLU}(z) = \max(\alpha z, z)$ with $\alpha \approx 0.01$. Small but non-zero gradient for $z < 0$.
- **ELU:** $\text{ELU}(z) = z$ for $z > 0$, $\alpha(e^z - 1)$ for $z \leq 0$. Smooth and non-zero for $z < 0$.
- **Proper initialisation** (He init) to keep pre-activations centred.
- **Lower learning rate** to avoid catastrophic updates.

> **Notebook reference.** Cell 6 trains 6-layer networks with sigmoid, tanh, and ReLU for 200 epochs. Sigmoid barely learns; tanh is slow; ReLU converges well.

---

## 6. Fix 1: Proper Initialisation

Covered in detail in [Week 11](../week11_nn_from_scratch/theory.md), Section 7. The key results:

$$\text{Xavier:}\quad \text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \qquad \text{(sigmoid/tanh)}$$

$$\text{He:}\quad \text{Var}(W) = \frac{2}{n_{\text{in}}} \qquad \text{(ReLU)}$$

**Why it fixes gradient flow.** If $\text{Var}(W) = c/n_{\text{in}}$ with appropriate $c$:

$$\text{Var}(z^{[l]}) = n_{\text{in}}\cdot\text{Var}(W)\cdot\text{Var}(a^{[l-1]}) = c\cdot\text{Var}(a^{[l-1]})$$

With $c = 1$ (Xavier forward) or $c = 1$ after accounting for ReLU's halving (He), the variance of activations stays constant across layers. By a symmetric argument, the variance of gradients also stays constant across layers in the backward pass.

> **Initialisation is necessary but not sufficient.** For very deep networks ($L > 20$), even He init leads to degradation after many layers. The remaining fixes address this.

> **Notebook reference.** Cell 8 compares small random, Xavier, and He init for an 8-layer ReLU network. He init gives the best loss curve.

---

## 7. Fix 2: Batch Normalisation

### 7.1 The Idea

**Batch Normalisation** (Ioffe & Szegedy, 2015) normalises the pre-activations in each layer to have zero mean and unit variance, using statistics from the current mini-batch.

By keeping activations in a well-behaved range, BatchNorm prevents both vanishing and exploding activations, regardless of depth.

---

### 7.2 The Algorithm

For a mini-batch $\{z_1, \ldots, z_m\}$ at a given layer (each $z_i \in \mathbb{R}^d$, where $d$ is the layer width):

**Step 1. Compute batch statistics:**

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m}z_i, \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(z_i - \mu_B)^2$$

**Step 2. Normalise:**

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

where $\epsilon \approx 10^{-5}$ prevents division by zero.

**Step 3. Scale and shift (learnable):**

$$\boxed{\tilde{z}_i = \gamma\odot\hat{z}_i + \beta}$$

where $\gamma, \beta \in \mathbb{R}^d$ are **learnable parameters** (initialised to $\gamma = 1, \beta = 0$).

> **Why the learnable $\gamma$ and $\beta$?** Without them, the normalisation would force all layer outputs to have mean 0 and variance 1, which might be too restrictive. The learnable parameters allow the network to **undo** the normalisation if it's not helpful — in the limit, setting $\gamma = \sigma_B, \beta = \mu_B$ recovers the original pre-activations.

---

### 7.3 Training vs. Inference

| Phase         | Batch statistics                                         | Running statistics                                                                           |
| ------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Training**  | $\mu_B, \sigma_B^2$ computed from the current mini-batch | Exponential moving average updated: $\hat{\mu} \leftarrow (1-\alpha)\hat{\mu} + \alpha\mu_B$ |
| **Inference** | Not used                                                 | Use the stored running mean $\hat{\mu}$ and variance $\hat{\sigma}^2$                        |

At inference time, we don't have a mini-batch (or the prediction should not depend on other samples in the batch). The running statistics accumulated during training are used instead.

```python
model.train()   # use batch statistics; update running stats
model.eval()    # use running statistics; freeze updates
```

> **Critical pitfall.** Forgetting to call `model.eval()` at test time means BatchNorm uses batch statistics from the test mini-batch, which can give inconsistent results (especially for small test batches).

---

### 7.4 Why BatchNorm Works

Several complementary explanations exist:

**1. Reduces internal covariate shift (original claim).** Each layer's input distribution stays stable, so each layer doesn't need to continuously adapt to shifting inputs from the previous layer.

**2. Smooths the loss landscape (Santurkar et al., 2018).** BatchNorm makes the loss surface more Lipschitz-smooth, allowing larger learning rates and faster convergence. This is likely the primary mechanism.

**3. Prevents activation saturation.** By centring pre-activations around zero and controlling their variance, BatchNorm keeps sigmoid/tanh out of saturation and keeps ReLU from systematic all-positive or all-negative regimes.

**4. Implicit regularisation.** The batch statistics add noise to the gradients (each sample's normalisation depends on the other samples in the mini-batch), which has a regularising effect similar to dropout.

---

### 7.5 Where to Place BatchNorm

Two conventions:

$$\text{Option A:}\quad \text{Linear} \to \text{BatchNorm} \to \text{Activation}$$

$$\text{Option B:}\quad \text{Linear} \to \text{Activation} \to \text{BatchNorm}$$

Option A (normalise the pre-activations) is more common and was proposed in the original paper. Option B (normalise the activations) works well in practice too.

> **Notebook reference.** Cell 10 implements `DeepNetWithBatchNorm` using Option A (Linear → BN → ReLU) and compares a 10-layer network with and without BatchNorm. The BatchNorm version converges faster and to a lower loss.

---

## 8. Fix 3: Layer Normalisation

**Layer Normalisation** (Ba et al., 2016) normalises across the **feature dimension** instead of the **batch dimension**:

$$\hat{z}_i = \frac{z_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \qquad \mu_i = \frac{1}{d}\sum_{j=1}^{d}z_{ij}, \qquad \sigma_i^2 = \frac{1}{d}\sum_{j=1}^{d}(z_{ij} - \mu_i)^2$$

| Property                    | BatchNorm                                         | LayerNorm                                 |
| --------------------------- | ------------------------------------------------- | ----------------------------------------- |
| **Normalises across**       | Batch dimension (samples)                         | Feature dimension (neurons)               |
| **Statistics**              | Per-feature: $\mu, \sigma$ shape $(d,)$           | Per-sample: $\mu, \sigma$ shape $(m,)$    |
| **Depends on batch?**       | Yes (different batches → different normalisation) | No (each sample normalised independently) |
| **Works for batch size 1?** | No (undefined)                                    | Yes                                       |
| **Running stats needed?**   | Yes (for inference)                               | No                                        |
| **Where used**              | CNNs, feedforward networks                        | Transformers ([Week 18](../../06_sequence_models/week18_transformers/theory.md)), RNNs              |

```python
# PyTorch
nn.BatchNorm1d(num_features)  # normalises across batch for each feature
nn.LayerNorm(num_features)     # normalises across features for each sample
```

> **Why Transformers use LayerNorm, not BatchNorm.** In sequence models, the effective "batch" changes at each token position. LayerNorm's independence from the batch makes it more stable for variable-length sequences.

> **Notebook reference.** Exercise 1 asks you to build `DeepNetWithLayerNorm` and compare it to BatchNorm.

---

## 9. Fix 4: Gradient Clipping

### 9.1 Clip-by-Norm

If the total gradient norm exceeds a threshold $\tau$, rescale it:

$$\boxed{\mathbf{g} \leftarrow \mathbf{g}\cdot\frac{\tau}{\max(\|\mathbf{g}\|, \tau)}}$$

This preserves the gradient **direction** but caps its **magnitude**.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Effect:** if $\|\mathbf{g}\| \leq \tau$, the gradient is unchanged. If $\|\mathbf{g}\| > \tau$, it is rescaled to have norm exactly $\tau$.

---

### 9.2 Clip-by-Value

Clip each gradient component independently:

$$g_j \leftarrow \text{clip}(g_j, -\tau, \tau)$$

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

This changes the gradient direction (unlike clip-by-norm). Generally, clip-by-norm is preferred.

---

### 9.3 When to Use Gradient Clipping

| Scenario                              | Recommendation                                                |
| ------------------------------------- | ------------------------------------------------------------- |
| **RNNs / LSTMs**                      | Almost always clip (long sequences cause exploding gradients) |
| **Transformers**                      | Commonly clip to 1.0                                          |
| **Deep feedforward (with BatchNorm)** | Usually not needed (BatchNorm stabilises gradients)           |
| **Training instability / NaN loss**   | Try clipping as a first diagnostic step                       |

**Choosing $\tau$:** monitor gradient norms during training. Set $\tau$ slightly above the typical norm so that clipping activates only on outlier batches.

> **Notebook reference.** Cell 12 compares training with and without gradient clipping ($\tau = 1.0$) on an 8-layer ReLU network. Without clipping, gradient norms spike and training is unstable; with clipping, training stabilises.

---

## 10. Fix 5: Residual (Skip) Connections

### 10.1 The Identity Shortcut

**He et al. (2015) — Deep Residual Learning.** Instead of learning $H(\mathbf{x})$ directly, learn the **residual** $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$:

$$\boxed{\mathbf{a}^{[l+1]} = \sigma\!\left(F(\mathbf{a}^{[l]}; W^{[l]}) + \mathbf{a}^{[l]}\right)}$$

The "$+ \mathbf{a}^{[l]}$" is the **skip connection** (or shortcut). It allows information (and gradients) to bypass the layer entirely.

```python
# In PyTorch
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x) + x)  # skip connection
```

---

### 10.2 Why Residual Connections Fix Gradient Flow

Consider the gradient through a residual block:

$$\frac{\partial\mathbf{a}^{[l+1]}}{\partial\mathbf{a}^{[l]}} = \frac{\partial F}{\partial\mathbf{a}^{[l]}} + I$$

The **identity matrix $I$** provides a direct gradient path. Even if $\partial F/\partial\mathbf{a}^{[l]}$ vanishes, the gradient through $I$ survives.

For a network with $L$ residual blocks, the gradient at layer $l$ is:

$$\frac{\partial\mathcal{L}}{\partial\mathbf{a}^{[l]}} = \frac{\partial\mathcal{L}}{\partial\mathbf{a}^{[L]}}\prod_{k=l}^{L-1}\left(I + \frac{\partial F_k}{\partial\mathbf{a}^{[k]}}\right)$$

Expanding the product, we get a **sum over all possible paths** from layer $l$ to layer $L$, including the direct path that skips all intermediate layers. This means the gradient always has at least one $O(1)$ path — it cannot vanish exponentially.

> **This is why ResNets can be 100+ layers deep.** Without skip connections, training even 20 layers is difficult. With skip connections, networks of 152 layers (original ResNet) and even 1000+ layers have been successfully trained.

> **Notebook reference.** Exercise 2 asks you to implement `DeepNetResidual` and compare gradient norms to the baseline.

---

## 11. Diagnostic Toolkit Summary

| Pathology               | Diagnostic                                               | Quick Fix                          | Permanent Fix                    |
| ----------------------- | -------------------------------------------------------- | ---------------------------------- | -------------------------------- |
| **Vanishing gradients** | Gradient norms drop exponentially toward early layers    | ReLU instead of sigmoid/tanh       | Residual connections + BatchNorm |
| **Exploding gradients** | Gradient norms spike; loss becomes NaN                   | Gradient clipping ($\tau = 1.0$)   | Proper init + BatchNorm          |
| **Saturated neurons**   | Activations pile up near $\pm 1$ (tanh) or 0/1 (sigmoid) | Clip pre-activations; lower $\eta$ | BatchNorm + ReLU                 |
| **Dead ReLU**           | Fraction of always-zero activations > 50%                | Lower $\eta$; reset dead weights   | Leaky ReLU / ELU + He init       |
| **Loss plateau**        | Loss stops decreasing despite continued training         | Increase $\eta$; try Adam          | Check all of the above           |
| **Oscillating loss**    | Loss bounces wildly between epochs                       | Decrease $\eta$; gradient clipping | Warm-up schedule + clipping      |

**The recommended "modern baseline" for deep networks:**

$$\text{He init} + \text{ReLU / LeakyReLU} + \text{BatchNorm (or LayerNorm)} + \text{Residual connections} + \text{Adam}$$

With these defaults, most training pathologies are avoided from the start.

---

## 12. Connections to the Rest of the Course

| Week                            | Connection                                                                                                                 |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **[[Week 01](../../02_fundamentals/week01_optimization/theory.md)](../../02_fundamentals/week01_optimization/theory.md)–[02](../../02_fundamentals/week02_advanced_optimizers/theory.md) (Optimisation)**   | Adam and momentum help in non-convex landscapes; learning rate schedules interact with BatchNorm                           |
| **[Week 06](../../02_fundamentals/week06_regularization/theory.md) (Regularisation)**    | BatchNorm has implicit regularising effect; weight decay interacts with the $\gamma$ parameter                             |
| **[Week 11](../week11_nn_from_scratch/theory.md) (NN from Scratch)**   | Xavier/He init derived from variance analysis; backprop equations give the gradient product                                |
| **[Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md) (PyTorch)**           | All fixes from this week have one-line PyTorch equivalents: `nn.BatchNorm1d`, `nn.init.kaiming_normal_`, `clip_grad_norm_` |
| **[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md) (Training at Scale)** | Learning rate warm-up prevents early-training instability; mixed precision interacts with gradient scaling                 |
| **[Week 15](../../05_deep_learning/week15_cnn_representations/theory.md) (CNNs)**              | `BatchNorm2d` applied after convolutional layers; residual blocks are the building blocks of ResNet                        |
| **[Week 16](../../05_deep_learning/week16_regularization_dl/theory.md) (Regularisation DL)** | Dropout + BatchNorm interaction; whether to use both simultaneously                                                        |
| **[Week 18](../../06_sequence_models/week18_transformers/theory.md) (Transformers)**      | LayerNorm is standard in Transformers; residual connections around every attention and FFN block                           |

---

## 13. Notebook Reference Guide

| Cell                   | Section               | What it demonstrates                                                         | Theory reference |
| ---------------------- | --------------------- | ---------------------------------------------------------------------------- | ---------------- |
| 4 (Gradient norms)     | 10-layer deep network | Gradient norms per layer for sigmoid/tanh/ReLU (log scale)                   | Section 3        |
| 6 (Activations)        | Training dynamics     | 6-layer network trained 200 epochs; loss curves for 3 activations            | Section 5        |
| 8 (Initialisation)     | Init comparison       | small vs. Xavier vs. He on 8-layer ReLU network                              | Section 6        |
| 10 (BatchNorm)         | Normalisation         | 10-layer with/without BatchNorm; loss comparison                             | Section 7        |
| 12 (Gradient clipping) | Clipping              | 8-layer network with/without clip ($\tau = 1.0$); gradient norm + loss plots | Section 9        |
| Ex. 1 (LayerNorm)      | Normalisation         | LayerNorm vs. BatchNorm vs. none                                             | Section 8        |
| Ex. 2 (Residual)       | Skip connections      | Residual vs. plain network; gradient norms                                   | Section 10       |
| Ex. 4 (LR × Init grid) | Interaction           | 3×3 grid of (init, lr); heatmap of final loss                                | Sections 6, 8.2  |

**Suggested modifications:**

| Modification                                                          | What it reveals                                                                                |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Increase depth to 20 or 50 layers (sigmoid)                           | Gradient norms become essentially zero after ~15 layers; total training failure                |
| Use very large init ($\sigma = 5.0$) with ReLU                        | Activations explode; loss NaN within 1–2 epochs                                                |
| Set BatchNorm momentum to 0.0 (no running average update)             | Inference behaves erratically (uses uninitialised running stats)                               |
| Remove $\gamma, \beta$ from BatchNorm                                 | Network loses representational power; some tasks degrade slightly                              |
| Add residual connections to the 10-layer sigmoid network              | Sigmoid + residuals trains much better — the skip path compensates for saturation              |
| Try gradient clipping thresholds $\tau \in \{0.1, 1.0, 10.0, 100.0\}$ | Too-small $\tau$ slows convergence; too-large $\tau$ doesn't prevent spikes                    |
| Compare SGD vs. Adam on the 10-layer network without BatchNorm        | Adam partially compensates for gradient pathologies via per-parameter scaling                  |
| Plot activation histograms per layer during training                  | Watch sigmoid activations migrate toward 0/1 (saturation); ReLU stays healthy with proper init |

---

## 14. Symbol Reference

| Symbol                      | Name                      | Meaning                                                             |
| --------------------------- | ------------------------- | ------------------------------------------------------------------- |
| $L$                         | Number of layers          | Depth of the network                                                |
| $l$                         | Layer index               | $l = 1, \ldots, L$                                                  |
| $Z^{[l]}$                   | Pre-activations           | Before nonlinearity at layer $l$                                    |
| $A^{[l]}$                   | Activations               | After nonlinearity at layer $l$                                     |
| $\sigma'(z)$                | Activation derivative     | Gradient multiplier; $\leq 0.25$ for sigmoid                        |
| $\|\mathbf{g}\|$            | Gradient norm             | $\sqrt{\sum_i g_i^2}$; tracked per layer                            |
| $c$                         | Effective scaling factor  | Product of $\|\sigma'(z)\|\cdot\|W\|$ per layer                     |
| $\mu_B$                     | Batch mean                | $\frac{1}{m}\sum z_i$ (per feature)                                 |
| $\sigma_B^2$                | Batch variance            | $\frac{1}{m}\sum (z_i - \mu_B)^2$                                   |
| $\hat{z}$                   | Normalised pre-activation | $(z - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$                          |
| $\gamma$                    | BN scale parameter        | Learnable; init $= 1$                                               |
| $\beta$                     | BN shift parameter        | Learnable; init $= 0$                                               |
| $\epsilon$                  | BN stability constant     | $\approx 10^{-5}$                                                   |
| $\hat{\mu}, \hat{\sigma}^2$ | Running statistics        | Exponential moving average for inference                            |
| $\tau$                      | Clipping threshold        | Max allowed gradient norm                                           |
| $F(\mathbf{x})$             | Residual function         | What the block learns; $H(\mathbf{x}) = F(\mathbf{x}) + \mathbf{x}$ |
| $I$                         | Identity matrix           | Provides the skip connection gradient                               |
| $\alpha$                    | Leaky ReLU slope          | Gradient for $z < 0$; typically 0.01                                |
| $\eta$                      | Learning rate             | Step size; interacts with all fixes                                 |

---

## 15. References

1. Ioffe, S. & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML*. — The original BatchNorm paper.
2. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization." *arXiv:1607.06450*. — LayerNorm; used in Transformers.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*. — ResNet; skip connections enabling 100+ layer networks.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving Deep into Rectifiers." *ICCV*. — He/Kaiming initialisation for ReLU.
5. Glorot, X. & Bengio, Y. (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." *AISTATS*. — Xavier initialisation; analysis of gradient flow.
6. Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). "How Does Batch Normalization Help Optimization?" *NeurIPS*. — Challenges the "internal covariate shift" explanation; shows BN smooths the loss landscape.
7. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). "On the Difficulty of Training Recurrent Neural Networks." *ICML*. — Gradient clipping for RNNs; characterises vanishing/exploding gradients.
8. Hochreiter, S. (1991). "Untersuchungen zu dynamischen neuronalen Netzen." Diploma thesis. — First identification of the vanishing gradient problem.
9. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapters 6–8. MIT Press. — Comprehensive treatment of training dynamics, initialisation, and normalisation.
10. Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). "Self-Normalizing Neural Networks." *NeurIPS*. — SELU activation with LeCun init; networks that automatically maintain unit variance.
