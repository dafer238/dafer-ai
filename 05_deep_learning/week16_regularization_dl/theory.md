# Regularisation for Deep Learning

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Overfitting in Deep Networks](#2-overfitting-in-deep-networks)
   - 2.1 [Why Deep Models Overfit](#21-why-deep-models-overfit)
   - 2.2 [The Bias–Variance Lens](#22-the-biasvariance-lens)
   - 2.3 [Detecting Overfitting](#23-detecting-overfitting)
3. [Dropout](#3-dropout)
   - 3.1 [The Algorithm](#31-the-algorithm)
   - 3.2 [Training vs. Inference](#32-training-vs-inference)
   - 3.3 [Why Dropout Works](#33-why-dropout-works)
   - 3.4 [Where to Place Dropout](#34-where-to-place-dropout)
   - 3.5 [Choosing the Dropout Rate](#35-choosing-the-dropout-rate)
4. [Weight Decay (L2 Regularisation)](#4-weight-decay-l2-regularisation)
   - 4.1 [The Penalty](#41-the-penalty)
   - 4.2 [Weight Decay vs. L2 Regularisation](#42-weight-decay-vs-l2-regularisation)
   - 4.3 [Effect on the Loss Landscape](#43-effect-on-the-loss-landscape)
5. [Data Augmentation](#5-data-augmentation)
   - 5.1 [The Idea](#51-the-idea)
   - 5.2 [Common Augmentations for Images](#52-common-augmentations-for-images)
   - 5.3 [PyTorch Transforms Pipeline](#53-pytorch-transforms-pipeline)
   - 5.4 [Augmentation as Implicit Regularisation](#54-augmentation-as-implicit-regularisation)
6. [BatchNorm as a Regulariser](#6-batchnorm-as-a-regulariser)
7. [BatchNorm + Dropout Interaction](#7-batchnorm--dropout-interaction)
8. [Early Stopping](#8-early-stopping)
9. [Ensemble Methods](#9-ensemble-methods)
   - 9.1 [Why Ensembles Regularise](#91-why-ensembles-regularise)
   - 9.2 [Prediction Averaging](#92-prediction-averaging)
   - 9.3 [Dropout as Approximate Ensemble](#93-dropout-as-approximate-ensemble)
10. [Label Smoothing](#10-label-smoothing)
11. [Regularisation Interactions and Ablation Protocol](#11-regularisation-interactions-and-ablation-protocol)
12. [Connections to the Rest of the Course](#12-connections-to-the-rest-of-the-course)
13. [Notebook Reference Guide](#13-notebook-reference-guide)
14. [Symbol Reference](#14-symbol-reference)
15. [References](#15-references)

---

## 1. Scope and Purpose

[Week 06](../../02_fundamentals/week06_regularization/theory.md) introduced regularisation for linear models (Ridge, Lasso, cross-validation). This week is the deep-learning counterpart: the goal — controlling overfitting — is identical, but the tools change because deep networks have millions of parameters, stochastic training, and layer-specific dynamics.

This week delivers:

1. **The deep-learning regularisation toolkit** — Dropout, weight decay, data augmentation, BatchNorm's regularising effect, ensembles, early stopping, label smoothing.
2. **Understanding interactions** — these techniques are not independent. Dropout + BatchNorm can conflict; weight decay + Adam behaves differently from weight decay + SGD; augmentation can substitute for explicit regularisation.
3. **The ablation discipline** — systematically varying one regulariser at a time and recording the result in a table. This methodology carries through every subsequent experiment.

**Prerequisites.** [Week 06](../../02_fundamentals/week06_regularization/theory.md) (L1/L2 regularisation, cross-validation), [Week 12](../../04_neural_networks/week12_training_pathologies/theory.md) (BatchNorm), [Week 13](../week13_pytorch_basics/theory.md) (PyTorch training loop), [Week 15](../week15_cnn_representations/theory.md) (CNN to regularise).

---

## 2. Overfitting in Deep Networks

### 2.1 Why Deep Models Overfit

A neural network with $P$ parameters and $N$ training samples:

| Regime        | Behaviour                                                  |
| ------------- | ---------------------------------------------------------- |
| $P \ll N$     | Underfitting likely — model too simple                     |
| $P \approx N$ | Classical overfitting regime                               |
| $P \gg N$     | **Deep learning regime** — can memorise the entire dataset |

Zhang et al. (2017) showed that standard deep networks can perfectly memorise random labels — their capacity exceeds the dataset size by orders of magnitude. Without regularisation, there is nothing preventing the model from fitting noise.

---

### 2.2 The Bias–Variance Lens

From [Week 06](../../02_fundamentals/week06_regularization/theory.md):

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Technique                         | Effect on bias | Effect on variance |
| --------------------------------- | -------------- | ------------------ |
| **More capacity** (wider, deeper) | ↓              | ↑                  |
| **Dropout**                       | ↑ (slightly)   | ↓↓                 |
| **Weight decay**                  | ↑ (slightly)   | ↓                  |
| **Data augmentation**             | —              | ↓↓                 |
| **Early stopping**                | ↑ (slightly)   | ↓                  |
| **Ensemble**                      | —              | ↓↓                 |

Regularisation accepts a small increase in bias to achieve a large decrease in variance. The net effect is lower test error.

---

### 2.3 Detecting Overfitting

| Signal                             | Meaning                                             |
| ---------------------------------- | --------------------------------------------------- |
| Train loss ↓, val loss ↑           | Classic overfitting — model memorises training data |
| Train acc → 100%, val acc plateaus | Gap = overfitting magnitude                         |
| Train loss ≈ val loss              | Well-regularised or underfitting                    |

**Always plot both train and validation curves.** The gap between them is the primary diagnostic.

---

## 3. Dropout

### 3.1 The Algorithm

During training, each neuron's output is independently set to zero with probability $p$. The surviving outputs are scaled by $1/(1-p)$ to maintain the expected value:

$$\boxed{\tilde{a}_j = \frac{m_j}{1 - p}\cdot a_j, \qquad m_j \sim \text{Bernoulli}(1 - p)}$$

where $a_j$ is the activation before dropout, $m_j$ is a binary mask, and $\tilde{a}_j$ is the dropped-out activation.

**Why the $1/(1-p)$ scaling?** Without it:

$$\mathbb{E}[\tilde{a}_j] = (1 - p)\cdot a_j \neq a_j$$

The scaling ensures $\mathbb{E}[\tilde{a}_j] = a_j$, so the expected output is the same during training and inference. This is called **inverted dropout** (the standard in PyTorch).

---

### 3.2 Training vs. Inference

| Phase           | Dropout behaviour                    |
| --------------- | ------------------------------------ |
| `model.train()` | Mask active, scaling active          |
| `model.eval()`  | Mask disabled, no scaling (identity) |

```python
model.train()   # dropout active
loss = criterion(model(x), y)

model.eval()    # dropout off
with torch.no_grad():
    pred = model(x_test)
```

> **Critical bug.** Forgetting `model.eval()` at test time means dropout is still active, giving noisy predictions and lower accuracy.

---

### 3.3 Why Dropout Works

Three complementary explanations:

**1. Implicit ensemble (Srivastava et al., 2014).** Each training step uses a different random sub-network (defined by the mask). A network with $d$ neurons has $2^d$ possible sub-networks. At test time, the scaled full network approximates the geometric mean of all sub-networks.

**2. Prevents co-adaptation.** Without dropout, neurons can specialise in narrow partnerships (e.g., neuron A fires only when neuron B fires). Dropout forces each neuron to learn features useful on its own, because its partners are randomly absent.

**3. Noise injection.** The random mask adds noise to the forward pass, similar to adding noise to the data. This noise acts as a regulariser, smoothing the loss landscape.

---

### 3.4 Where to Place Dropout

```python
class MLPWithDropout(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(dropout_p)    # after activation
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(dropout_p)    # after activation
        self.fc3 = nn.Linear(256, 10)            # no dropout before output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        return self.fc3(x)
```

**Guidelines:**
- **After activation**, before the next linear layer.
- **Not after the output layer** — you want the full prediction, not a randomly masked one.
- **In CNNs:** less common between conv layers (spatial dropout `nn.Dropout2d` is sometimes used). More common after the flatten→FC transition.
- **In Transformers:** after attention and after the feedforward block ([Week 18](../../06_sequence_models/week18_transformers/theory.md)).

---

### 3.5 Choosing the Dropout Rate

| $p$     | Regime     | Typical use                              |
| ------- | ---------- | ---------------------------------------- |
| 0.0     | No dropout | Baseline                                 |
| 0.1–0.2 | Light      | When data is plentiful or model is small |
| 0.3–0.5 | Standard   | Default for FC layers                    |
| 0.5–0.8 | Heavy      | Very large models, limited data          |

**Practical rule.** Start with $p = 0.5$ for FC layers (the original paper's recommendation). Reduce if training struggles to converge; increase if overfitting persists.

> **Notebook reference.** Cell 1 compares `MLPWithDropout(dropout_p=0.0)` vs. `dropout_p=0.5` on MNIST. Without dropout, training loss is lower but the test accuracy gap is larger.

---

## 4. Weight Decay (L2 Regularisation)

### 4.1 The Penalty

Add a penalty on the squared magnitude of all weights to the loss:

$$\boxed{\tilde{\mathcal{L}} = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2 = \mathcal{L}(\theta) + \frac{\lambda}{2}\sum_j\theta_j^2}$$

The gradient becomes:

$$\nabla_\theta\tilde{\mathcal{L}} = \nabla_\theta\mathcal{L} + \lambda\theta$$

The update rule:

$$\theta_{t+1} = \theta_t - \eta(\nabla_\theta\mathcal{L} + \lambda\theta_t) = (1 - \eta\lambda)\theta_t - \eta\nabla_\theta\mathcal{L}$$

The factor $(1 - \eta\lambda)$ shrinks weights toward zero at every step — hence the name "weight decay."

---

### 4.2 Weight Decay vs. L2 Regularisation

For SGD, L2 regularisation and weight decay are mathematically equivalent: both produce the $(1 - \eta\lambda)$ shrinkage. For **Adam**, they differ:

**L2 regularisation (modified gradient):**

$$\hat{g}_t = g_t + \lambda\theta_t \qquad\text{(penalty added to gradient)}$$

Adam normalises this modified gradient by $\sqrt{v_t}$, partially undoing the regularisation for parameters with large gradients.

**Decoupled weight decay (AdamW, Loshchilov & Hutter 2019):**

$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Weight decay is applied **directly to the parameters**, not through the gradient. This preserves the intended regularisation regardless of Adam's adaptive scaling.

```python
# L2 regularisation (standard Adam)
optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Decoupled weight decay (recommended)
optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

> **Practical recommendation.** Use `AdamW` with `weight_decay=0.01`–`0.1` instead of `Adam` with `weight_decay`. The default in modern training recipes is AdamW.

---

### 4.3 Effect on the Loss Landscape

Weight decay penalises large weights, which:

1. **Keeps the model in a "simpler" region** of parameter space (smaller weights → smoother function → lower complexity).
2. **Prevents weights from growing unboundedly** during long training runs.
3. **Interacts with learning rate schedules** — the effective regularisation strength is $\eta\lambda$, so reducing $\eta$ also reduces the shrinkage.

> **Notebook reference.** Cell 3 sweeps weight decay over $\{0, 10^{-4}, 10^{-3}, 10^{-2}\}$ and plots accuracy. Moderate weight decay helps; too much hurts.

---

## 5. Data Augmentation

### 5.1 The Idea

Data augmentation creates new training samples by applying label-preserving transformations to existing ones. This effectively increases the dataset size and exposes the model to variations it will encounter at test time.

**Formal view.** If $T$ is a transformation such that $y(T(x)) = y(x)$ (the label doesn't change), then augmentation creates virtual samples $(T(x), y)$. The model is trained to be invariant to $T$.

---

### 5.2 Common Augmentations for Images

| Transform                   | Parameters                                 | Effect                        |
| --------------------------- | ------------------------------------------ | ----------------------------- |
| **Random horizontal flip**  | $p = 0.5$                                  | Mirror image left-right       |
| **Random rotation**         | degrees $\in [-15, 15]$                    | Small rotations               |
| **Random crop**             | padding, then crop to original size        | Small translations            |
| **Random affine**           | translate, scale, shear                    | Geometric distortions         |
| **Colour jitter**           | brightness, contrast, saturation, hue      | Photometric variations        |
| **Random erasing / Cutout** | patch size                                 | Occlusion robustness          |
| **Mixup**                   | $\tilde{x} = \lambda x_i + (1-\lambda)x_j$ | Convex combination of samples |
| **CutMix**                  | Paste patch from one image onto another    | Regional mixing               |

---

### 5.3 PyTorch Transforms Pipeline

```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST('./data', train=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
```

> **Critical:** augmentation is applied only to training data, never to test data. Test-time augmentation (TTA) — running multiple augmented versions at test time and averaging — is a separate technique.

---

### 5.4 Augmentation as Implicit Regularisation

Data augmentation is the **most effective single regulariser** in deep learning for image tasks. It works by:

1. **Increasing effective dataset size.** Each epoch sees different augmented versions.
2. **Encoding invariances.** Random flips teach the model that left-right orientation doesn't matter; rotations teach rotational tolerance.
3. **Reducing reliance on spurious features.** Random cropping prevents the model from relying on absolute pixel positions.

> **Notebook reference.** Cell 2 applies `RandomRotation(10)` + `RandomAffine` to MNIST and compares to the un-augmented baseline.

---

## 6. BatchNorm as a Regulariser

From [Week 12](../../04_neural_networks/week12_training_pathologies/theory.md), BatchNorm normalises activations using mini-batch statistics. This introduces **noise** because:

- Each sample's normalisation depends on the other samples in the batch.
- Different batches produce different $\mu_B, \sigma_B^2$.

This noise acts similarly to dropout — it prevents the model from relying too precisely on any single activation pattern. Experiments show that BatchNorm alone can reduce overfitting enough to make dropout unnecessary in some architectures (e.g., ResNets).

**Evidence (Ioffe & Szegedy, 2015; Santurkar et al., 2018):**
- Models with BatchNorm generalise better than without, even when both achieve near-zero training loss.
- Removing BatchNorm and adding stronger dropout often gives similar regularisation but with slower training.

---

## 7. BatchNorm + Dropout Interaction

Using both simultaneously can cause problems:

**The issue.** During training, dropout scales activations by $1/(1-p)$. BatchNorm then computes running statistics on these scaled activations. At inference, dropout is off but BatchNorm uses the training-time running statistics — which were computed with the dropout scaling. This **mismatch** can hurt accuracy.

**Solutions:**

| Approach                    | Description                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| **Use BatchNorm only**      | Remove dropout entirely; rely on BN's implicit regularisation                               |
| **Use dropout only**        | Remove BatchNorm; add dropout between FC layers                                             |
| **Dropout after BatchNorm** | If using both, place dropout **after** BN so BN statistics are computed on the clean signal |
| **Lower dropout rate**      | Use $p = 0.1$–$0.2$ with BN instead of the standard $p = 0.5$                               |

> **Modern practice.** Most architectures (ResNet, EfficientNet) use BatchNorm but not dropout in convolutional layers. Dropout (if used) appears only in the final FC layers.

> **Notebook reference.** Exercise 1 asks you to test all four combinations: (BN off/on) × (Dropout off/on).

---

## 8. Early Stopping

Stop training when the validation metric stops improving:

```python
best_val_loss = float('inf')
patience_counter = 0
patience = 5

for epoch in range(max_epochs):
    train_loss = train_epoch(model, ...)
    val_loss = evaluate(model, ...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')   # save best
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model (not the last one)
model.load_state_dict(torch.load('best_model.pth'))
```

**Why it regularises.** Neural network training follows a characteristic trajectory:

1. **Early training:** model learns general patterns (low bias, low variance).
2. **Middle training:** model refines predictions (bias continues to drop).
3. **Late training:** model starts memorising noise (variance increases, val loss rises).

Early stopping halts at the optimal point on this curve — it implicitly constrains the effective complexity of the model.

**Formal view (Bishop, 2006).** For a linear model trained with gradient descent, early stopping is equivalent to L2 regularisation with $\lambda \propto 1/t$ where $t$ is the number of update steps. The model explores progressively complex functions as training continues; stopping early constrains it to simpler ones.

> **Notebook reference.** Exercise 4 asks you to implement early stopping with patience=5 and compare the stopped epoch to a full training run.

---

## 9. Ensemble Methods

### 9.1 Why Ensembles Regularise

Train $M$ models independently and average their predictions. Each model overfits differently (different random seed → different initialisation, different mini-batch order). Averaging cancels out the individual overfitting.

Formally, if each model has prediction variance $\sigma^2$ and the models are uncorrelated:

$$\text{Var}\!\left(\frac{1}{M}\sum_{m=1}^{M}f_m(x)\right) = \frac{\sigma^2}{M}$$

Variance drops as $1/M$. In practice, models are correlated, so the reduction is less than $1/M$ but still substantial.

---

### 9.2 Prediction Averaging

**For classification (logit averaging):**

$$\hat{y}_{\text{ensemble}} = \arg\max_c\frac{1}{M}\sum_{m=1}^{M}z_m^{(c)}$$

where $z_m^{(c)}$ is the $c$-th logit from model $m$.

```python
def ensemble_predict(models, x):
    with torch.no_grad():
        logits = torch.stack([m(x) for m in models])  # (M, N, C)
        avg_logits = logits.mean(dim=0)                 # (N, C)
    return avg_logits.argmax(dim=1)
```

**For regression:** simply average the predictions.

---

### 9.3 Dropout as Approximate Ensemble

Gal & Ghahramani (2016) showed that dropout at test time (called **MC Dropout**) approximates Bayesian inference:

```python
model.train()   # keep dropout active
predictions = torch.stack([model(x) for _ in range(T)])  # T forward passes

mean_pred = predictions.mean(dim=0)     # point estimate
uncertainty = predictions.std(dim=0)     # epistemic uncertainty
```

This connects to [Week 08](../../03_probability/week08_uncertainty/theory.md) (uncertainty): MC Dropout provides uncertainty estimates without training an ensemble.

> **Notebook reference.** Cell 4 trains 5 models with different seeds and averages their predictions, showing ~0.5–1% improvement over a single model.

---

## 10. Label Smoothing

Instead of hard targets $y \in \{0, 1\}^C$ (one-hot), use soft targets:

$$\tilde{y}_c = \begin{cases}1 - \epsilon + \epsilon/C & \text{if } c = y_{\text{true}}\\ \epsilon/C & \text{otherwise}\end{cases}$$

with smoothing parameter $\epsilon$ (typically 0.1).

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Why it helps:**

1. **Prevents overconfidence.** Without smoothing, the model is encouraged to push logits to $\pm\infty$ (to make softmax approach one-hot). This leads to poorly calibrated probabilities.
2. **Regularises.** The model cannot achieve zero loss (the target is not a delta function), which prevents memorisation.
3. **Improved calibration.** Probabilities better reflect true confidence ([Week 08](../../03_probability/week08_uncertainty/theory.md) connection).

---

## 11. Regularisation Interactions and Ablation Protocol

Regularisers interact — using all of them together is not necessarily optimal:

| Combination                 | Interaction                                                  | Recommendation                                            |
| --------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| Dropout + BatchNorm         | Statistical mismatch (Section 7)                             | Use BN; add dropout only in final FC with low $p$         |
| Weight decay + Adam         | L2 is partially undone by Adam's scaling                     | Use AdamW instead                                         |
| Weight decay + BatchNorm    | WD on BN's $\gamma, \beta$ can hurt; BN re-normalises anyway | Exclude BN params from weight decay                       |
| Augmentation + dropout      | Both inject noise; may be redundant                          | Augmentation first, add dropout only if still overfitting |
| Early stopping + all others | Always useful as a safety net                                | Always use early stopping                                 |

**Ablation protocol:**

1. **Baseline:** no regularisation (dropout=0, wd=0, no augmentation).
2. **One-at-a-time:** add each regulariser independently and measure val accuracy.
3. **Best combo:** combine the top performers and verify improvement.
4. **Interaction grid:** e.g., dropout $\in \{0, 0.3, 0.5\}$ × weight decay $\in \{0, 10^{-4}, 10^{-3}\}$ → 9 experiments.
5. **Report:** table of results with mean ± std across 3+ seeds.

```
| Dropout | Weight Decay | Val Acc (%) |
| ------- | ------------ | ----------- |
| 0.0     | 0            | 97.2 ± 0.1  |
| 0.0     | 1e-4         | 97.5 ± 0.2  |
| 0.3     | 0            | 97.8 ± 0.1  |
| 0.3     | 1e-4         | 98.1 ± 0.1  | ← best     |
| 0.5     | 1e-3         | 97.0 ± 0.3  | ← too much |
```

> **Notebook reference.** Exercise 3 asks for a 3×3 ablation grid of dropout × weight decay on MNIST.

---

## 12. Connections to the Rest of the Course

| Week                                     | Connection                                                                                                      |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **[Week 06](../../02_fundamentals/week06_regularization/theory.md) (Regularisation)**             | Ridge/Lasso for linear models; same $\lambda\|\theta\|^2$ penalty now applied to neural networks                |
| **[Week 08](../../03_probability/week08_uncertainty/theory.md) (Uncertainty)**                | MC Dropout for uncertainty; calibration improved by label smoothing                                             |
| **[Week 12](../../04_neural_networks/week12_training_pathologies/theory.md) (Training Pathologies)**       | BatchNorm's regularising side effect; interaction with dropout                                                  |
| **[Week 14](../week14_training_at_scale/theory.md) (Training at Scale)**          | Weight decay interacts with LR schedule ($\eta\lambda$ effective strength); augmentation in DataLoader pipeline |
| **[Week 15](../week15_cnn_representations/theory.md) (CNNs)**                       | The CNN built there is the model being regularised here; spatial dropout `Dropout2d`                            |
| **[[Week 17](../../06_sequence_models/week17_attention/theory.md)](../../06_sequence_models/week17_attention/theory.md)–[18](../../06_sequence_models/week18_transformers/theory.md) (Attention, Transformers)** | Dropout in attention layers and FFN; weight decay with AdamW is standard for Transformers                       |
| **[Week 19](../../07_transfer_learning/week19_finetuning/theory.md) (Fine-Tuning)**                | Fine-tuning is a form of regularisation: pretrained weights constrain the solution space                        |

---

## 13. Notebook Reference Guide

| Cell                   | Section                 | What it demonstrates                                                 | Theory reference |
| ---------------------- | ----------------------- | -------------------------------------------------------------------- | ---------------- |
| 1 (Dropout)            | Dropout comparison      | `MLPWithDropout(p=0)` vs `p=0.5`; loss and accuracy curves           | Section 3        |
| 2 (Augmentation)       | Data augmentation       | `RandomRotation`, `RandomAffine` on MNIST; compare to baseline       | Section 5        |
| 3 (Weight decay)       | WD sweep                | $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}\}$; accuracy vs. WD plot | Section 4        |
| 4 (Ensemble)           | Ensemble methods        | 5 models with different seeds; average logits; accuracy improvement  | Section 9        |
| Ex. 1 (BN + Dropout)   | Interaction             | 4 combos: (BN off/on) × (Dropout off/on)                             | Section 7        |
| Ex. 3 (Ablation grid)  | Ablation                | $3 \times 3$ grid: dropout × weight decay; pandas DataFrame          | Section 11       |
| Ex. 4 (Early stopping) | Implicit regularisation | Patience=5; compare stopped vs. full run                             | Section 8        |

**Suggested modifications:**

| Modification                                                                                                      | What it reveals                                                      |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Try Dropout2d (`nn.Dropout2d(0.25)`) after conv layers in [Week 15](../week15_cnn_representations/theory.md)'s CNN                                           | Spatial dropout drops entire feature maps; different from per-neuron |
| Sweep dropout rate $p \in \{0.1, 0.2, 0.3, 0.5, 0.7\}$ and plot                                                   | Shows optimal $p$; too high → underfitting                           |
| Compare `Adam(weight_decay=0.01)` vs. `AdamW(weight_decay=0.01)` on same model                                    | AdamW usually gives better generalisation — decoupled decay matters  |
| Add Mixup: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$                 | Powerful augmentation; requires soft cross-entropy                   |
| Train with label smoothing $\epsilon \in \{0, 0.05, 0.1, 0.2\}$                                                   | $\epsilon = 0.1$ is usually best; $\epsilon = 0.2$ can hurt          |
| Combine ALL regularisers (dropout=0.3, wd=1e-4, augmentation, BN, label smoothing) and compare to no-reg baseline | Shows cumulative benefit but also diminishing returns                |
| Implement MC Dropout: 50 forward passes at test time with `model.train()`                                         | Gives uncertainty estimates; compare to ensemble uncertainty         |
| Plot weight norm $\|\theta\|_2$ over training epochs with and without weight decay                                | WD keeps norms bounded; without WD, norms grow continuously          |

---

## 14. Symbol Reference

| Symbol            | Name                          | Meaning                                                   |
| ----------------- | ----------------------------- | --------------------------------------------------------- |
| $p$               | Dropout probability           | Fraction of neurons zeroed during training                |
| $m_j$             | Dropout mask                  | $m_j \sim \text{Bernoulli}(1 - p)$; 1 = keep, 0 = drop    |
| $1/(1-p)$         | Inverted dropout scale        | Maintains $\mathbb{E}[\tilde{a}] = a$                     |
| $\lambda$         | Weight decay / L2 coefficient | Strength of the penalty                                   |
| $\|\theta\|_2^2$  | Squared L2 norm               | $\sum_j \theta_j^2$ — sum of squared parameters           |
| $\epsilon$        | Label smoothing parameter     | Fraction of probability mass redistributed; typically 0.1 |
| $\tilde{y}_c$     | Smoothed label                | $(1 - \epsilon)\cdot\mathbf{1}[c = y] + \epsilon/C$       |
| $M$               | Number of ensemble models     | Typically 3–10                                            |
| $T$               | MC Dropout passes             | Number of stochastic forward passes                       |
| $\sigma^2$        | Prediction variance           | Reduced by $1/M$ in uncorrelated ensemble                 |
| $\text{patience}$ | Early stopping patience       | Epochs without improvement before stopping                |
| $\text{AdamW}$    | Decoupled weight decay Adam   | Weight decay applied to params, not gradients             |

---

## 15. References

1. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*. — The original dropout paper.
2. Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*. — AdamW; why L2 $\neq$ weight decay for Adam.
3. Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). "Understanding Deep Learning Requires Rethinking Generalization." *ICLR*. — Deep nets memorise random labels.
4. Ioffe, S. & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training." *ICML*. — BatchNorm's implicit regularisation.
5. Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*. — MC Dropout for uncertainty.
6. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). "Rethinking the Inception Architecture for Computer Vision." *CVPR*. — Label smoothing introduced.
7. Müller, R., Kornblith, S., & Hinton, G. (2019). "When Does Label Smoothing Help?" *NeurIPS*. — Analysis of label smoothing's effects.
8. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." *ICCV*. — CutMix augmentation.
9. Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). "mixup: Beyond Empirical Risk Minimization." *ICLR*. — Mixup augmentation.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 7: Regularization. MIT Press. — Comprehensive treatment of regularisation strategies.
