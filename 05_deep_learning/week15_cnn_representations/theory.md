# Convolutional Neural Networks and Representation Learning

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [From Fully Connected to Convolutional](#2-from-fully-connected-to-convolutional)
   - 2.1 [The Problem with MLPs on Images](#21-the-problem-with-mlps-on-images)
   - 2.2 [Inductive Biases of Convolution](#22-inductive-biases-of-convolution)
3. [The Convolution Operation](#3-the-convolution-operation)
   - 3.1 [1-D Convolution](#31-1-d-convolution)
   - 3.2 [2-D Convolution](#32-2-d-convolution)
   - 3.3 [Cross-Correlation vs. Convolution](#33-cross-correlation-vs-convolution)
   - 3.4 [Multi-Channel Convolution](#34-multi-channel-convolution)
   - 3.5 [Convolution as Matrix Multiplication](#35-convolution-as-matrix-multiplication)
4. [Padding, Stride, and Output Size](#4-padding-stride-and-output-size)
   - 4.1 [The Output-Size Formula](#41-the-output-size-formula)
   - 4.2 [Common Padding Strategies](#42-common-padding-strategies)
   - 4.3 [Stride](#43-stride)
   - 4.4 [Dilated (Atrous) Convolution](#44-dilated-atrous-convolution)
5. [Pooling](#5-pooling)
   - 5.1 [Max Pooling](#51-max-pooling)
   - 5.2 [Average Pooling](#52-average-pooling)
   - 5.3 [Global Average Pooling](#53-global-average-pooling)
6. [Receptive Fields](#6-receptive-fields)
7. [Feature Hierarchies](#7-feature-hierarchies)
8. [Building a CNN in PyTorch](#8-building-a-cnn-in-pytorch)
   - 8.1 [Layer-by-Layer Construction](#81-layer-by-layer-construction)
   - 8.2 [Parameter Counting](#82-parameter-counting)
   - 8.3 [BatchNorm2d in CNNs](#83-batchnorm2d-in-cnns)
9. [Visualising Filters and Activations](#9-visualising-filters-and-activations)
   - 9.1 [First-Layer Filters](#91-first-layer-filters)
   - 9.2 [Forward Hooks for Activations](#92-forward-hooks-for-activations)
10. [Transfer Learning and Pretrained Features](#10-transfer-learning-and-pretrained-features)
    - 10.1 [Why Transfer Learning Works](#101-why-transfer-learning-works)
    - 10.2 [Feature Extraction (Frozen Backbone)](#102-feature-extraction-frozen-backbone)
    - 10.3 [Fine-Tuning (Unfrozen Backbone)](#103-fine-tuning-unfrozen-backbone)
11. [Ablation Studies](#11-ablation-studies)
12. [Classic CNN Architectures](#12-classic-cnn-architectures)
13. [Connections to the Rest of the Course](#13-connections-to-the-rest-of-the-course)
14. [Notebook Reference Guide](#14-notebook-reference-guide)
15. [Symbol Reference](#15-symbol-reference)
16. [References](#16-references)

---

## 1. Scope and Purpose

This week introduces **convolutional neural networks** (CNNs) — the architecture that made deep learning practical for images, and whose principles (local connectivity, weight sharing, hierarchical features) recur in every modality.

The week delivers three things:

1. **The convolution operation** — what it computes, why it works for spatial data, and how it differs from a fully connected layer.
2. **Representation learning** — how filters evolve from edge detectors (layer 1) to object-part detectors (deeper layers), building a hierarchy of increasingly abstract features.
3. **Transfer learning preview** — extracting features from a pretrained model and reusing them, a pattern formalised in Week 19.

**Prerequisites.** Week 11 (backpropagation), Week 12 (BatchNorm), Week 13 (PyTorch `nn.Module`, training loop), Week 14 (efficient DataLoader, checkpointing).

---

## 2. From Fully Connected to Convolutional

### 2.1 The Problem with MLPs on Images

A 28×28 grayscale image has $d = 784$ pixels. A fully connected layer with 512 hidden units has:

$$784 \times 512 + 512 = 401{,}920 \text{ parameters}$$

A 224×224 RGB image has $d = 150{,}528$ pixels. The same FC layer would need:

$$150{,}528 \times 512 = 77{,}070{,}336 \text{ parameters}$$

The problems:
1. **Parameter explosion.** Too many weights to learn from limited data — massive overfitting.
2. **No spatial structure.** An MLP treats pixel $(0, 0)$ and pixel $(27, 27)$ as completely independent features. There is no notion of "nearby" or "local pattern."
3. **No translation invariance.** A cat in the top-left corner produces completely different activations from a cat in the bottom-right — the MLP must learn each position separately.

---

### 2.2 Inductive Biases of Convolution

A convolution layer encodes three assumptions about image data:

| Inductive bias               | Meaning                                                 | Implementation                                                          |
| ---------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Locality**                 | Important patterns are local (edges, textures)          | Small kernel (e.g., 3×3) looks at a neighbourhood, not the entire image |
| **Weight sharing**           | The same pattern can appear anywhere in the image       | One kernel is applied at every spatial position                         |
| **Translation equivariance** | Shifting the input shifts the output by the same amount | Consequence of weight sharing                                           |

These biases dramatically reduce parameters and encode the right prior for spatial data.

---

## 3. The Convolution Operation

### 3.1 1-D Convolution

For an input signal $x[n]$ and a kernel $w[k]$ of length $K$:

$$(x * w)[n] = \sum_{k=0}^{K-1}x[n + k]\cdot w[k]$$

The kernel slides along the signal, computing a dot product at each position.

**Example.** Input $x = [1, 2, 3, 4, 5]$, kernel $w = [1, 0, -1]$:

$$(x * w)[0] = 1 \cdot 1 + 2 \cdot 0 + 3 \cdot (-1) = -2$$
$$(x * w)[1] = 2 \cdot 1 + 3 \cdot 0 + 4 \cdot (-1) = -2$$
$$(x * w)[2] = 3 \cdot 1 + 4 \cdot 0 + 5 \cdot (-1) = -2$$

This kernel computes a finite difference — an approximation to the derivative. It's a (very simple) edge detector.

---

### 3.2 2-D Convolution

For an input image $X \in \mathbb{R}^{H \times W}$ and a kernel $W \in \mathbb{R}^{K_H \times K_W}$:

$$(X * W)[i, j] = \sum_{m=0}^{K_H-1}\sum_{n=0}^{K_W-1}X[i + m,\, j + n]\cdot W[m, n]$$

The kernel slides across both spatial dimensions, computing a dot product at each position.

**A 3×3 kernel applied to a 5×5 input (no padding, stride 1):**

```
Input (5×5)              Kernel (3×3)         Output (3×3)
┌─────────────┐          ┌───────┐
│ a b c d e   │          │ w₁ w₂ w₃ │
│ f g h i j   │    *     │ w₄ w₅ w₆ │   →   3×3 feature map
│ k l m n o   │          │ w₇ w₈ w₉ │
│ p q r s t   │          └───────┘
│ u v w x y   │
└─────────────┘
```

Each output pixel is a weighted sum of a 3×3 patch of the input, using the same 9 weights everywhere.

---

### 3.3 Cross-Correlation vs. Convolution

Strictly, mathematical convolution flips the kernel before sliding:

$$\text{(true convolution)} \quad (X * W)[i,j] = \sum_m\sum_n X[i - m, j - n]\cdot W[m, n]$$

In deep learning, we use **cross-correlation** (no flip) and call it "convolution." Since the kernel is learned, the distinction is irrelevant — the network learns the flipped version if needed.

> All neural network frameworks, including PyTorch's `nn.Conv2d`, implement cross-correlation.

---

### 3.4 Multi-Channel Convolution

Real inputs have $C_{\text{in}}$ channels (e.g., 3 for RGB). Each kernel is a 3-D tensor $W \in \mathbb{R}^{C_{\text{in}} \times K_H \times K_W}$, and the convolution sums across channels:

$$(X * W)[i, j] = \sum_{c=0}^{C_{\text{in}}-1}\sum_{m=0}^{K_H-1}\sum_{n=0}^{K_W-1}X[c, i+m, j+n]\cdot W[c, m, n] + b$$

One kernel produces one output channel (feature map). To produce $C_{\text{out}}$ feature maps, we need $C_{\text{out}}$ kernels:

$$\text{Weight tensor shape:}\quad C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W$$

```python
nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
# Weight shape: (64, 3, 3, 3) → 64 filters, each 3×3×3
```

---

### 3.5 Convolution as Matrix Multiplication

Convolution can be reformulated as matrix multiplication using the **im2col** trick:

1. Extract every $K \times K$ patch from the input and stack them as rows of a matrix $X_{\text{col}} \in \mathbb{R}^{(H'W') \times (C_{\text{in}}K^2)}$.
2. Reshape the $C_{\text{out}}$ kernels into a weight matrix $W_{\text{col}} \in \mathbb{R}^{(C_{\text{in}}K^2) \times C_{\text{out}}}$.
3. The output is $Y = X_{\text{col}} \cdot W_{\text{col}}$, reshaped to $(C_{\text{out}}, H', W')$.

This shows that convolution is a **constrained linear transformation**: the weight matrix of the equivalent FC layer would be huge, and most entries would be tied (weight sharing) or zero (locality).

$$\text{Conv: } 9 \text{ unique weights} \quad\text{vs.}\quad \text{FC: } 784 \times 784 = 614{,}656 \text{ weights}$$

---

## 4. Padding, Stride, and Output Size

### 4.1 The Output-Size Formula

For input size $H$, kernel size $K$, padding $P$, stride $S$, and dilation $D$:

$$\boxed{H_{\text{out}} = \left\lfloor\frac{H + 2P - D(K - 1) - 1}{S} + 1\right\rfloor}$$

The same formula applies independently to width.

---

### 4.2 Common Padding Strategies

| Strategy               | Padding $P$               | Output size (stride 1, no dilation) | Use case                             |
| ---------------------- | ------------------------- | ----------------------------------- | ------------------------------------ |
| **Valid** (no padding) | $P = 0$                   | $H - K + 1$                         | When spatial shrinkage is acceptable |
| **Same**               | $P = \lfloor K/2 \rfloor$ | $H$ (same as input)                 | Preserve spatial dimensions          |
| **Full**               | $P = K - 1$               | $H + K - 1$                         | Transposed convolution contexts      |

```python
nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 'same' for 3×3 kernel
nn.Conv2d(32, 64, kernel_size=5, padding=2)   # 'same' for 5×5 kernel
```

---

### 4.3 Stride

Stride $S > 1$ skips positions, downsampling the output:

$$H_{\text{out}} = \left\lfloor\frac{H + 2P - K}{S} + 1\right\rfloor$$

**Example.** Input $H = 28$, kernel $K = 3$, padding $P = 1$, stride $S = 2$:

$$H_{\text{out}} = \left\lfloor\frac{28 + 2 - 3}{2} + 1\right\rfloor = \left\lfloor14.5\right\rfloor = 14$$

Strided convolution acts as convolution + downsampling in a single operation — often preferred over separate pooling.

---

### 4.4 Dilated (Atrous) Convolution

Dilation $D > 1$ inserts gaps between kernel elements, expanding the receptive field without increasing parameters:

```
Standard (D=1):     Dilated (D=2):
■ ■ ■               ■ . ■ . ■
■ ■ ■               . . . . .
■ ■ ■               ■ . ■ . ■
                     . . . . .
                     ■ . ■ . ■

3×3 kernel           3×3 kernel, effective 5×5 receptive field
```

$$\text{Effective kernel size} = D(K - 1) + 1$$

Used in semantic segmentation (DeepLab) and WaveNet for audio.

---

## 5. Pooling

### 5.1 Max Pooling

Select the maximum value in each spatial window:

$$\text{MaxPool}_{k \times k}(X)[i, j] = \max_{m,n \in [0, k)} X[Si + m, Sj + n]$$

```python
nn.MaxPool2d(kernel_size=2, stride=2)  # halves spatial dimensions
```

**Effect.** Reduces spatial resolution by $2\times$ (with $k = s = 2$), retaining the strongest activation in each region. This introduces a small amount of **translation invariance**: a feature can shift by up to $k - 1$ pixels without affecting the pooled output.

---

### 5.2 Average Pooling

Average instead of max:

$$\text{AvgPool}_{k \times k}(X)[i, j] = \frac{1}{k^2}\sum_{m,n \in [0, k)} X[Si + m, Sj + n]$$

Less common in intermediate layers but used in Global Average Pooling.

---

### 5.3 Global Average Pooling

Average the entire spatial extent of each feature map into a single number:

$$\text{GAP}(X_c) = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}X_c[i, j]$$

For $C$ channels, this produces a vector of length $C$.

```python
nn.AdaptiveAvgPool2d(1)  # output: (batch, C, 1, 1) → squeeze to (batch, C)
```

**Why GAP?** It replaces the large fully connected layer that would otherwise flatten the feature maps. Comparison:

| Approach     | Parameters (64 channels, 7×7 spatial, 10 classes) |
| ------------ | ------------------------------------------------- |
| Flatten + FC | $(64 \times 7 \times 7) \times 10 = 31{,}360$     |
| GAP + FC     | $64 \times 10 = 640$                              |

GAP forces each channel to encode a single class-relevant feature, acting as a structural regulariser.

> **Notebook reference.** Exercise 1 asks you to build a 5-layer CNN with GAP.

---

## 6. Receptive Fields

The **receptive field** of a neuron at layer $l$ is the region of the original input that influences its value. For a stack of layers with kernel size $K_l$, stride $S_l$, and dilation $D_l$:

$$\boxed{R_l = R_{l-1} + (K_l - 1) \cdot D_l \cdot \prod_{k=1}^{l-1}S_k}$$

with $R_0 = 1$ (one pixel).

**Example: SimpleCNN in the notebook**

| Layer              | $K$ | $S$ | $D$ | Stride product (preceding) | $R$                            |
| ------------------ | --- | --- | --- | -------------------------- | ------------------------------ |
| Input              | —   | —   | —   | 1                          | 1                              |
| Conv1 (3×3, pad 1) | 3   | 1   | 1   | 1                          | $1 + 2 \times 1 \times 1 = 3$  |
| MaxPool (2×2)      | 2   | 2   | 1   | 1                          | $3 + 1 \times 1 \times 1 = 4$  |
| Conv2 (3×3, pad 1) | 3   | 1   | 1   | 2                          | $4 + 2 \times 1 \times 2 = 8$  |
| MaxPool (2×2)      | 2   | 2   | 1   | 2                          | $8 + 1 \times 1 \times 2 = 10$ |

A neuron in the final feature map of SimpleCNN "sees" a 10×10 patch of the 28×28 input.

> **Notebook reference.** Exercise 2 asks you to implement the receptive field formula and compute it for each layer.

---

## 7. Feature Hierarchies

As depth increases, CNN layers learn **increasingly abstract features**:

| Layer depth  | Features learned                 | Receptive field   | Example                                     |
| ------------ | -------------------------------- | ----------------- | ------------------------------------------- |
| Layer 1      | Edges, gradients, colours        | Small (3–5 px)    | Vertical edge, horizontal edge, colour blob |
| Layers 2–3   | Textures, corners, simple shapes | Medium (10–30 px) | Corner, grid, circle, stripe pattern        |
| Layers 4–6   | Object parts                     | Large (50–100 px) | Eye, wheel, window, fur texture             |
| Final layers | Whole objects, scene context     | Full image        | Face, car, building                         |

This hierarchy emerges automatically from training — it is not hand-designed. The compositionality comes from the network's depth: layer $l$ builds its features by combining features from layer $l - 1$.

**Why this is "representation learning."** Each layer transforms the input into a new representation that is progressively better for the task. The final representation (just before the classifier) is a compact, discriminative feature vector.

---

## 8. Building a CNN in PyTorch

### 8.1 Layer-by-Layer Construction

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # (1,28,28) → (32,28,28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (32,14,14) → (64,14,14)
        self.pool = nn.MaxPool2d(2, 2)                             # halves spatial dims
        self.fc1 = nn.Linear(64 * 7 * 7, 128)                     # after 2 pools: 28→14→7
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (B,1,28,28)→(B,32,14,14)
        x = self.pool(self.relu(self.conv2(x)))   # →(B,64,7,7)
        x = x.view(x.size(0), -1)                 # flatten: (B,64*7*7)
        x = self.relu(self.fc1(x))                 # (B,128)
        x = self.fc2(x)                            # (B,10)
        return x
```

**Dimension tracking.** The key discipline is computing output sizes at each layer:

| Layer                | Input shape       | Output shape      | Calculation                            |
| -------------------- | ----------------- | ----------------- | -------------------------------------- |
| Conv1 (3×3, pad 1)   | $(B, 1, 28, 28)$  | $(B, 32, 28, 28)$ | Same padding: $28 + 2(1) - 3 + 1 = 28$ |
| Pool (2×2, stride 2) | $(B, 32, 28, 28)$ | $(B, 32, 14, 14)$ | $\lfloor 28/2 \rfloor = 14$            |
| Conv2 (3×3, pad 1)   | $(B, 32, 14, 14)$ | $(B, 64, 14, 14)$ | Same padding                           |
| Pool (2×2, stride 2) | $(B, 64, 14, 14)$ | $(B, 64, 7, 7)$   | $\lfloor 14/2 \rfloor = 7$             |
| Flatten              | $(B, 64, 7, 7)$   | $(B, 3136)$       | $64 \times 7 \times 7$                 |
| FC1                  | $(B, 3136)$       | $(B, 128)$        |                                        |
| FC2                  | $(B, 128)$        | $(B, 10)$         |                                        |

> **Notebook reference.** Cell 1 builds this exact `SimpleCNN` and prints the total parameter count.

---

### 8.2 Parameter Counting

For `nn.Conv2d(C_in, C_out, K)`:

$$\text{Parameters} = C_{\text{out}} \times (C_{\text{in}} \times K^2 + 1)$$

The $+1$ is the bias (one per output channel).

| Layer               | Parameters                               |
| ------------------- | ---------------------------------------- |
| `Conv2d(1, 32, 3)`  | $32 \times (1 \times 9 + 1) = 320$       |
| `Conv2d(32, 64, 3)` | $64 \times (32 \times 9 + 1) = 18{,}496$ |
| `Linear(3136, 128)` | $3136 \times 128 + 128 = 401{,}536$      |
| `Linear(128, 10)`   | $128 \times 10 + 10 = 1{,}290$           |
| **Total**           | **421,642**                              |

> Note that the FC layer (`fc1`) dominates — 95% of parameters. This is the motivation for Global Average Pooling.

---

### 8.3 BatchNorm2d in CNNs

In a CNN, BatchNorm normalises per-channel, computing mean and variance across the batch and spatial dimensions:

$$\mu_c = \frac{1}{m \cdot H \cdot W}\sum_{i,j,k}X_{i,c,j,k}, \qquad \sigma_c^2 = \frac{1}{m \cdot H \cdot W}\sum_{i,j,k}(X_{i,c,j,k} - \mu_c)^2$$

```python
nn.BatchNorm2d(num_features=64)  # 64 channels → 64 means, 64 variances
```

Learnable parameters: $\gamma_c, \beta_c$ for each of the $C$ channels → $2C$ parameters.

> A typical modern CNN block: `Conv2d → BatchNorm2d → ReLU`.

---

## 9. Visualising Filters and Activations

### 9.1 First-Layer Filters

The first convolutional layer operates directly on pixels. Its $K \times K$ filters are interpretable as image patches:

```python
filters = model.conv1.weight.data.cpu().numpy()   # shape: (C_out, C_in, K, K)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < filters.shape[0]:
        ax.imshow(filters[i, 0], cmap='gray')
        ax.axis('off')
```

Typical first-layer filters on natural images: horizontal edges, vertical edges, diagonal edges, colour blobs. On MNIST: stroke detectors at various orientations.

> **Deeper layers** are harder to visualise directly because each filter has $C_{\text{in}}$ channels. Instead, we visualise their **activations** (output feature maps).

---

### 9.2 Forward Hooks for Activations

PyTorch **hooks** let you capture intermediate outputs without modifying the model:

```python
activations = {}

def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))

# Run a forward pass
_ = model(sample_image.unsqueeze(0))

# activations['conv1'] has shape (1, 32, 14, 14)
# activations['conv2'] has shape (1, 64, 7, 7)
```

Visualising `conv1` activations shows which spatial regions activate each filter — bright regions correspond to the pattern the filter detects.

> **Notebook reference.** Cell 3 visualises both first-layer filters and activations for a test digit.

---

## 10. Transfer Learning and Pretrained Features

### 10.1 Why Transfer Learning Works

A CNN trained on ImageNet (1.2M images, 1000 classes) learns a general visual feature hierarchy:
- **Early layers:** edges and textures (universal across visual domains).
- **Middle layers:** shapes and patterns (broadly transferable).
- **Late layers:** object-specific features (task-specific).

The early and middle layers are useful for almost any image task. **Transfer learning** reuses these features instead of learning them from scratch.

**When to transfer:**

| Scenario                        | Your dataset                     | Strategy                                      |
| ------------------------------- | -------------------------------- | --------------------------------------------- |
| Small dataset, similar domain   | 1K–10K images, natural images    | Freeze backbone, train new head               |
| Small dataset, different domain | 1K–10K images, medical/satellite | Fine-tune last few layers                     |
| Large dataset, similar domain   | 100K+ images                     | Fine-tune everything with small LR            |
| Large dataset, different domain | 100K+ images                     | Train from scratch (or fine-tune with warmup) |

---

### 10.2 Feature Extraction (Frozen Backbone)

Freeze all pretrained layers and train only the final classifier:

```python
# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer (only this is trainable)
num_features = model.fc.in_features   # 512 for ResNet18
model.fc = nn.Linear(num_features, num_classes)

# Only model.fc.parameters() have requires_grad=True
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable} / {total}")  # 5,130 / 11,181,642 for 10 classes
```

**Advantages:**
- Extremely fast to train (only the classifier head).
- Works well even with very small datasets (100s of images).
- No risk of catastrophically forgetting the pretrained features.

> **Notebook reference.** Cell 4 loads ResNet18 with ImageNet weights, freezes all layers, and replaces the final FC for MNIST classification.

---

### 10.3 Fine-Tuning (Unfrozen Backbone)

For better performance, unfreeze some or all backbone layers and train with a small learning rate:

```python
# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

# Use discriminative learning rates (lower for backbone, higher for head)
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
], lr=1e-5)  # default for everything else
```

**Discriminative learning rates.** Early layers (general features) need minimal adaptation → small LR. Later layers (task-specific) need more → larger LR. The final head (random init) needs the most → largest LR.

> Fine-tuning is covered in depth in Week 19.

---

## 11. Ablation Studies

An **ablation study** measures the contribution of individual components by removing or varying them one at a time. This is the experimental discipline CNN week introduces.

**Ablation variables for CNNs:**

| Variable           | Values to try                        | What it measures                   |
| ------------------ | ------------------------------------ | ---------------------------------- |
| Depth              | 2, 3, 5, 8 conv layers               | Benefit of deeper representations  |
| Width              | 16, 32, 64, 128 channels             | Capacity at each layer             |
| Kernel size        | 3×3 vs. 5×5 vs. 7×7                  | Size of local patterns captured    |
| Pooling type       | MaxPool vs. AvgPool vs. Strided conv | Effect of downsampling method      |
| BatchNorm          | With vs. without                     | Stabilisation benefit              |
| GAP vs. Flatten+FC | Compare param count and accuracy     | Structural regularisation from GAP |

**Protocol:**
1. Fix a baseline configuration.
2. Change exactly one variable at a time.
3. Train with identical hyperparameters (LR, epochs, optimizer).
4. Report mean and standard deviation across 3+ seeds.
5. Present results in a table or bar chart.

> **Notebook reference.** Exercise 1 asks for a deeper CNN (5 layers) comparison. The exercises section lists depth/width ablation as a deliverable.

---

## 12. Classic CNN Architectures

| Architecture              | Year | Key innovation                                            | Depth    |
| ------------------------- | ---- | --------------------------------------------------------- | -------- |
| **LeNet-5**               | 1998 | First practical CNN (handwriting)                         | 5        |
| **AlexNet**               | 2012 | ReLU, dropout, GPU training; won ImageNet                 | 8        |
| **VGGNet**                | 2014 | Uniform 3×3 convolutions, very deep                       | 16–19    |
| **GoogLeNet / Inception** | 2014 | Inception modules (parallel 1×1, 3×3, 5×5 convs)          | 22       |
| **ResNet**                | 2015 | Residual (skip) connections (Week 12)                     | 50–152   |
| **DenseNet**              | 2016 | Dense connections (each layer connects to all subsequent) | 121–201  |
| **EfficientNet**          | 2019 | Compound scaling (depth × width × resolution)             | variable |

**The VGG insight.** Two stacked 3×3 convolutions have the same receptive field as one 5×5 convolution, but with fewer parameters and more nonlinearity:

$$\text{Two 3×3:}\quad 2 \times (C \times C \times 9) = 18C^2 \qquad\text{One 5×5:}\quad C \times C \times 25 = 25C^2$$

This is why modern CNNs almost exclusively use 3×3 kernels.

**The ResNet insight.** Skip connections (Week 12) enable training 100+ layer networks by providing gradient shortcuts through the identity path.

---

## 13. Connections to the Rest of the Course

| Week                               | Connection                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| **Week 00a (AI Landscape)**        | Inductive bias — CNNs encode locality and translation equivariance                           |
| **Week 11 (NN from Scratch)**      | Convolution is backpropagated the same way: chain rule through the conv operation            |
| **Week 12 (Training Pathologies)** | BatchNorm2d, residual connections, He init — all used in CNNs                                |
| **Week 13 (PyTorch)**              | `nn.Conv2d`, `nn.MaxPool2d` — same `nn.Module` API                                           |
| **Week 14 (Training at Scale)**    | Multi-worker DataLoader essential for image datasets; mixed precision for conv-heavy models  |
| **Week 16 (Regularisation DL)**    | Dropout after conv layers; data augmentation in `transforms`; weight decay                   |
| **Week 17 (Attention)**            | Self-attention can be seen as a data-dependent convolution with a global receptive field     |
| **Week 19 (Fine-Tuning)**          | Feature extraction and fine-tuning patterns from this week formalised with adapters and LoRA |

---

## 14. Notebook Reference Guide

| Cell                    | Section               | What it demonstrates                                             | Theory reference   |
| ----------------------- | --------------------- | ---------------------------------------------------------------- | ------------------ |
| 1 (SimpleCNN)           | Model building        | 2-conv + 2-pool + 2-FC; parameter count                          | Section 8          |
| 2 (Training)            | Training loop         | MNIST training (5 epochs, Adam); test accuracy                   | Week 13, Section 8 |
| 3 (Visualisation)       | Filters & activations | `conv1` filters as 3×3 images; activation maps via forward hooks | Section 9          |
| 4 (Transfer learning)   | Feature extraction    | ResNet18 frozen backbone + new FC head; trainable param count    | Section 10         |
| Ex. 1 (Deeper CNN)      | Architecture          | 5-layer CNN with BatchNorm and GAP; compare to SimpleCNN         | Sections 5.3, 8.3  |
| Ex. 2 (Receptive field) | Theory                | Implement RF formula; compute per layer                          | Section 6          |
| Ex. 4 (Frozen ResNet)   | Transfer learning     | Train only FC on MNIST (3 epochs); compare to from-scratch       | Section 10.2       |

**Suggested modifications:**

| Modification                                                         | What it reveals                                                           |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Replace MaxPool with stride-2 convolutions                           | Strided convs are learnable downsampling; often slightly better accuracy  |
| Add BatchNorm2d after each conv and retrain                          | Faster convergence, often +1–2% accuracy                                  |
| Visualise `conv2` activations (64 channels, 7×7 each)                | Later layers respond to more abstract patterns (curves, loops for MNIST)  |
| Try CIFAR-10 instead of MNIST (3 channels, 32×32)                    | Harder task; adjust `Conv2d(3, 32, ...)` for 3-channel input              |
| Remove all pooling layers, use stride-2 convs everywhere             | Pure convolutional downsampling; compare to pooling-based                 |
| Train ResNet18 from scratch on MNIST vs. pretrained+fine-tune        | Shows the massive benefit of pretrained features even on a "easy" dataset |
| Visualise the gradient of the loss w.r.t. the input image (`x.grad`) | Saliency map: which pixels the model cares about most                     |
| Plot training accuracy vs. number of conv layers (2, 3, 4, 5)        | Diminishing returns of depth on simple datasets like MNIST                |

---

## 15. Symbol Reference

| Symbol                          | Name                            | Meaning                                       |
| ------------------------------- | ------------------------------- | --------------------------------------------- |
| $H, W$                          | Height, width                   | Spatial dimensions of input or feature map    |
| $C_{\text{in}}, C_{\text{out}}$ | Input/output channels           | Number of feature maps                        |
| $K$ ($K_H, K_W$)                | Kernel size                     | Spatial extent of the filter                  |
| $P$                             | Padding                         | Zeros added around the border                 |
| $S$                             | Stride                          | Step size of the sliding kernel               |
| $D$                             | Dilation                        | Gap between kernel elements                   |
| $R_l$                           | Receptive field at layer $l$    | Input pixels that influence one output neuron |
| $B$                             | Batch size                      | Samples per mini-batch                        |
| $*$                             | Convolution (cross-correlation) | Sliding dot product                           |
| $\text{MaxPool}$                | Max pooling                     | Take maximum in each window                   |
| $\text{GAP}$                    | Global average pooling          | Average entire spatial extent per channel     |
| $b$                             | Bias                            | One per output channel in a conv layer        |
| $\gamma, \beta$                 | BatchNorm2d parameters          | Per-channel scale and shift (Week 12)         |
| `state_dict`                    | Model state                     | Dictionary of all parameter tensors           |
| `register_forward_hook`         | Hook API                        | Capture intermediate activations              |

---

## 16. References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*. — LeNet-5; the foundational CNN paper.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*. — AlexNet; CNN revolution on ImageNet.
3. Simonyan, K. & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*. — VGGNet; 3×3 filter stacking.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*. — ResNet; skip connections.
5. Lin, M., Chen, Q., & Yan, S. (2014). "Network in Network." *ICLR*. — Global Average Pooling as a replacement for FC layers.
6. Zeiler, M. D. & Fergus, R. (2014). "Visualizing and Understanding Convolutional Networks." *ECCV*. — Deconvolution-based filter visualisation.
7. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). "How Transferable Are Features in Deep Neural Networks?" *NeurIPS*. — Empirical study of layer-wise transferability.
8. Tan, M. & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*. — Compound scaling.
9. Stanford CS231n. *Convolutional Neural Networks for Visual Recognition.* https://cs231n.github.io/ — Reference notes on convolution, pooling, and architectures.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 9. MIT Press. — Theoretical treatment of convolution, pooling, and equivariance.
