# Attention Mechanisms

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Sequential Data and the Need for Attention](#2-sequential-data-and-the-need-for-attention)
    - 2.1 [The Problem with Fixed-Length Representations](#21-the-problem-with-fixed-length-representations)
    - 2.2 [RNNs in Brief](#22-rnns-in-brief)
    - 2.3 [The Bottleneck Problem](#23-the-bottleneck-problem)
3. [Attention as Soft Lookup](#3-attention-as-soft-lookup)
    - 3.1 [The Query-Key-Value Framework](#31-the-query-key-value-framework)
    - 3.2 [Hard vs. Soft Attention](#32-hard-vs-soft-attention)
4. [Scaled Dot-Product Attention](#4-scaled-dot-product-attention)
    - 4.1 [The Formula](#41-the-formula)
    - 4.2 [Why Scale by $\sqrt{d_k}$?](#42-why-scale-by-sqrtd_k)
    - 4.3 [Step-by-Step Walkthrough](#43-step-by-step-walkthrough)
    - 4.4 [Masking](#44-masking)
5. [Scoring Functions](#5-scoring-functions)
6. [Multi-Head Attention](#6-multi-head-attention)
    - 6.1 [The Idea](#61-the-idea)
    - 6.2 [The Algorithm](#62-the-algorithm)
    - 6.3 [Parameter Count](#63-parameter-count)
    - 6.4 [Why Multiple Heads?](#64-why-multiple-heads)
7. [Self-Attention vs. Cross-Attention](#7-self-attention-vs-cross-attention)
8. [Attention in Sequence-to-Sequence Models](#8-attention-in-sequence-to-sequence-models)
    - 8.1 [Encoder–Decoder Architecture](#81-encoderdecoder-architecture)
    - 8.2 [Additive (Bahdanau) Attention](#82-additive-bahdanau-attention)
9. [Visualising Attention Weights](#9-visualising-attention-weights)
10. [Computational Complexity of Attention](#10-computational-complexity-of-attention)
11. [From Attention to Transformers (Preview)](#11-from-attention-to-transformers-preview)
12. [Connections to the Rest of the Course](#12-connections-to-the-rest-of-the-course)
13. [Notebook Reference Guide](#13-notebook-reference-guide)
14. [Symbol Reference](#14-symbol-reference)
15. [References](#15-references)

---

## 1. Scope and Purpose

This week introduces **attention mechanisms** — the core operation that replaced recurrence in modern sequence models and enabled Transformers. Implementing attention from scratch this week means that [Week 18](../week18_transformers/theory.md#1-scope-and-purpose) (Transformers) is assembling known components, not absorbing an alien architecture.

The week delivers:

1. **Scaled dot-product attention** — the fundamental operation: queries, keys, values, softmax.
2. **Multi-head attention** — running several attention operations in parallel to capture different relationship types.
3. **Self-attention vs. cross-attention** — same mechanism, different sources of Q, K, V.
4. **A working seq2seq model with attention** — trained on a toy task, with attention weights you can visualise.

**Prerequisites.** [Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md#4-building-models-with-nnmodule) (PyTorch `nn.Module`, training loop). The concept of "sequential data" from [Week 00a](../../01_intro/week00_ai_landscape/theory.md).

---

## 2. Sequential Data and the Need for Attention

### 2.1 The Problem with Fixed-Length Representations

A sequence of $T$ tokens $x_1, x_2, \ldots, x_T$ (a sentence, a time series, a DNA strand) has **variable length** and **order matters**. A fully connected network requires a fixed-size input — it cannot natively handle variable $T$.

---

### 2.2 RNNs in Brief

Recurrent neural networks process sequences one step at a time, maintaining a hidden state:

$$h_t = f(h_{t-1}, x_t;\; \theta)$$

The hidden state $h_t$ is a fixed-size vector that summarises all information from $x_1, \ldots, x_t$. Common variants:

| Architecture    | Key idea                                               | Advantage                                    |
| --------------- | ------------------------------------------------------ | -------------------------------------------- |
| **Vanilla RNN** | $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$               | Simple                                       |
| **LSTM**        | Gates (forget, input, output) control information flow | Handles longer dependencies                  |
| **GRU**         | Simplified gating (reset, update)                      | Lighter than LSTM, often similar performance |

**Critical limitation.** RNNs are sequential by construction — $h_t$ depends on $h_{t-1}$. This prevents parallelisation and, despite gating, gradients still degrade over long sequences (connections to [Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#3-vanishing-gradients): vanishing gradients).

---

### 2.3 The Bottleneck Problem

In an encoder–decoder model without attention, the encoder compresses the entire source sequence into a single vector $h_T$ (the final hidden state). The decoder must generate the full output from this one vector.

$$\underbrace{x_1, x_2, \ldots, x_T}_{\text{source sequence}} \xrightarrow{\text{encoder}} \underbrace{h_T}_{\text{single vector}} \xrightarrow{\text{decoder}} \underbrace{y_1, y_2, \ldots, y_{T'}}_{\text{target sequence}}$$

For long sequences, $h_T$ cannot retain all the information — early tokens are "forgotten." This is the **information bottleneck**.

> **Attention solves this:** instead of compressing into one vector, the decoder can look back at all encoder states $h_1, \ldots, h_T$ and dynamically select which ones are relevant for each output token.

---

## 3. Attention as Soft Lookup

### 3.1 The Query-Key-Value Framework

Attention is a **differentiable lookup table**:

| Component       | Analogy                 | Role                                                          |
| --------------- | ----------------------- | ------------------------------------------------------------- |
| **Query** ($Q$) | What you're looking for | The current decoder state (or current token's representation) |
| **Key** ($K$)   | Index of each entry     | Representation of each source position                        |
| **Value** ($V$) | Content of each entry   | Information to retrieve from each source position             |

The attention mechanism computes a weighted average of values, where the weights are determined by how well each key matches the query:

$$\text{Attention}(q, K, V) = \sum_{i=1}^{T}\alpha_i\, v_i, \qquad \alpha_i = \frac{\text{score}(q, k_i)}{\sum_j\text{score}(q, k_j)}$$

The weights $\alpha_i$ are non-negative and sum to 1 (softmax normalisation).

---

### 3.2 Hard vs. Soft Attention

| Type               | Selection                     | Differentiable?           | Example                            |
| ------------------ | ----------------------------- | ------------------------- | ---------------------------------- |
| **Hard attention** | Pick one $k_i$ (argmax)       | No (requires REINFORCE)   | Image captioning (Xu et al., 2015) |
| **Soft attention** | Weighted average of all $v_i$ | Yes (end-to-end backprop) | Transformers, all modern models    |

Soft attention is used almost universally because it is differentiable and trainable with standard backpropagation.

---

## 4. Scaled Dot-Product Attention

### 4.1 The Formula

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}$$

where:
- $Q \in \mathbb{R}^{n \times d_k}$ — queries ($n$ query vectors, each of dimension $d_k$)
- $K \in \mathbb{R}^{m \times d_k}$ — keys ($m$ key vectors)
- $V \in \mathbb{R}^{m \times d_v}$ — values ($m$ value vectors, dimension $d_v$)
- The output is $\in \mathbb{R}^{n \times d_v}$

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)   # (B, n, m)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)                              # (B, n, m)
    output = torch.matmul(weights, V)                                # (B, n, d_v)
    return output, weights
```

---

### 4.2 Why Scale by $\sqrt{d_k}$?

The dot product $q^\top k$ has expected value 0 and variance $d_k$ when $q$ and $k$ are i.i.d. with zero mean and unit variance:

$$\text{Var}(q^\top k) = \text{Var}\!\left(\sum_{i=1}^{d_k}q_i k_i\right) = \sum_{i=1}^{d_k}\text{Var}(q_i)\text{Var}(k_i) = d_k$$

For large $d_k$, the scores have large magnitudes, pushing softmax into saturation (outputting near-zero or near-one). In saturation, gradients vanish (same problem as sigmoid, [Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#51-sigmoid-and-tanh-saturation)).

Dividing by $\sqrt{d_k}$ normalises the variance to 1:

$$\text{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

This keeps softmax in its sensitive region, where gradients flow.

---

### 4.3 Step-by-Step Walkthrough

For a single batch, $n = 3$ queries, $m = 4$ keys, $d_k = d_v = 2$:

$$Q = \begin{pmatrix}1 & 0\\0 & 1\\1 & 1\end{pmatrix}, \quad K = \begin{pmatrix}1 & 0\\0 & 1\\1 & 1\\0 & 0\end{pmatrix}, \quad V = \begin{pmatrix}1 & 2\\3 & 4\\5 & 6\\7 & 8\end{pmatrix}$$

**Step 1.** Scores $= QK^\top / \sqrt{2}$:

$$QK^\top = \begin{pmatrix}1 & 0 & 1 & 0\\0 & 1 & 1 & 0\\1 & 1 & 2 & 0\end{pmatrix} \xrightarrow{\div\sqrt{2}} \begin{pmatrix}0.71 & 0 & 0.71 & 0\\0 & 0.71 & 0.71 & 0\\0.71 & 0.71 & 1.41 & 0\end{pmatrix}$$

**Step 2.** Softmax (row-wise):

Row 0: $[0.28, 0.14, 0.28, 0.14] \to$ normalised to $[0.31, 0.17, 0.31, 0.20]$ (approximate)

**Step 3.** Output $= \text{softmax}(\cdot) \times V$: a weighted average of the value vectors for each query.

The key insight: query $q_0 = [1, 0]$ has high scores for $k_0 = [1, 0]$ and $k_2 = [1, 1]$ (both have a 1 in the first dimension) — it "attends" to those positions.

---

### 4.4 Masking

Two types of masks:

**1. Padding mask.** Sequences in a batch have different lengths. Shorter sequences are padded with zeros. The mask prevents attention from attending to padding positions:

```python
# mask shape: (B, 1, m) — broadcast over query dimension
mask = (key_tokens != PAD_TOKEN).unsqueeze(1)   # True for real tokens
scores = scores.masked_fill(mask == 0, float('-inf'))
# softmax(-inf) = 0 → padding positions contribute nothing
```

**2. Causal (autoregressive) mask.** In language generation, token $t$ should only attend to tokens $1, \ldots, t$ (not future tokens):

```python
causal_mask = torch.tril(torch.ones(n, n))   # lower-triangular matrix
# Position i can attend to positions 0..i
```

$$\text{Causal mask} = \begin{pmatrix}1 & 0 & 0 & 0\\1 & 1 & 0 & 0\\1 & 1 & 1 & 0\\1 & 1 & 1 & 1\end{pmatrix}$$

> **Notebook reference.** Exercise 1 (causal mask) asks you to implement a causal mask and verify the upper triangle of the attention weights is zero.

---

## 5. Scoring Functions

Different ways to compute the compatibility between a query $q$ and a key $k$:

| Name                       | Formula                                           | Complexity | Used in                             |
| -------------------------- | ------------------------------------------------- | ---------- | ----------------------------------- |
| **Dot product**            | $\text{score}(q, k) = q^\top k$                   | $O(d)$     | Transformers (with scaling)         |
| **Scaled dot product**     | $\text{score}(q, k) = q^\top k / \sqrt{d_k}$      | $O(d)$     | Transformers (Vaswani et al., 2017) |
| **Additive (Bahdanau)**    | $\text{score}(q, k) = v^\top\tanh(W_q q + W_k k)$ | $O(d)$     | Original seq2seq attention          |
| **Multiplicative (Luong)** | $\text{score}(q, k) = q^\top W k$                 | $O(d^2)$   | Luong et al., 2015                  |

**Dot-product attention is the modern default** because it can be computed as a single matrix multiplication ($QK^\top$), which GPUs parallelise efficiently.

---

## 6. Multi-Head Attention

### 6.1 The Idea

A single attention operation computes one set of attention weights. But a token might relate to different tokens for different reasons:
- **Syntactic:** "The cat that **sat** on the mat" — verb relates to subject.
- **Semantic:** "The **cat** that sat on the **mat**" — noun relates to location.
- **Positional:** nearby tokens relate to each other.

**Multi-head attention** runs $h$ independent attention operations in parallel, each in a different learned subspace, then concatenates the results.

---

### 6.2 The Algorithm

$$\boxed{\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\cdot W^O}$$

where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

with projection matrices $W_i^Q \in \mathbb{R}^{d_\text{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_\text{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_\text{model} \times d_v}$, and $W^O \in \mathbb{R}^{hd_v \times d_\text{model}}$.

Typically $d_k = d_v = d_\text{model} / h$, so the total computation is similar to single-head attention with full $d_\text{model}$.

**Implementation detail — head splitting:** rather than creating $h$ separate weight matrices, we project into the full $d_\text{model}$, then reshape:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)   # projects to all heads at once
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # (B, h, T, d_k)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (B, h, T, d_k) → (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(attn_out), weights
```

---

### 6.3 Parameter Count

For `MultiHeadAttention(d_model, h)`:

$$\text{Parameters} = \underbrace{4 \times d_\text{model}^2}_{W_Q, W_K, W_V, W_O} + \underbrace{4 \times d_\text{model}}_{\text{biases}}$$

For $d_\text{model} = 64$, $h = 8$:

$$4 \times 64^2 + 4 \times 64 = 16{,}384 + 256 = 16{,}640$$

> **Notebook reference.** Cell 3 builds `MultiHeadAttention(64, 8)` and prints the parameter count.

---

### 6.4 Why Multiple Heads?

1. **Different subspaces capture different relationships.** Head 1 might learn positional proximity; head 3 might learn syntactic dependency.
2. **Parallelisation.** All heads compute simultaneously — no sequential bottleneck.
3. **Richer gradients.** Each head provides an independent gradient signal for the Q/K/V projections.

Empirically, 8 heads is the standard for $d_\text{model} = 512$ (Transformer base). More heads with smaller $d_k$ often work as well as fewer heads with larger $d_k$, but there's a diminishing return.

---

## 7. Self-Attention vs. Cross-Attention

| Type                | Q source         | K, V source      | Use case                                                     |
| ------------------- | ---------------- | ---------------- | ------------------------------------------------------------ |
| **Self-attention**  | Same sequence    | Same sequence    | Each position attends to all others within the same sequence |
| **Cross-attention** | Decoder sequence | Encoder sequence | Decoder attends to encoder outputs                           |

**Self-attention** (Q = K = V = same input):

```python
output, weights = mha(x, x, x)   # all from the same sequence
```

This is the attention used in each encoder layer and (with causal mask) in each decoder layer of a Transformer.

**Cross-attention** (Q from decoder, K and V from encoder):

```python
output, weights = mha(decoder_hidden, encoder_output, encoder_output)
```

This is how the decoder "looks at" the encoder — queries come from the decoder's current state, keys and values come from the encoder.

> **Notebook reference.** Exercise 2 asks you to build a cross-attention layer with different-length Q and K/V sequences.

---

## 8. Attention in Sequence-to-Sequence Models

### 8.1 Encoder–Decoder Architecture

The notebook builds a GRU-based seq2seq model with additive attention for a toy task (reverse a sequence):

```
Source: [3, 7, 1, 5]  →  Encoder  →  hidden states h₁, h₂, h₃, h₄
                                                ↓ attention
Target: [5, 1, 7, 3]  ←  Decoder  ←  context vectors at each step
```

At each decoder step $t$:

1. **Compute attention weights** $\alpha_{t,i}$ using the current decoder hidden state $s_t$ and all encoder states $h_i$.
2. **Compute context vector** $c_t = \sum_i \alpha_{t,i} h_i$.
3. **Feed** $[y_{t-1}; c_t]$ into the decoder GRU and predict $y_t$.

---

### 8.2 Additive (Bahdanau) Attention

The original attention mechanism (Bahdanau et al., 2015):

$$\text{score}(s_t, h_i) = v^\top\tanh(W_s s_t + W_h h_i)$$

$$\alpha_{t,i} = \frac{\exp(\text{score}(s_t, h_i))}{\sum_j\exp(\text{score}(s_t, h_j))}$$

$$c_t = \sum_{i=1}^{T}\alpha_{t,i}\, h_i$$

The notebook simplifies this slightly by concatenating $[s_t; h_i]$ and passing through a linear layer:

```python
self.attention = nn.Linear(hidden_dim * 2, 1)
attn_input = torch.cat([hidden_expanded, encoder_outputs], dim=-1)
attn_scores = self.attention(attn_input).squeeze(-1)
```

> **Notebook reference.** Cell 4 builds `Seq2SeqWithAttention` (GRU encoder + additive attention + GRU decoder) and trains it on the sequence-reversal task for 5 epochs.

---

## 9. Visualising Attention Weights

Attention weights $\alpha$ form a matrix of shape $(n, m)$ — one weight per (query, key) pair. Visualising this matrix as a heatmap reveals what the model "looks at":

```python
sns.heatmap(attn_weights.detach().numpy(), cmap='viridis', annot=True)
plt.xlabel('Key position'); plt.ylabel('Query position')
```

**What to look for:**

| Pattern                 | Meaning                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| **Diagonal**            | Each position attends to itself (identity)                                               |
| **Off-diagonal stripe** | Positions attend to a fixed offset (positional pattern)                                  |
| **Sparse columns**      | All queries focus on a few key positions (information bottleneck)                        |
| **Uniform**             | All weights equal — attention isn't selective (possibly untrained or too many positions) |
| **Anti-diagonal**       | For the reverse task: position $i$ attends to position $T - i$                           |

> **Notebook reference.** Cell 2 visualises attention for a 6-word sentence. Exercise 3 asks for per-head heatmaps in multi-head attention.

---

## 10. Computational Complexity of Attention

For sequence length $T$ and dimension $d$:

| Operation                 | FLOPS      | Memory                        |
| ------------------------- | ---------- | ----------------------------- |
| $QK^\top$                 | $O(T^2 d)$ | $O(T^2)$ for the score matrix |
| Softmax                   | $O(T^2)$   | $O(T^2)$                      |
| $\text{softmax} \times V$ | $O(T^2 d)$ | $O(Td)$ for the output        |
| **Total**                 | $O(T^2 d)$ | $O(T^2 + Td)$                 |

The **quadratic** scaling in $T$ is the main limitation. For $T = 1{,}024$, the attention matrix has $\sim 10^6$ entries. For $T = 100{,}000$, it has $\sim 10^{10}$ — infeasible.

**Comparison:**

| Model              | Time per layer                | Parallelisable?      |
| ------------------ | ----------------------------- | -------------------- |
| RNN                | $O(Td^2)$ — linear in $T$     | No (sequential)      |
| Self-attention     | $O(T^2 d)$ — quadratic in $T$ | Yes (fully parallel) |
| CNN ($k \times k$) | $O(T k d^2)$ — linear in $T$  | Yes                  |

For typical NLP ($T \leq 512$, $d = 512$): $T^2 d = 512^2 \times 512 \approx 10^8$, while $Td^2 = 512 \times 512^2 \approx 10^8$ — comparable. For long sequences, linear attention variants (Performer, Linformer) trade expressivity for $O(Td)$ complexity.

---

## 11. From Attention to Transformers (Preview)

A Transformer ([Week 18](../week18_transformers/theory.md#2-the-transformer-at-a-glance)) stacks attention with two more components:

$$\text{Transformer Layer} = \text{LayerNorm}(\text{Self-Attention}(x) + x) \xrightarrow{\text{residual}} \text{LayerNorm}(\text{FFN}(\cdot) + \cdot)$$

Components from this week:
- **Scaled dot-product attention** (Section 4) — the core operation.
- **Multi-head attention** (Section 6) — used for self-attention and cross-attention.
- **Masking** (Section 4.4) — causal mask for decoder, padding mask for variable-length batches.

Components from earlier weeks:
- **Residual connections** ([Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#10-fix-5-residual-skip-connections)) — $x + \text{Attention}(x)$.
- **Layer normalisation** ([Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#8-fix-3-layer-normalisation)) — normalise across features, not batch.
- **Feedforward network** — two-layer MLP applied position-wise.
- **Positional encoding** — since attention has no inherent notion of order.

[Week 18](../week18_transformers/theory.md#2-the-transformer-at-a-glance) assembles these into the full architecture.

---

## 12. Connections to the Rest of the Course

| Week                               | Connection                                                                                                                                      |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#3-vanishing-gradients) (Training Pathologies)** | Residual connections and LayerNorm are essential in Transformers; vanishing gradients in RNNs motivate the move to attention                    |
| **[Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md#4-building-models-with-nnmodule) (PyTorch)**              | Attention implemented as `nn.Module`; same training loop, `state_dict`, checkpointing                                                           |
| **[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md#6-mixed-precision-training) (Training at Scale)**    | Long sequences → large memory; mixed precision essential; gradient clipping common                                                              |
| **[Week 15](../../05_deep_learning/week15_cnn_representations/theory.md#12-classic-cnn-architectures) (CNNs)**                 | Self-attention can be seen as a data-dependent convolution with a global receptive field — each "filter" is determined by content, not position |
| **[Week 16](../../05_deep_learning/week16_regularization_dl/theory.md#3-dropout) (Regularisation)**       | Dropout applied after attention weights and after FFN; weight decay with AdamW standard for attention models                                    |
| **[Week 18](../week18_transformers/theory.md#2-the-transformer-at-a-glance) (Transformers)**         | Assembles multi-head attention into a full encoder–decoder architecture with positional encoding                                                |
| **[Week 19](../../07_transfer_learning/week19_finetuning/theory.md#3-adaptation-strategies) (Fine-Tuning)**          | Attention weights in pretrained models encode learned relationships; fine-tuning adapts them to new tasks                                       |

---

## 13. Notebook Reference Guide

| Cell                       | Section             | What it demonstrates                                                                     | Theory reference |
| -------------------------- | ------------------- | ---------------------------------------------------------------------------------------- | ---------------- |
| 1 (Scaled dot-product)     | Core attention      | `scaled_dot_product_attention(Q, K, V, mask)` from scratch; shape verification           | Section 4        |
| 2 (Visualisation)          | Heatmaps            | Attention weight heatmap for a 6-word sentence                                           | Section 9        |
| 3 (Multi-head)             | MHA module          | `MultiHeadAttention(64, 8)` with head splitting; parameter count                         | Section 6        |
| 4 (Seq2Seq)                | Encoder–decoder     | `Seq2SeqWithAttention` (GRU + additive attention) trained on sequence reversal; 5 epochs | Section 8        |
| Ex. 1 (Causal mask)        | Masking             | Lower-triangular mask; verify no future leakage in weights                               | Section 4.4      |
| Ex. 2 (Cross-attention)    | Cross-attention     | Q from decoder, K/V from encoder; different sequence lengths                             | Section 7        |
| Ex. 3 (Head patterns)      | Multi-head analysis | Per-head heatmaps for 8 heads; identify local vs. global patterns                        | Section 6.4      |
| Ex. 4 (Attention vs. none) | Comparison          | Vanilla GRU seq2seq vs. attention seq2seq; loss curves side-by-side                      | Section 2.3      |

**Suggested modifications:**

| Modification                                                                            | What it reveals                                                                             |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Remove the $\sqrt{d_k}$ scaling and observe softmax outputs                             | Scores have large magnitude → softmax saturates → near-one-hot weights → poor gradient flow |
| Increase $d_k$ from 8 to 128 while keeping no scaling                                   | The saturation effect worsens with higher dimension (variance = $d_k$)                      |
| Replace softmax with sparsemax or top-$k$ selection                                     | Sparser attention; some positions get exactly 0 weight                                      |
| Train the seq2seq reverse task for 20 epochs and visualise the learned attention matrix | Should converge to a near-perfect anti-diagonal (position $i$ → position $T-i$)             |
| Add a causal mask to the self-attention and train a simple next-token predictor         | Autoregressive generation; foundation for GPT-style models                                  |
| Compare 1, 4, 8, 16 heads at fixed $d_\text{model} = 64$                                | Diminishing returns after ~8 heads; 1 head is significantly worse                           |
| Plot attention entropy $H = -\sum_i \alpha_i \log \alpha_i$ per head                    | Low entropy = focused attention; high entropy = diffuse; useful diagnostic                  |
| Measure wall-clock time for self-attention with $T \in \{64, 128, 256, 512, 1024\}$     | Quadratic scaling empirically visible; motivates efficient attention in [Week 18](../week18_transformers/theory.md#82-flops-and-memory)+            |

---

## 14. Symbol Reference

| Symbol           | Name                  | Meaning                                                            |
| ---------------- | --------------------- | ------------------------------------------------------------------ |
| $Q$              | Query matrix          | $(n \times d_k)$; what each position is looking for                |
| $K$              | Key matrix            | $(m \times d_k)$; what each position offers to match               |
| $V$              | Value matrix          | $(m \times d_v)$; what each position contributes to the output     |
| $d_k$            | Key dimension         | Dimension of query/key vectors                                     |
| $d_v$            | Value dimension       | Dimension of value vectors                                         |
| $d_\text{model}$ | Model dimension       | Total embedding dimension; $d_\text{model} = h \cdot d_k$          |
| $h$              | Number of heads       | Parallel attention operations                                      |
| $n$              | Number of queries     | Query sequence length                                              |
| $m$              | Number of keys/values | Key/value sequence length                                          |
| $T$              | Sequence length       | General sequence length variable                                   |
| $\alpha_{i,j}$   | Attention weight      | Weight assigned by query $i$ to key $j$; $\sum_j \alpha_{i,j} = 1$ |
| $W^Q, W^K, W^V$  | Projection matrices   | Map input to query/key/value subspaces                             |
| $W^O$            | Output projection     | Maps concatenated heads back to $d_\text{model}$                   |
| $s_t$            | Decoder hidden state  | Current state of the decoder RNN at step $t$                       |
| $h_i$            | Encoder hidden state  | Encoder output at position $i$                                     |
| $c_t$            | Context vector        | Weighted sum of encoder states: $\sum_i \alpha_{t,i} h_i$          |
| $\text{mask}$    | Attention mask        | Binary matrix; 0 = don't attend, 1 = attend                        |

---

## 15. References

1. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." *NeurIPS*. — The Transformer paper; scaled dot-product and multi-head attention.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR*. — Additive attention for seq2seq; the paper that introduced attention to NLP.
3. Luong, M.-T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation." *EMNLP*. — Dot-product and general attention variants.
4. Cho, K., van Merriënboer, B., Gulcehre, C., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *EMNLP*. — GRU; encoder–decoder architecture.
5. Xu, K., Ba, J., Kiros, R., et al. (2015). "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." *ICML*. — Hard vs. soft attention for image captioning.
6. Alammar, J. "The Illustrated Transformer." http://jalammar.github.io/illustrated-transformer/ — Excellent visual guide.
7. Olah, C. & Carter, S. (2016). "Attention and Augmented Recurrent Neural Networks." *Distill*. — Interactive visualisations of attention mechanisms.
8. Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2022). "Efficient Transformers: A Survey." *ACM Computing Surveys*. — Survey of linear and sparse attention variants.
9. Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*. — LSTM; the gated architecture that preceded attention.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 10 (Sequence Modeling). MIT Press. — RNN foundations.
