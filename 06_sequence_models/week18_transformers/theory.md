# Transformers

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [The Transformer at a Glance](#2-the-transformer-at-a-glance)
3. [Input Representation](#3-input-representation)
    - 3.1 [Token Embeddings](#31-token-embeddings)
    - 3.2 [Sinusoidal Positional Encoding](#32-sinusoidal-positional-encoding)
    - 3.3 [Learned Positional Embeddings](#33-learned-positional-embeddings)
    - 3.4 [Embedding Scale Factor](#34-embedding-scale-factor)
4. [The Transformer Encoder Layer](#4-the-transformer-encoder-layer)
    - 4.1 [Sub-layer 1 — Multi-Head Self-Attention](#41-sub-layer-1--multi-head-self-attention)
    - 4.2 [Sub-layer 2 — Position-Wise Feed-Forward Network](#42-sub-layer-2--position-wise-feed-forward-network)
    - 4.3 [Residual Connections](#43-residual-connections)
    - 4.4 [Layer Normalisation](#44-layer-normalisation)
    - 4.5 [Post-LN vs. Pre-LN](#45-post-ln-vs-pre-ln)
    - 4.6 [The Complete Encoder Layer](#46-the-complete-encoder-layer)
5. [Stacking: The Full Transformer Encoder](#5-stacking-the-full-transformer-encoder)
6. [The Transformer Decoder](#6-the-transformer-decoder)
    - 6.1 [Masked Self-Attention](#61-masked-self-attention)
    - 6.2 [Cross-Attention](#62-cross-attention)
    - 6.3 [Full Decoder Layer](#63-full-decoder-layer)
7. [Encoder–Decoder vs. Encoder-Only vs. Decoder-Only](#7-encoderdecoder-vs-encoder-only-vs-decoder-only)
8. [Computational Analysis](#8-computational-analysis)
    - 8.1 [Parameter Count](#81-parameter-count)
    - 8.2 [FLOPS and Memory](#82-flops-and-memory)
    - 8.3 [Scaling Laws](#83-scaling-laws)
9. [Training Considerations](#9-training-considerations)
    - 9.1 [Learning Rate Warm-Up](#91-learning-rate-warm-up)
    - 9.2 [Dropout Placement](#92-dropout-placement)
    - 9.3 [Weight Initialisation](#93-weight-initialisation)
10. [Classification Head — Pooling Strategies](#10-classification-head--pooling-strategies)
11. [Attention Patterns Across Layers](#11-attention-patterns-across-layers)
12. [Depth vs. Width Trade-offs](#12-depth-vs-width-trade-offs)
13. [Connections to the Rest of the Course](#13-connections-to-the-rest-of-the-course)
14. [Notebook Reference Guide](#14-notebook-reference-guide)
15. [Symbol Reference](#15-symbol-reference)
16. [References](#16-references)

---

## 1. Scope and Purpose

This week assembles components from [Week 17](../week17_attention/theory.md#31-the-query-key-value-framework) (attention) and earlier weeks (residual connections, layer normalisation, dropout) into the **Transformer** — the architecture behind BERT, GPT, T5, Vision Transformers, and virtually every modern foundation model.

After this week you will be able to:

1. **Read the original paper** ("Attention Is All You Need") and map every equation to code.
2. **Build a Transformer encoder from scratch** — positional encodings, multi-head self-attention, feed-forward layers, residual connections, layer norm.
3. **Train a Transformer classifier** on a sequence task.
4. **Profile** the computational cost as a function of depth, width, and sequence length.

**Prerequisites.** [Week 17](../week17_attention/theory.md#4-scaled-dot-product-attention) (scaled dot-product attention, multi-head attention, masking). [Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#10-fix-5-residual-skip-connections) (residual connections, LayerNorm, BatchNorm). [Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md#4-building-models-with-nnmodule)–[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md#4-learning-rate-schedules) (PyTorch `nn.Module`, training loops).

---

## 2. The Transformer at a Glance

$$\boxed{\text{Transformer} = \text{Embedding} + \text{Positional Encoding} + N \times \text{Transformer Layer}}$$

Each **Transformer Layer** contains:

```
Input x
  │
  ├── Multi-Head Self-Attention ──► Add & Norm ──► y₁
  │         ▲                           │
  │         │  (residual)               │
  │         └───────────────────────────┘
  │
  y₁
  │
  ├── Feed-Forward Network ──► Add & Norm ──► y₂
  │         ▲                      │
  │         │  (residual)          │
  │         └──────────────────────┘
  │
  y₂ (output)
```

The **encoder** stacks $N$ such layers. The **decoder** adds a third sub-layer (cross-attention to the encoder output) and uses causal masking in self-attention.

Key insight: **there is no recurrence and no convolution.** All sequence modelling is done through attention. Order information comes entirely from positional encodings.

---

## 3. Input Representation

The input to a Transformer is a sequence of token indices $[t_1, t_2, \ldots, t_T]$. These are converted to continuous vectors via embedding + positional encoding.

### 3.1 Token Embeddings

A learnable lookup table $E \in \mathbb{R}^{|\mathcal{V}| \times d_\text{model}}$ maps each token index to a dense vector:

$$e_i = E[t_i] \in \mathbb{R}^{d_\text{model}}$$

```python
self.embedding = nn.Embedding(vocab_size, d_model)
```

---

### 3.2 Sinusoidal Positional Encoding

Since attention is **permutation-equivariant** (it treats the input as a set), we must explicitly inject position information. The original Transformer uses deterministic sinusoidal functions:

$$\text{PE}(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/d_\text{model}}}\right)$$

$$\text{PE}(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/d_\text{model}}}\right)$$

where $p$ is the position index and $i$ is the dimension index.

**Why sinusoids?**

1. **Bounded.** Values stay in $[-1, 1]$ regardless of sequence length.
2. **Unique.** Each position gets a distinct pattern across dimensions.
3. **Relative position captures.** For any fixed offset $k$, $\text{PE}(p + k)$ can be expressed as a linear function of $\text{PE}(p)$:

$$\begin{pmatrix}\sin(\omega_i(p+k))\\\cos(\omega_i(p+k))\end{pmatrix} = \begin{pmatrix}\cos(\omega_i k) & \sin(\omega_i k)\\-\sin(\omega_i k) & \cos(\omega_i k)\end{pmatrix}\begin{pmatrix}\sin(\omega_i p)\\\cos(\omega_i p)\end{pmatrix}$$

where $\omega_i = 1 / 10000^{2i/d_\text{model}}$. The rotation matrix depends only on the offset $k$, not on the absolute position $p$. This means dot products $\text{PE}(p)^\top \text{PE}(p+k)$ depend on $k$ alone — the model can learn to attend to relative positions.

4. **Multi-scale.** Low dimensions oscillate rapidly (local position); high dimensions oscillate slowly (global position). This creates a binary-like encoding across scales.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                 # (1, max_len, d_model)
        self.register_buffer('pe', pe)       # not a parameter — no gradients

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

> **`register_buffer`** stores the tensor in `state_dict` so it is saved/loaded with the model, but it is not updated by the optimiser.

---

### 3.3 Learned Positional Embeddings

An alternative: use a second `nn.Embedding(max_len, d_model)` indexed by position.

```python
self.pos_embedding = nn.Embedding(max_len, d_model)
# In forward:
positions = torch.arange(T, device=x.device).unsqueeze(0)
x = self.token_embedding(tokens) + self.pos_embedding(positions)
```

| | Sinusoidal | Learned |
|---|---|---|
| **Parameters** | 0 (deterministic) | $\text{max\_len} \times d_\text{model}$ |
| **Extrapolation** | Works beyond `max_len` (analytic formula) | Fails beyond trained length |
| **Performance** | Excellent for moderate $T$ | Matches sinusoidal; sometimes slightly better |
| **Used by** | Original Transformer, T5 | BERT, GPT-2 |

> **Notebook reference.** Exercise 1 asks you to replace sinusoidal with learned PE and compare validation accuracy.

---

### 3.4 Embedding Scale Factor

The original paper multiplies token embeddings by $\sqrt{d_\text{model}}$ before adding positional encodings:

$$x_i = \sqrt{d_\text{model}} \cdot E[t_i] + \text{PE}(i)$$

**Why?** Embedding vectors are initialised with small values (roughly $\mathcal{N}(0, 1)$ per component), so their $\ell_2$ norm is $\approx \sqrt{d_\text{model}} \cdot 1 = \sqrt{d_\text{model}}$… but that's the *expected norm under standard init*. After training, embedding magnitudes can drift. Multiplying by $\sqrt{d_\text{model}}$ ensures the embeddings are on a comparable scale to the positional encodings (which have values in $[-1, 1]$ and norm $\approx \sqrt{d_\text{model}/2}$) at initialisation.

```python
x = self.embedding(tokens) * np.sqrt(self.d_model)
x = self.pos_encoding(x)
```

---

## 4. The Transformer Encoder Layer

### 4.1 Sub-layer 1 — Multi-Head Self-Attention

Exactly the multi-head attention from [Week 17](../week17_attention/theory.md#6-multi-head-attention), applied as **self-attention** ($Q = K = V = x$):

$$\text{MultiHead}(x, x, x) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(xW_i^Q,\; xW_i^K,\; xW_i^V) = \text{softmax}\!\left(\frac{(xW_i^Q)(xW_i^K)^\top}{\sqrt{d_k}}\right)(xW_i^V)$$

The output has the same shape as the input: $(B, T, d_\text{model})$. Every position attends to every other position (bidirectional in the encoder).

---

### 4.2 Sub-layer 2 — Position-Wise Feed-Forward Network

A two-layer MLP applied **independently** to each position:

$$\text{FFN}(x) = \max(0,\; xW_1 + b_1)W_2 + b_2$$

with $W_1 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}$, $W_2 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}}$.

Typically $d_\text{ff} = 4 \times d_\text{model}$. The inner dimension $d_\text{ff}$ is the "expansion factor" — the network projects up, applies a non-linearity, and projects back down.

```python
self.ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),       # expand
    nn.ReLU(),                       # non-linearity
    nn.Dropout(dropout),
    nn.Linear(d_ff, d_model)        # contract
)
```

**"Position-wise"** means the same weights are shared across all $T$ positions, but each position is processed independently. This is equivalent to a 1×1 convolution over the sequence dimension.

**Modern variants:**
- **GELU** (BERT, GPT) replaces ReLU: $\text{GELU}(x) = x\,\Phi(x)$ where $\Phi$ is the standard normal CDF.
- **SwiGLU** (PaLM, LLaMA): $\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3)W_2$ — a gated variant with three weight matrices.

---

### 4.3 Residual Connections

Each sub-layer is wrapped in a residual (skip) connection:

$$\text{output} = \text{SubLayer}(x) + x$$

This is identical to the residual connections from [Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#10-fix-5-residual-skip-connections) (ResNets). Benefits:

1. **Gradient highway.** Gradients flow directly through the identity path, preventing vanishing gradients in deep stacks.
2. **Easy to learn identity.** If a layer is unhelpful, the network can learn $\text{SubLayer}(x) \approx 0$, effectively skipping it.
3. **Enables depth.** Without residuals, training Transformers deeper than ~2 layers is extremely difficult.

---

### 4.4 Layer Normalisation

After the residual connection (in the "Post-LN" variant), layer normalisation is applied:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu$ and $\sigma^2$ are the mean and variance computed **across the feature dimension** for each individual token position:

$$\mu = \frac{1}{d_\text{model}}\sum_{i=1}^{d_\text{model}}x_i, \qquad \sigma^2 = \frac{1}{d_\text{model}}\sum_{i=1}^{d_\text{model}}(x_i - \mu)^2$$

$\gamma, \beta \in \mathbb{R}^{d_\text{model}}$ are learnable scale and shift parameters.

**LayerNorm vs. BatchNorm** (from [Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#8-fix-3-layer-normalisation)):

| | LayerNorm | BatchNorm |
|---|---|---|
| Normalises over | Features (one token at a time) | Batch (one feature at a time) |
| Depends on batch? | No | Yes |
| Works with variable seq len? | Yes | Problematic |
| Used in | Transformers | CNNs |
| Parameters | $2 \times d_\text{model}$ per norm | $2 \times d_\text{model}$ per norm |

LayerNorm is preferred for Transformers because it normalises each sample independently — no inter-sample dependency, which matters for variable sequence lengths and small batches.

---

### 4.5 Post-LN vs. Pre-LN

**Post-LN** (original Transformer):

$$y = \text{LayerNorm}(x + \text{SubLayer}(x))$$

**Pre-LN** (used in GPT-2, many modern models):

$$y = x + \text{SubLayer}(\text{LayerNorm}(x))$$

```
Post-LN:   x ──► SubLayer ──► Add ──► LayerNorm ──► y
                    ▲          │
                    └── x ─────┘

Pre-LN:    x ──► LayerNorm ──► SubLayer ──► Add ──► y
                                             ▲
                                    x ───────┘
```

| | Post-LN | Pre-LN |
|---|---|---|
| Gradient scale | Grows with depth (can be unstable) | Approximately constant |
| Needs warm-up? | Yes (critical) | Less critical |
| Final quality | Slightly better (when trained carefully) | Slightly worse, but much easier to train |
| Needs final LN? | No | Yes (after last layer, before output head) |

> **Notebook reference.** Exercise 2 asks you to implement Pre-LN and compare training stability with Post-LN.

---

### 4.6 The Complete Encoder Layer

Combining all components (Post-LN variant, matching the notebook):

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sub-layer 1: Multi-head self-attention + residual + norm
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Sub-layer 2: Feed-forward + residual + norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x, attn_weights
```

---

## 5. Stacking: The Full Transformer Encoder

The complete encoder chains $N$ identical layers, preceded by embedding and positional encoding:

$$z_0 = \text{Dropout}(\text{PE}(\sqrt{d_\text{model}} \cdot E[\text{tokens}]))$$

$$z_\ell = \text{EncoderLayer}_\ell(z_{\ell-1}), \quad \ell = 1, \ldots, N$$

$$\text{encoder\_output} = z_N \in \mathbb{R}^{B \times T \times d_\text{model}}$$

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)

        return x, attn_weights_list
```

**Typical configurations:**

| Model | $N$ | $d_\text{model}$ | $h$ | $d_\text{ff}$ | Parameters |
|---|---|---|---|---|---|
| Transformer Base | 6 | 512 | 8 | 2048 | 65M |
| Transformer Big | 6 | 1024 | 16 | 4096 | 213M |
| BERT-base | 12 | 768 | 12 | 3072 | 110M |
| GPT-2 Small | 12 | 768 | 12 | 3072 | 117M |
| GPT-3 | 96 | 12288 | 96 | 49152 | 175B |
| Notebook example | 6 | 128 | 8 | 512 | ~1M |

---

## 6. The Transformer Decoder

The decoder generates output tokens autoregressively. Each decoder layer has **three** sub-layers instead of two.

### 6.1 Masked Self-Attention

The decoder's self-attention uses a **causal mask** to prevent position $t$ from attending to positions $t+1, \ldots, T$ (the future):

$$\text{mask}_{ij} = \begin{cases}0 & \text{if } j \leq i \\\text{$-\infty$} & \text{if } j > i\end{cases}$$

$$M = \begin{pmatrix}0 & -\infty & -\infty & -\infty\\0 & 0 & -\infty & -\infty\\0 & 0 & 0 & -\infty\\0 & 0 & 0 & 0\end{pmatrix}$$

After adding $M$ to the score matrix, softmax converts $-\infty$ entries to 0 — effectively masking out future positions.

```python
causal_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)
```

---

### 6.2 Cross-Attention

The second sub-layer in the decoder attends to the **encoder output**:

- Queries $Q$: from the decoder's previous sub-layer output
- Keys $K$, Values $V$: from the encoder output

$$\text{CrossAttn}(z_\text{dec}, z_\text{enc}) = \text{MultiHead}(Q = z_\text{dec},\; K = z_\text{enc},\; V = z_\text{enc})$$

This is how the decoder "reads" the source sequence. The decoder can have a different sequence length than the encoder.

---

### 6.3 Full Decoder Layer

```
Input z_dec, z_enc
  │
  ├── Masked Self-Attention(z_dec, z_dec, z_dec) ──► Add & Norm ──► y₁
  │
  ├── Cross-Attention(y₁, z_enc, z_enc) ──► Add & Norm ──► y₂
  │
  ├── Feed-Forward(y₂) ──► Add & Norm ──► y₃
  │
  y₃ (output)
```

Each sub-layer has its own residual connection and layer normalisation (6 LayerNorm modules per decoder layer, vs. 2 per encoder layer).

---

## 7. Encoder–Decoder vs. Encoder-Only vs. Decoder-Only

The original Transformer is an encoder–decoder model (for machine translation). Modern models specialise:

| Architecture | Components | Attention type | Example models | Typical tasks |
|---|---|---|---|---|
| **Encoder–Decoder** | Full Transformer | Bidirectional (enc) + Causal (dec) + Cross | T5, BART, mBART | Translation, summarisation |
| **Encoder-Only** | Encoder stack only | Bidirectional self-attention | BERT, RoBERTa | Classification, NER, QA |
| **Decoder-Only** | Decoder stack only (no cross-attn) | Causal self-attention | GPT, LLaMA, Mistral | Text generation, language modelling |

**Encoder-only** models see all tokens at once — every position attends to every other. This is ideal for understanding tasks (classification, extraction).

**Decoder-only** models generate tokens left-to-right — each position only attends to previous positions. This is the architecture behind modern LLMs. The causal mask is the only structural difference from an encoder.

---

## 8. Computational Analysis

### 8.1 Parameter Count

For a single encoder layer with $d_\text{model} = d$, $h$ heads, $d_\text{ff} = 4d$:

| Component | Parameters | Formula |
|---|---|---|
| $W^Q, W^K, W^V$ | $3(d^2 + d)$ | Three (weight + bias) projections |
| $W^O$ | $d^2 + d$ | Output projection |
| FFN layer 1 | $d \cdot d_\text{ff} + d_\text{ff} = 4d^2 + 4d$ | Expand |
| FFN layer 2 | $d_\text{ff} \cdot d + d = 4d^2 + d$ | Contract |
| LayerNorm ($\times 2$) | $2 \times 2d = 4d$ | Scale + shift, twice |
| **Total per layer** | $12d^2 + 13d$ | |

For $N$ layers plus the embedding:

$$\text{Total} = |\mathcal{V}| \cdot d + N(12d^2 + 13d)$$

For the notebook's configuration ($|\mathcal{V}| = 1000$, $d = 128$, $N = 6$):

$$1000 \times 128 + 6 \times (12 \times 128^2 + 13 \times 128) = 128{,}000 + 6 \times (196{,}608 + 1{,}664) = 128{,}000 + 1{,}189{,}632 \approx 1.3\text{M}$$

---

### 8.2 FLOPS and Memory

For sequence length $T$, model dimension $d$, per layer:

| Operation | FLOPS | Memory |
|---|---|---|
| Self-attention ($QK^\top$) | $O(T^2 d)$ | $O(T^2)$ for score matrix |
| Self-attention ($\alpha V$) | $O(T^2 d)$ | $O(Td)$ for output |
| Q, K, V projections | $O(Td^2)$ | $O(Td)$ |
| FFN | $O(Td \cdot d_\text{ff}) = O(4Td^2)$ | $O(Td_\text{ff})$ |
| **Total per layer** | $O(T^2 d + Td^2)$ | $O(T^2 + Td)$ |

The two scaling regimes:
- **Short sequences** ($T < d$): FFN dominates ($Td^2$ term). Cost is essentially linear in $T$.
- **Long sequences** ($T > d$): Attention dominates ($T^2 d$ term). Cost is quadratic in $T$.

For $d = 512$ and $T = 512$: both terms are $\approx 10^8$, roughly balanced. For $T = 4096$: attention is $16\times$ larger.

> **Notebook reference.** Cell 5 profiles inference time for $T \in \{50, 100, 200\}$ with fixed $d = 128$, $h = 8$, $N = 4$.

---

### 8.3 Scaling Laws

Kaplan et al. (2020) showed that Transformer loss follows power laws:

$$L(N_\text{params}) \propto N_\text{params}^{-\alpha}, \qquad L(D) \propto D^{-\beta}, \qquad L(C) \propto C^{-\gamma}$$

where $D$ is dataset size and $C$ is compute budget. Key findings:
- Performance is a smooth function of scale — bigger models are predictably better.
- Larger models are **more sample-efficient** — they reach a given loss with fewer training tokens.
- Optimal allocation: scale model size and data jointly (Chinchilla, Hoffmann et al., 2022).

---

## 9. Training Considerations

### 9.1 Learning Rate Warm-Up

The original Transformer uses a schedule that warms up linearly for `warmup_steps`, then decays:

$$\text{lr}(t) = d_\text{model}^{-0.5} \cdot \min(t^{-0.5},\; t \cdot \text{warmup\_steps}^{-1.5})$$

```
lr
 │        ╱╲
 │      ╱    ╲
 │    ╱        ╲───────
 │  ╱                  ───────
 │╱                           ───
 └──────────────────────────────── step
   warmup        decay
```

**Why warm up?**
- At initialisation, attention weights are near-uniform (all keys score similarly). Gradients for Q, K, V projections are noisy.
- A large LR amplifies this noise, causing divergent updates — particularly harmful because several components (attention, FFN, residual) interact.
- Warm-up lets the model settle into a reasonable region before taking larger steps.

Modern practice: Adam with warm-up + cosine decay, or AdamW with linear warm-up + linear decay.

---

### 9.2 Dropout Placement

Dropout is applied at four locations per layer:

1. **After attention weights** (before multiplying by $V$): regularises the attention distribution.
2. **After the attention sublayer output** (before the residual add).
3. **Inside FFN** (between the two linear layers).
4. **After the FFN sublayer output** (before the residual add).

Plus:
5. **After embedding + positional encoding** (before entering the first layer).

Typical rate: 0.1 for base models, 0.3 for small models on small data.

---

### 9.3 Weight Initialisation

PyTorch default (Kaiming uniform) works, but the original paper and many implementations use **Xavier/Glorot** initialisation for linear layers:

$$W \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{d_\text{in} + d_\text{out}}},\; \sqrt{\frac{6}{d_\text{in} + d_\text{out}}}\right)$$

For deep Transformers ($N > 12$), some approaches scale residual branch weights by $1/\sqrt{2N}$ (GPT-2) to prevent the output variance from growing with depth.

---

## 10. Classification Head — Pooling Strategies

The encoder output is $(B, T, d_\text{model})$ — one vector per position. For classification, we need a single vector per sequence.

| Strategy | Formula | Used in |
|---|---|---|
| **[CLS] token** | $h_\text{CLS} = z_N[:, 0, :]$ | BERT |
| **Mean pooling** | $h = \frac{1}{T}\sum_{t=1}^{T}z_N[:, t, :]$ | Sentence-BERT, notebook |
| **Max pooling** | $h_i = \max_t z_N[:, t, i]$ | Some classification models |
| **Weighted pooling** | $h = \sum_t \alpha_t z_N[:, t, :]$ (learned weights) | Attention pooling |

The notebook uses mean pooling:

```python
pooled = encoder_output.mean(dim=1)    # (B, d_model)
logits = self.classifier(pooled)        # (B, num_classes)
```

Mean pooling is a strong, parameter-free default. The [CLS] approach requires a special token and training that specifically endows position 0 with summary information.

---

## 11. Attention Patterns Across Layers

As data flows through $N$ layers, attention patterns evolve:

| Layer depth | Typical pattern | Intuition |
|---|---|---|
| **Early layers** (1–2) | Attend to nearby positions; broad, diffuse | Building local features |
| **Middle layers** (3–4) | Mixed: some heads attend locally, some globally | Composing local features into phrases/concepts |
| **Late layers** (5–6) | Task-specific, sharper patterns | Focusing on tokens most relevant to the output task |

**Attention entropy** $H = -\sum_j \alpha_j \log \alpha_j$ measures how focused a head is:
- High $H$ → uniform attention (diffuse)
- Low $H$ → concentrated on a few positions (focused)

Early layers tend toward higher entropy (more uniform); later layers toward lower entropy (more focused).

> **Notebook reference.** Exercise 4 asks you to visualise attention weights from each of the 6 encoder layers for a single input sample.

---

## 12. Depth vs. Width Trade-offs

For a fixed parameter budget, should you add more layers (depth) or make each layer wider ($d_\text{model}$)?

| Property | Deeper (more layers $N$) | Wider (larger $d_\text{model}$) |
|---|---|---|
| **Representational power** | More stages of composition; can learn hierarchical features | Each layer has more capacity per step |
| **Training stability** | Harder (vanishing/exploding gradients, more residual paths) | Easier (fewer layers to propagate through) |
| **Computational cost** | Same FLOPS can be distributed differently | More FLOPS per layer, fewer layers |
| **Empirical finding** | Depth helps more for tasks requiring reasoning/composition | Width helps more for memorisation-heavy tasks |

General rule of thumb: for sequence tasks, **depth matters more** than width. A 6-layer model outperforms a 2-layer model even if the 2-layer model has 3× more parameters per layer. However, returns diminish: a 12-layer model is rarely 2× better than a 6-layer model.

> **Notebook reference.** Exercise 3 trains four TransformerClassifier variants ($N \in \{2, 6\}$, $d_\text{model} \in \{64, 128\}$) and compares.

---

## 13. Connections to the Rest of the Course

| Week | Connection |
|---|---|
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#5-backpropagation) (NN from Scratch)** | Backpropagation through the computational graph still applies — the chain rule handles attention→softmax→matmul |
| **[Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#10-fix-5-residual-skip-connections) (Training Pathologies)** | Residual connections and LayerNorm are essential Transformer components; originated from "vanishing gradient" solutions |
| **[Week 13](../../05_deep_learning/week13_pytorch_basics/theory.md#4-building-models-with-nnmodule) (PyTorch)** | `nn.Module` pattern, `state_dict`, training loop — all reused for Transformer training |
| **[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md#4-learning-rate-schedules) (Training at Scale)** | Warm-up schedules, mixed precision (FP16 attention scores), gradient accumulation for large batch sizes |
| **[Week 15](../../05_deep_learning/week15_cnn_representations/theory.md#12-classic-cnn-architectures) (CNNs)** | FFN is like a 1×1 convolution; Transformers in vision (ViT) replace convolutions entirely |
| **[Week 16](../../05_deep_learning/week16_regularization_dl/theory.md#3-dropout) (Regularisation)** | Dropout at five locations; weight decay with AdamW; data augmentation for Transformer pre-training (masking, span corruption) |
| **[Week 17](../week17_attention/theory.md#31-the-query-key-value-framework) (Attention)** | Scaled dot-product attention, multi-head attention, masking — all building blocks assembled here |
| **[Week 19](../../07_transfer_learning/week19_finetuning/theory.md#3-adaptation-strategies) (Fine-Tuning)** | Pretrained Transformer weights from HuggingFace — knowing the architecture lets you fine-tune intelligently |

---

## 14. Notebook Reference Guide

| Cell | Section | What it demonstrates | Theory reference |
|---|---|---|---|
| 1 (Positional Encoding) | Sinusoidal PE | `PositionalEncoding` class; heatmap (position × dimension); per-dimension line plot | Section 3.2 |
| 2 (Encoder Layer) | Single layer | `TransformerEncoderLayer` with `nn.MultiheadAttention`, FFN, LayerNorm, residual connections; parameter count | Section 4 |
| 3 (Full Encoder) | Stacked layers | `TransformerEncoder` with 6 layers, vocab=1000, d=128, h=8, d_ff=512 | Section 5 |
| 4 (Classification) | End-to-end task | `TransformerClassifier` (mean pooling → linear); synthetic 3-class task; Adam lr=0.001, 5 epochs; loss/accuracy plots | Section 10 |
| 5 (Profiling) | Compute costs | Inference time vs. parameters; inference time vs. sequence length; 5 configurations | Section 8 |
| Ex. 1 (Learned PE) | PE comparison | Replace sinusoidal with `nn.Embedding`; compare val accuracy | Section 3.3 |
| Ex. 2 (Pre-LN) | Norm variant | Pre-LayerNorm encoder layer; compare training stability | Section 4.5 |
| Ex. 3 (Depth vs. Width) | Scaling | 4 variants: $N \in \{2,6\}$, $d \in \{64,128\}$; 2×2 grid of accuracy curves | Section 12 |
| Ex. 4 (Attn per layer) | Visualisation | Per-layer attention heatmaps for a single sample | Section 11 |

**Suggested modifications:**

| Modification | What it reveals |
|---|---|
| Remove positional encoding entirely and train the classifier | Accuracy drops — the model cannot distinguish word order; verifies that attention is permutation-equivariant |
| Remove residual connections from all layers | Training collapses for $N > 2$; gradients vanish just as they did in deep MLPs ([Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#3-vanishing-gradients)) |
| Remove LayerNorm | Training becomes unstable; loss spikes; note how norm stabilises the residual stream |
| Increase $d_\text{ff}$ from $4d$ to $8d$ and compare | More parameters in FFN; marginal accuracy gain, higher memory; the FFN is often the parameter bottleneck |
| Replace ReLU with GELU in the FFN | GELU is the BERT/GPT default; slightly smoother gradients; usually small accuracy gain |
| Profile with $T \in \{50, 100, 200, 400, 800\}$ | The $T^2$ scaling becomes unmistakable; motivates Flash Attention and efficient attention variants |
| Add label smoothing (0.1) to CrossEntropyLoss | Regularises the classifier; prevents overconfident predictions; common in Transformer training |
| Train for 50 epochs on the synthetic task and plot attention entropy per layer over training | Early: all heads are diffuse; late: some heads focus, others go dormant — a known Transformer phenomenon |

---

## 15. Symbol Reference

| Symbol | Name | Meaning |
|---|---|---|
| $T$ | Sequence length | Number of tokens in the input |
| $d_\text{model}$ | Model dimension | Width of the Transformer; size of all residual-stream vectors |
| $d_k$ | Key/query dimension | Per-head dimension; $d_k = d_\text{model} / h$ |
| $d_v$ | Value dimension | Per-head value dimension; typically $d_v = d_k$ |
| $d_\text{ff}$ | Feed-forward dimension | Inner dimension of the FFN; typically $4 \times d_\text{model}$ |
| $h$ | Number of heads | Parallel attention operations |
| $N$ | Number of layers | Depth of the encoder (or decoder) stack |
| $|\mathcal{V}|$ | Vocabulary size | Number of distinct tokens |
| $E$ | Embedding matrix | $\mathbb{R}^{|\mathcal{V}| \times d_\text{model}}$; maps token index to vector |
| $\text{PE}(p, i)$ | Positional encoding | Value at position $p$, dimension $i$ |
| $W^Q, W^K, W^V$ | Projection matrices | Map input to Q/K/V subspaces for each head |
| $W^O$ | Output projection | Maps concatenated heads back to $d_\text{model}$ |
| $W_1, W_2$ | FFN weights | First (expand) and second (contract) linear layers |
| $\gamma, \beta$ | LayerNorm parameters | Learnable scale and shift |
| $\alpha_{ij}$ | Attention weight | Weight from query $i$ to key $j$; $\sum_j \alpha_{ij} = 1$ |
| $z_\ell$ | Layer $\ell$ output | Residual stream after the $\ell$-th Transformer layer |

---

## 16. References

1. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." *NeurIPS*. — The original Transformer paper.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*. — Encoder-only Transformer with masked language modelling.
3. Radford, A., Wu, J., Child, R., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2). — Decoder-only Transformer; Pre-LN architecture.
4. Raffel, C., Shazeer, N., Roberts, A., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*. — T5; encoder–decoder Transformer; relative positional encodings.
5. Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). "Scaling Laws for Neural Language Models." *arXiv 2001.08361*. — Power-law relationships between model size, data, compute, and loss.
6. Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla). *arXiv 2203.15556*. — Optimal model/data scaling.
7. Xiong, R., Yang, Y., He, J., et al. (2020). "On Layer Normalization in the Transformer Architecture." *ICML*. — Analysis of Pre-LN vs. Post-LN.
8. Rush, A. (2018). "The Annotated Transformer." http://nlp.seas.harvard.edu/2018/04/03/attention.html — Line-by-line implementation of the original paper.
9. Alammar, J. "The Illustrated Transformer." https://jalammar.github.io/illustrated-transformer/ — Visual guide to every component.
10. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*. — Efficient attention implementation that avoids materialising the $T^2$ score matrix.
11. Dehghani, M., Djolonga, J., Mustafa, B., et al. (2023). "Scaling Vision Transformers to 22 Billion Parameters." *arXiv 2302.05442*. — Transformers applied to vision at scale.
12. Phuong, M. & Hutter, M. (2022). "Formal Algorithms for Transformers." *arXiv 2207.09238*. — Rigorous pseudocode for every Transformer component.
