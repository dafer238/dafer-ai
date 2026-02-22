# Fine-Tuning and Transfer Learning

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Transfer Learning — The Core Idea](#2-transfer-learning--the-core-idea)
   - 2.1 [Why Transfer?](#21-why-transfer)
   - 2.2 [Pre-training → Fine-Tuning Pipeline](#22-pre-training--fine-tuning-pipeline)
   - 2.3 [Domain Gap and Distribution Shift](#23-domain-gap-and-distribution-shift)
3. [Adaptation Strategies](#3-adaptation-strategies)
   - 3.1 [Feature Extraction (Frozen Backbone)](#31-feature-extraction-frozen-backbone)
   - 3.2 [Full Fine-Tuning](#32-full-fine-tuning)
   - 3.3 [Partial Fine-Tuning (Gradual Unfreezing)](#33-partial-fine-tuning-gradual-unfreezing)
   - 3.4 [Decision Framework](#34-decision-framework)
4. [The Mechanics of Fine-Tuning](#4-the-mechanics-of-fine-tuning)
   - 4.1 [Replacing the Head](#41-replacing-the-head)
   - 4.2 [Learning Rate Selection](#42-learning-rate-selection)
   - 4.3 [Discriminative Learning Rates](#43-discriminative-learning-rates)
   - 4.4 [Warm-Up and Scheduling](#44-warm-up-and-scheduling)
   - 4.5 [Epochs and Early Stopping](#45-epochs-and-early-stopping)
5. [Parameter-Efficient Fine-Tuning (PEFT)](#5-parameter-efficient-fine-tuning-peft)
   - 5.1 [Adapters](#51-adapters)
   - 5.2 [LoRA (Low-Rank Adaptation)](#52-lora-low-rank-adaptation)
   - 5.3 [Prefix Tuning and Prompt Tuning](#53-prefix-tuning-and-prompt-tuning)
   - 5.4 [Comparison of PEFT Methods](#54-comparison-of-peft-methods)
6. [Feature Extraction in Detail](#6-feature-extraction-in-detail)
   - 6.1 [When and Why](#61-when-and-why)
   - 6.2 [Which Layer to Extract From](#62-which-layer-to-extract-from)
7. [Fine-Tuning Pretrained Transformers (HuggingFace Workflow)](#7-fine-tuning-pretrained-transformers-huggingface-workflow)
   - 7.1 [Loading a Pretrained Model](#71-loading-a-pretrained-model)
   - 7.2 [Tokenisation](#72-tokenisation)
   - 7.3 [TrainingArguments and Trainer](#73-trainingarguments-and-trainer)
8. [Fine-Tuning for Vision (Transfer from CNNs)](#8-fine-tuning-for-vision-transfer-from-cnns)
9. [Catastrophic Forgetting](#9-catastrophic-forgetting)
10. [Evaluation and Data Considerations](#10-evaluation-and-data-considerations)
    - 10.1 [Small-Data Regimes](#101-small-data-regimes)
    - 10.2 [Class Imbalance](#102-class-imbalance)
    - 10.3 [Metrics Beyond Accuracy](#103-metrics-beyond-accuracy)
11. [Connections to the Rest of the Course](#11-connections-to-the-rest-of-the-course)
12. [Notebook Reference Guide](#12-notebook-reference-guide)
13. [Symbol Reference](#13-symbol-reference)
14. [References](#14-references)

---

## 1. Scope and Purpose

This week covers the practical skill that dominates real-world AI: **taking a pretrained model and adapting it to a new task with limited data**. Training from scratch is expensive, data-hungry, and usually unnecessary. Transfer learning makes high-quality models accessible with small datasets and modest compute.

After this week you will be able to:

1. **Choose the right adaptation strategy** — feature extraction, full fine-tuning, partial freezing, or parameter-efficient methods — for a given data regime and domain gap.
2. **Fine-tune a HuggingFace Transformer** (e.g., DistilBERT) on a downstream classification task.
3. **Implement and compare** LoRA, adapters, and frozen-backbone approaches.
4. **Avoid catastrophic forgetting** and select appropriate learning rates, schedules, and regularisation.

**Prerequisites.** [Week 18](../../06_sequence_models/week18_transformers/theory.md) (Transformer architecture — you must know what is inside the model you are adapting). [Week 15](../../05_deep_learning/week15_cnn_representations/theory.md) (CNNs for vision transfer). [[Week 06](../../02_fundamentals/week06_regularization/theory.md)](../../02_fundamentals/week06_regularization/theory.md)/[16](../../05_deep_learning/week16_regularization_dl/theory.md) (regularisation — freezing layers is a form of parameter regularisation).

---

## 2. Transfer Learning — The Core Idea

### 2.1 Why Transfer?

A model pretrained on a large corpus (e.g., Wikipedia + BooksCorpus for BERT, ImageNet for ResNet) has already learned:

- **Low-level features:** edges, textures, word morphology, syntax.
- **Mid-level features:** object parts, phrase structure, semantic relationships.
- **High-level features:** object categories, sentence semantics, discourse patterns.

These features are **largely task-agnostic**. Fine-tuning reuses them and adapts only what is needed for the target task.

| Approach | Data required | Compute cost | When to use |
|---|---|---|---|
| Train from scratch | $10^5$–$10^9$ samples | Very high | Novel domain with massive data |
| Fine-tune all parameters | $10^2$–$10^5$ samples | Moderate | Moderate data, close domain |
| Feature extraction | $10^1$–$10^3$ samples | Low | Very small data, close domain |
| PEFT (LoRA, adapters) | $10^2$–$10^4$ samples | Low–moderate | Limited GPU memory; many tasks |

---

### 2.2 Pre-training → Fine-Tuning Pipeline

$$\boxed{\text{Large corpus} \xrightarrow{\text{pre-train}} \theta_\text{pre} \xrightarrow{\text{fine-tune on } \mathcal{D}_\text{target}} \theta_\text{fine}}$$

**Pre-training objectives:**

| Domain | Objective | Model |
|---|---|---|
| NLP | Masked language modelling (MLM): predict masked tokens | BERT, RoBERTa |
| NLP | Causal language modelling (CLM): predict next token | GPT, LLaMA |
| NLP | Span corruption: predict corrupted spans | T5 |
| Vision | Supervised classification on ImageNet (1.2M images, 1000 classes) | ResNet, EfficientNet |
| Vision | Self-supervised (contrastive learning, MAE) | DINO, MAE |

The pretrained weights $\theta_\text{pre}$ encode general knowledge. Fine-tuning adjusts them — usually with a much smaller learning rate — on the target dataset $\mathcal{D}_\text{target}$.

---

### 2.3 Domain Gap and Distribution Shift

Transfer works best when source and target domains are **similar**. The concept of **domain gap** measures how different the two distributions are:

$$\text{Domain gap} = d(\mathcal{P}_\text{source}, \mathcal{P}_\text{target})$$

| Gap size | Example | Strategy |
|---|---|---|
| **Small** | BERT pretrained on English Wikipedia → English sentiment analysis | Feature extraction or light fine-tuning |
| **Medium** | ResNet trained on ImageNet → medical X-ray classification | Fine-tune with frozen early layers |
| **Large** | English BERT → Japanese patent classification | Full fine-tuning; possibly new tokeniser; consider pre-training on target domain first |

**Types of distribution shift:**

1. **Covariate shift.** Input distribution changes ($P(x)$ differs), but the labelling function $P(y|x)$ is the same.
2. **Label shift.** Class proportions change.
3. **Concept drift.** The relationship $P(y|x)$ itself changes (most severe; transfer may hurt).

---

## 3. Adaptation Strategies

### 3.1 Feature Extraction (Frozen Backbone)

Freeze all pretrained parameters. Only train a new classification head on top.

```python
# Freeze the backbone
for param in model.base_model.parameters():
    param.requires_grad = False

# Only the classifier head is trainable
for param in model.classifier.parameters():
    param.requires_grad = True
```

The pretrained model acts as a **fixed feature extractor**. The input passes through the frozen network to produce embeddings, and only a simple classifier (linear layer or small MLP) is trained on those embeddings.

**Advantages:**
- Fast training (few parameters to update).
- Low risk of overfitting on small datasets.
- No risk of corrupting pretrained features.

**Disadvantages:**
- Cannot adapt representations to the target domain.
- Performs poorly when the domain gap is large.

---

### 3.2 Full Fine-Tuning

Update **all** parameters, including the backbone, using a small learning rate:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

The pretrained weights serve as a strong initialisation. The small learning rate ensures the model moves slowly from the pretrained solution, retaining useful features while adapting to the target.

**Advantages:**
- Maximum adaptability — the entire model is tailored to the task.
- Best accuracy when sufficient target data is available.

**Disadvantages:**
- Risk of catastrophic forgetting (Section 9).
- Risk of overfitting with very small datasets.
- Higher compute cost.

---

### 3.3 Partial Fine-Tuning (Gradual Unfreezing)

A middle ground: initially freeze most layers, then progressively unlock deeper layers:

**Epoch 1–2:** Only train the classification head.
**Epoch 3–4:** Unfreeze the last Transformer layer.
**Epoch 5–6:** Unfreeze the last two layers.
**Epoch 7+:** Unfreeze all layers (optionally with discriminative LRs).

```python
# Phase 1: freeze everything except the head
for param in model.base_model.parameters():
    param.requires_grad = False

# Phase 2: unfreeze the last encoder layer
for param in model.base_model.encoder.layer[-1].parameters():
    param.requires_grad = True

# Phase 3: unfreeze all
for param in model.parameters():
    param.requires_grad = True
```

This approach — popularised by ULMFiT (Howard & Ruder, 2018) — reduces the risk of catastrophic forgetting while still allowing deep adaptation.

---

### 3.4 Decision Framework

```
                    Target dataset size?
                   /                    \
              Small (< 1k)          Large (> 10k)
               /       \                  |
         Domain gap?   Domain gap?    Full fine-tuning
          /      \        |
      Small    Large    Large
       |         |        |
  Feature     Partial   Full fine-tune
  extraction  freeze    + more data
  (frozen)    or PEFT   augmentation
```

**Rules of thumb:**

| Scenario | Recommended strategy |
|---|---|
| Small data, small domain gap | Feature extraction |
| Small data, large domain gap | PEFT (LoRA/adapters) or gradual unfreezing |
| Large data, small domain gap | Full fine-tuning (converges quickly) |
| Large data, large domain gap | Full fine-tuning with longer training |
| Multiple tasks, shared backbone | PEFT (one adapter per task) |
| Deployment memory constrained | PEFT (store only the adapter deltas) |

---

## 4. The Mechanics of Fine-Tuning

### 4.1 Replacing the Head

Pretrained models have a head designed for the pre-training task (e.g., masked LM head for BERT). For a downstream task, **replace the head**:

```python
from transformers import AutoModelForSequenceClassification

# Loads pretrained backbone + new random classification head
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2        # binary classification
)
```

Under the hood, this:
1. Loads the pretrained DistilBERT weights.
2. Discards the pre-training head (masked LM prediction layer).
3. Adds a new `nn.Linear(768, 2)` with random weights.

The new head must be trained from scratch — it has no pretrained knowledge. The backbone already has strong representations.

---

### 4.2 Learning Rate Selection

The single most important hyperparameter in fine-tuning:

| Learning rate | Effect |
|---|---|
| **Too high** ($> 10^{-4}$) | Catastrophic forgetting — pretrained features are destroyed |
| **Good range** ($2 \times 10^{-5}$ to $5 \times 10^{-5}$) | Standard for Transformer fine-tuning |
| **Too low** ($< 10^{-6}$) | Barely moves from pretrained weights; slow convergence |

**Why so small?** Pretrained weights are already in a good basin of the loss landscape. Large updates push the model out of this basin, losing the benefit of pre-training.

Compare to training from scratch:
- From scratch: $\text{lr} \sim 10^{-3}$ (need to traverse a large distance in parameter space).
- Fine-tuning: $\text{lr} \sim 2 \times 10^{-5}$ (need to make small adjustments within an existing basin).

---

### 4.3 Discriminative Learning Rates

Different layers need different learning rates. Early layers encode general features (change less); later layers encode task-specific features (change more).

$$\text{lr}_\ell = \text{lr}_\text{base} \cdot \eta^{L - \ell}$$

where $\ell$ is the layer index, $L$ is the total number of layers, and $\eta < 1$ is a decay factor (typically $\eta = 0.95$).

For a 6-layer Transformer with $\text{lr}_\text{base} = 3 \times 10^{-5}$, $\eta = 0.95$:

| Layer | LR |
|---|---|
| Head (classifier) | $3.0 \times 10^{-5}$ |
| Layer 6 | $2.85 \times 10^{-5}$ |
| Layer 5 | $2.71 \times 10^{-5}$ |
| Layer 4 | $2.57 \times 10^{-5}$ |
| Layer 3 | $2.44 \times 10^{-5}$ |
| Layer 2 | $2.32 \times 10^{-5}$ |
| Layer 1 | $2.21 \times 10^{-5}$ |
| Embeddings | $2.10 \times 10^{-5}$ |

Implementation with parameter groups:

```python
param_groups = []
for i, layer in enumerate(model.base_model.encoder.layer):
    lr = base_lr * (decay ** (num_layers - i))
    param_groups.append({'params': layer.parameters(), 'lr': lr})
param_groups.append({'params': model.classifier.parameters(), 'lr': base_lr})
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

---

### 4.4 Warm-Up and Scheduling

Fine-tuning typically uses:

1. **Linear warm-up** for 6–10% of total steps: LR ramps from 0 to the target value.
2. **Linear decay** (or cosine decay) for the remaining steps: LR decreases to 0.

$$\text{lr}(t) = \begin{cases}\text{lr}_\text{max} \cdot t / t_\text{warmup} & t \leq t_\text{warmup}\\\text{lr}_\text{max} \cdot (1 - (t - t_\text{warmup})/(t_\text{total} - t_\text{warmup})) & t > t_\text{warmup}\end{cases}$$

**Why warm-up for fine-tuning?** The new classification head has random weights. In early steps, its gradients are large and noisy. Without warm-up, these noisy gradients propagate into the backbone and corrupt pretrained features. Warm-up lets the head stabilise before the backbone receives large updates.

---

### 4.5 Epochs and Early Stopping

Fine-tuning converges fast — pretrained weights are already close to a good solution:

| Setting | Typical epochs |
|---|---|
| NLP (BERT/DistilBERT on GLUE) | 2–4 |
| Vision (ResNet on small dataset) | 5–15 |
| PEFT (LoRA) | 3–10 |

**Early stopping:** monitor validation loss; stop when it increases for $k$ consecutive evaluations (patience). With only 3 epochs, early stopping is less critical — but still use it to pick the best checkpoint.

---

## 5. Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all $N$ parameters. For a 110M-parameter BERT, that means storing 110M gradients and optimizer states. PEFT methods train only a small number of **additional** parameters, keeping the backbone frozen.

### 5.1 Adapters

Small bottleneck modules inserted **inside** each Transformer layer:

$$\text{Adapter}(x) = x + f(xW_\text{down})W_\text{up}$$

where $W_\text{down} \in \mathbb{R}^{d_\text{model} \times r}$ and $W_\text{up} \in \mathbb{R}^{r \times d_\text{model}}$ with bottleneck dimension $r \ll d_\text{model}$ (e.g., $r = 64$).

```
Input x (d_model)
    │
    ├──────────────────────┐
    │                      │ (residual)
    ▼                      │
  Linear(d → r)            │
    │                      │
  ReLU / GELU              │
    │                      │
  Linear(r → d)            │
    │                      │
    + ◄────────────────────┘
    │
  Output (d_model)
```

Parameters per adapter: $2 \times d_\text{model} \times r + r + d_\text{model}$ (weights + biases). For $d = 768$, $r = 64$: $\approx 99$K per adapter, two adapters per layer (after attention and after FFN), 12 layers: $\approx 2.4$M total — about **2% of BERT's parameters**.

---

### 5.2 LoRA (Low-Rank Adaptation)

Instead of adding new modules, LoRA modifies existing weight matrices with a low-rank update:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d_\text{out} \times r}$, $A \in \mathbb{R}^{r \times d_\text{in}}$, and $r \ll \min(d_\text{in}, d_\text{out})$.

The original weight $W$ is **frozen**. Only $A$ and $B$ are trained.

$$\text{output} = xW^\top + x(BA)^\top = xW^\top + xA^\top B^\top$$

**Key properties:**

1. **Parameters.** Per adapted matrix: $r \times (d_\text{in} + d_\text{out})$. For $W^Q$ in BERT ($768 \times 768$), $r = 8$: $8 \times 1536 = 12{,}288$ parameters (vs. $589{,}824$ in the original matrix — a $48\times$ reduction).

2. **No inference overhead.** After training, merge: $W_\text{merged} = W + BA$. The model has identical architecture and latency as the original.

3. **Initialisation.** $A$ is initialised with small random values (Gaussian); $B$ is initialised to zero. So $\Delta W = BA = 0$ at the start — the model begins exactly at the pretrained weights.

4. **Scaling factor.** A constant $\alpha / r$ scales the update: $W' = W + (\alpha/r)BA$. This decouples the magnitude from the rank.

Typical LoRA configuration:
- Apply to $W^Q$ and $W^V$ (sometimes all four attention matrices).
- Rank $r \in \{4, 8, 16\}$.
- Total trainable parameters: $< 1\%$ of the model.

```python
# Using the peft library:
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 66,955,010 || trainable%: 0.44%
```

---

### 5.3 Prefix Tuning and Prompt Tuning

Instead of modifying weights, these methods prepend **learnable virtual tokens** to the input:

**Prompt tuning** (Lester et al., 2021): learn a sequence of soft embeddings $P \in \mathbb{R}^{k \times d_\text{model}}$ prepended to the input embeddings. Only $P$ is trained. Parameters: $k \times d_\text{model}$ (e.g., $20 \times 768 = 15{,}360$).

**Prefix tuning** (Li & Liang, 2021): prepend learnable key-value pairs to the attention layers at every layer. More expressive than prompt tuning because it injects information deeper into the network.

| Method | Where it acts | Params | Inference overhead |
|---|---|---|---|
| Prompt tuning | Input embeddings only | Very few ($\sim 15$K) | Slight (longer sequence) |
| Prefix tuning | KV pairs at every layer | More ($\sim 200$K) | Slight (longer sequence) |
| Adapters | Inside each layer | Moderate ($\sim 2$M) | Slight (extra computation) |
| LoRA | Weight matrices | Low ($\sim 300$K) | **None** after merging |

---

### 5.4 Comparison of PEFT Methods

| Method | Trainable params | Memory savings | Inference cost | Best for |
|---|---|---|---|---|
| **Full fine-tuning** | 100% | None | None | Large data, single task |
| **Feature extraction** | $< 1\%$ (head only) | High | None | Very small data, close domain |
| **Adapters** | 2–5% | Moderate | Small overhead | Multi-task serving |
| **LoRA** | 0.1–1% | High | None after merge | Default PEFT choice |
| **Prompt tuning** | $< 0.01\%$ | Highest | Slight | Extremely large models |

> **Notebook reference.** Exercise 3 asks you to implement adapter or partial-freeze strategies and compare parameter efficiency with full fine-tuning.

---

## 6. Feature Extraction in Detail

### 6.1 When and Why

Feature extraction is the **extreme** of transfer learning: the pretrained model is used purely as a function $f_\theta: x \mapsto z$ that maps inputs to fixed representations. A lightweight classifier is trained on $z$.

Use cases:
- **Very few labelled examples** (10–100): fine-tuning would overfit; frozen features + linear probe is stable.
- **Computational constraints:** no GPU for backprop through a large model.
- **Rapid prototyping:** extract features once, train many classifiers quickly.

---

### 6.2 Which Layer to Extract From

Not all layers are equally useful. In a Transformer:

| Extraction point | What it captures | Best for |
|---|---|---|
| **Last layer** | High-level, task-oriented | Close domain (most common) |
| **Second-to-last layer** | Slightly more general | Often better than last |
| **Concatenation of last 4 layers** | Multi-scale features | NER, token-level tasks |
| **All layers (learned weighted sum)** | Optimal combination | ELMo-style, if compute allows |

For CNNs: features before the final classification layer (after global average pooling) are standard.

```python
# Extract features from a specific layer in BERT
from transformers import AutoModel

model = AutoModel.from_pretrained('distilbert-base-uncased')
model.eval()

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    # outputs.hidden_states is a tuple of (num_layers + 1) tensors
    last_hidden = outputs.hidden_states[-1]          # last layer
    second_last = outputs.hidden_states[-2]          # second-to-last
    pooled = last_hidden.mean(dim=1)                 # mean pooling → (B, d)
```

---

## 7. Fine-Tuning Pretrained Transformers (HuggingFace Workflow)

### 7.1 Loading a Pretrained Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
```

- `AutoModelForSequenceClassification` adds a classification head on top of the pretrained backbone.
- `num_labels` determines the output dimension: 2 for binary, $C$ for multi-class.
- The backbone weights come from pre-training; the classification head is randomly initialised.

---

### 7.2 Tokenisation

HuggingFace tokenisers handle subword tokenisation (WordPiece for BERT, BPE for GPT):

```python
encoded = tokenizer(
    "Transfer learning is powerful.",
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
# encoded['input_ids']:     [101, 4651, 4083, 2003, 3928, 1012, 102, 0, 0, ...]
# encoded['attention_mask']: [1,   1,    1,    1,    1,    1,    1,   0, 0, ...]
```

Important details:
- **[CLS]** (token 101) and **[SEP]** (token 102) are special tokens BERT expects.
- **Padding** (token 0) fills sequences to uniform length; the attention mask ensures pad tokens are ignored.
- **Truncation** clips sequences longer than `max_length`.
- The tokeniser must match the pretrained model — using a different tokeniser corrupts the input.

---

### 7.3 TrainingArguments and Trainer

HuggingFace's `Trainer` abstracts the training loop:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,             # 6% of steps for warm-up
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_steps=50,
    fp16=True,                     # mixed precision
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
```

Key parameters:
- `learning_rate=2e-5`: standard for fine-tuning.
- `weight_decay=0.01`: AdamW decoupled weight decay ([Week 16](../../05_deep_learning/week16_regularization_dl/theory.md) regularisation).
- `warmup_ratio=0.06`: linear warm-up for 6% of total training steps.
- `fp16=True`: mixed precision for faster training and lower memory ([Week 14](../../05_deep_learning/week14_training_at_scale/theory.md)).
- `load_best_model_at_end=True`: early stopping via best checkpoint.

---

## 8. Fine-Tuning for Vision (Transfer from CNNs)

The same principles apply to vision models. Typical workflow with torchvision:

```python
import torchvision.models as models

# Load pretrained ResNet-50
model = models.resnet50(weights='IMAGENET1K_V2')

# Strategy 1: Feature extraction
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)   # new head

# Strategy 2: Full fine-tuning
model.fc = nn.Linear(2048, num_classes)   # new head
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
```

**Vision-specific considerations:**

1. **Input normalisation.** Pretrained models expect inputs normalised with ImageNet mean/std: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$. Using different normalisation **breaks transfer**.

2. **Resolution.** Most pretrained CNNs expect $224 \times 224$. Fine-tuning at higher resolution can help (features are more detailed) but requires more memory.

3. **Data augmentation.** Critical for small datasets — random crops, flips, colour jitter, RandAugment. Connections to [Week 16](../../05_deep_learning/week16_regularization_dl/theory.md) (augmentation as regularisation).

4. **Layer freezing pattern.** CNNs have a natural hierarchy:
   - Early layers: edges, textures → very generic, rarely need updating.
   - Middle layers: parts, shapes → somewhat generic.
   - Late layers: object categories → task-specific, benefit from fine-tuning.

---

## 9. Catastrophic Forgetting

When fine-tuning overwrites pretrained representations, the model "forgets" what it learned during pre-training. This manifests as:

1. **Poor generalisation** — the model overfits to the fine-tuning data.
2. **Loss of capability** — if you evaluate on the original pre-training task, performance drops dramatically.

**Causes:**
- Learning rate too high.
- Too many fine-tuning epochs.
- Target dataset is small and not representative.

**Mitigation strategies:**

| Strategy | Mechanism |
|---|---|
| Small learning rate | Limits parameter movement from pretrained basin |
| Warm-up | Lets random head stabilise before updating backbone |
| Gradual unfreezing | Protects early layers from early noisy gradients |
| Weight decay | Regularises parameters toward zero (implicit constraint) |
| LoRA | Only modifies a low-rank subspace; backbone stays intact |
| EWC (Elastic Weight Consolidation) | Penalises changes to parameters important for prior tasks: $\mathcal{L}_\text{EWC} = \mathcal{L}_\text{task} + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_{\text{pre},i})^2$ where $F_i$ is the Fisher information |
| Replay | Mix in a small fraction of pre-training data during fine-tuning |

For standard fine-tuning (single downstream task, few epochs, small LR), catastrophic forgetting is manageable. It becomes a serious concern in **continual learning** (sequentially adapting to many tasks).

---

## 10. Evaluation and Data Considerations

### 10.1 Small-Data Regimes

With $< 1{,}000$ labelled examples:

- **Variance is high.** Results change significantly across random seeds and train/val splits.
- **Use k-fold cross-validation** (e.g., $k = 5$) with reported mean ± std.
- **Augmentation is critical** — both standard (crop, flip) and advanced (back-translation for NLP, mixup).
- **Prefer simpler adaptation** — feature extraction or LoRA over full fine-tuning.

---

### 10.2 Class Imbalance

Target datasets are often imbalanced. Standard mitigations:

1. **Weighted loss:**

$$\mathcal{L} = -\sum_{i=1}^{N}w_{y_i}\log\hat{p}(y_i \mid x_i), \qquad w_c = \frac{N}{C \cdot n_c}$$

where $n_c$ is the number of samples in class $c$.

2. **Oversampling** the minority class (or undersampling the majority).
3. **Stratified splits** to ensure class proportions are preserved in train/val/test.

---

### 10.3 Metrics Beyond Accuracy

For imbalanced or nuanced tasks, accuracy alone is insufficient:

| Metric | Formula | When to use |
|---|---|---|
| **Precision** | $\text{TP} / (\text{TP} + \text{FP})$ | When false positives are costly |
| **Recall** | $\text{TP} / (\text{TP} + \text{FN})$ | When false negatives are costly |
| **F1 score** | $2 \cdot \text{Precision} \cdot \text{Recall} / (\text{Precision} + \text{Recall})$ | Balanced summary |
| **Macro F1** | Average F1 across classes (equal weight per class) | Imbalanced multi-class |
| **AUROC** | Area under the ROC curve | Ranking quality; threshold-independent |
| **Matthews Correlation** | $(TP \cdot TN - FP \cdot FN) / \sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}$ | Balanced metric even with imbalance |

---

## 11. Connections to the Rest of the Course

| Week | Connection |
|---|---|
| **[Week 06](../../02_fundamentals/week06_regularization/theory.md) (Regularisation)** | Freezing layers is a form of parameter regularisation; weight decay prevents fine-tuning from straying too far from pretrained weights |
| **[Week 12](../../04_neural_networks/week12_training_pathologies/theory.md) (Training Pathologies)** | Catastrophic forgetting is a training pathology; residual connections in the backbone enable effective gradient flow during fine-tuning |
| **[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md) (Training at Scale)** | Mixed precision (`fp16`) and gradient accumulation are essential for fine-tuning large models on limited hardware |
| **[Week 15](../../05_deep_learning/week15_cnn_representations/theory.md) (CNNs)** | Vision transfer learning uses the same CNN architectures; feature extraction from intermediate layers |
| **[Week 16](../../05_deep_learning/week16_regularization_dl/theory.md) (Regularisation DL)** | Dropout, weight decay, augmentation — all regularisation tools apply during fine-tuning, especially with small data |
| **[Week 17](../../06_sequence_models/week17_attention/theory.md) (Attention)** | LoRA modifies attention weight matrices ($W^Q$, $W^V$); understanding the Q/K/V projections lets you choose *which* matrices to adapt |
| **[Week 18](../../06_sequence_models/week18_transformers/theory.md) (Transformers)** | The Transformer is the model being fine-tuned; knowing the architecture (embedding, LayerNorm, FFN, multi-head attention) lets you make informed decisions about *what* to freeze and *what* to update |
| **[Week 20](../../08_deployment/week20_deployment/theory.md) (Deployment)** | The fine-tuned model is your deployment candidate; PEFT methods like LoRA enable serving multiple task-specific models from a single backbone |

---

## 12. Notebook Reference Guide

| Cell | Section | What it demonstrates | Theory reference |
|---|---|---|---|
| 1 (Setup) | Imports | HuggingFace Transformers, AutoModel, AutoTokenizer, cache utilities | Section 7 |
| 2 (Exercise 1) | Quick fine-tune | Skeleton: load DistilBERT, tokenise dataset, TrainingArguments, Trainer, train & eval | Sections 4, 7 |
| 3 (Exercise 2) | Feature extraction vs. fine-tuning | Freeze backbone (`requires_grad = False`), train head only; compare with full fine-tuning; metrics table | Sections 3.1, 3.2, 6 |
| 4 (Exercise 3) | PEFT methods | Adapter or partial-freeze strategies; parameter count comparison | Section 5 |
| Deliverables | Checklist | Fine-tuning run, comparison table, capstone recommendation | — |

**Suggested modifications:**

| Modification | What it reveals |
|---|---|
| Fine-tune with $\text{lr} = 10^{-3}$ (100× too high) and plot validation loss | Catastrophic forgetting: loss may decrease initially then spike as pretrained features are destroyed |
| Compare 1 epoch vs. 3 epochs vs. 10 epochs of fine-tuning | Fine-tuning saturates quickly; 10 epochs likely overfits on small data |
| Implement discriminative learning rates ($\eta = 0.95$ decay per layer) | May improve accuracy by 0.5–1% on tasks with moderate domain gap |
| Train a linear probe on each layer separately and plot accuracy vs. layer | Reveals which layers are most useful for the target task; usually peak at layer $L-1$ or $L-2$ |
| Apply LoRA with ranks $r \in \{1, 4, 8, 16, 64\}$ and plot accuracy vs. rank | Diminishing returns after $r \approx 8$ for most tasks; $r = 1$ is surprisingly competitive |
| Fine-tune on 50, 200, 1000, 5000 samples and plot accuracy vs. data size | Learning curves: feature extraction dominates at 50; fine-tuning catches up at 1000; gap closes at 5000 |
| Compare DistilBERT (66M) vs. BERT-base (110M) fine-tuning | DistilBERT is 40% smaller and 60% faster with < 3% accuracy drop on most tasks |
| Fine-tune, then evaluate on the original MLM task | Quantifies catastrophic forgetting: MLM perplexity increases dramatically after fine-tuning |

---

## 13. Symbol Reference

| Symbol | Name | Meaning |
|---|---|---|
| $\theta_\text{pre}$ | Pretrained parameters | Model weights after pre-training |
| $\theta_\text{fine}$ | Fine-tuned parameters | Model weights after adaptation |
| $\mathcal{D}_\text{target}$ | Target dataset | Labelled data for the downstream task |
| $\mathcal{P}_\text{source}$ | Source distribution | Data distribution seen during pre-training |
| $\mathcal{P}_\text{target}$ | Target distribution | Data distribution of the downstream task |
| $W$ | Frozen weight matrix | Original pretrained weights (in LoRA context) |
| $\Delta W = BA$ | LoRA update | Low-rank perturbation to $W$ |
| $A$ | LoRA down-projection | $\mathbb{R}^{r \times d_\text{in}}$; trainable |
| $B$ | LoRA up-projection | $\mathbb{R}^{d_\text{out} \times r}$; trainable, initialised to 0 |
| $r$ | LoRA rank / adapter bottleneck | Low-rank dimension; typically 4–16 |
| $\alpha$ | LoRA scaling factor | Scales the update: $W' = W + (\alpha/r)BA$ |
| $W_\text{down}, W_\text{up}$ | Adapter projections | Down-project to $r$, up-project back to $d_\text{model}$ |
| $\eta$ | Discriminative LR decay | Per-layer learning rate multiplier; $\eta < 1$ |
| $F_i$ | Fisher information | Importance of parameter $i$ for prior task (used in EWC) |
| $\text{lr}_\text{base}$ | Base learning rate | Learning rate for the top layer / classifier head |

---

## 14. References

1. Howard, J. & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *ACL*. — ULMFiT: gradual unfreezing, discriminative learning rates, slanted triangular LR schedule.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*. — Established the pretrain → fine-tune paradigm for NLP.
3. Hu, E. J., Shen, Y., Wallis, P., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*. — The LoRA method.
4. Houlsby, N., Giber, A., Jastrzebski, S., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." *ICML*. — Adapter modules.
5. Li, X. L. & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." *ACL*. — Prefix tuning.
6. Lester, B., Al-Rfou, R., & Constant, N. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." *EMNLP*. — Prompt tuning.
7. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." *PNAS*. — Elastic Weight Consolidation (EWC).
8. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). "How Transferable Are Features in Deep Neural Networks?" *NeurIPS*. — Seminal study: early layers are general, later layers are task-specific.
9. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*. — ResNet: the standard vision backbone for transfer learning.
10. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). "DistilBERT, a Distilled Version of BERT." *NeurIPS Workshop*. — Knowledge distillation for efficient fine-tuning.
11. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." *NeurIPS*. — LoRA on 4-bit quantised models; fine-tuning 65B models on a single GPU.
12. Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." *EMNLP Demo*. — The HuggingFace Transformers library.
