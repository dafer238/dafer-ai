# Week 00 — The AI Landscape: What, Why, and How

## Overview

Before touching a single equation, step back and map the territory. This week answers the question nobody asks clearly enough: _what actually is machine learning, and how do all the pieces relate?_ You will learn to categorize problems, match them to algorithmic families, and understand the vocabulary that every week from Week 01 forward takes for granted.

No code is required this week. The deliverable is a **mental model** you can draw on a whiteboard — and the habit of asking "what family does this problem belong to?" before reaching for a tool.

---

## The Core Idea

Classical programming:

```
rules + data → answers
```

Machine learning:

```
data + answers → rules (learned automatically)
```

The rules are learned by defining what "wrong" looks like (a **loss function**), and adjusting the program's internal numbers (**parameters**) to be less wrong. That adjustment process is **optimization**, the subject of Week 01. Everything else in this course builds on this loop.

---

## The Three Learning Paradigms

### Supervised Learning

You provide labelled pairs `(input, correct output)`. The model learns to map inputs to outputs.

| Task                      | Input                  | Output           | Example algorithm                       |
| ------------------------- | ---------------------- | ---------------- | --------------------------------------- |
| **Regression**            | numbers                | a real number    | Linear regression, neural network       |
| **Classification**        | numbers / text / image | a category label | Logistic regression, SVM, decision tree |
| **Ranking**               | query + candidates     | ordered list     | LambdaRank                              |
| **Structured prediction** | sequence               | sequence         | Transformer                             |

Applications you will build in this course:
- Energy demand forecasting (regression) — Week 01–04
- Regime detection (classification) — Week 09–10
- Fault detection from sensor windows (classification) — Week 11–12
- Time-series price forecasting (structured prediction) — Week 13–14

**Mental model:** the model is a student; labelled data is a teacher marking its homework.

---

### Unsupervised Learning

You only have inputs. No labels. The goal is to find **structure, patterns, or a compact description** of the data.

| Task                         | Goal                                | Example algorithm                            |
| ---------------------------- | ----------------------------------- | -------------------------------------------- |
| **Clustering**               | Group similar points                | K-Means, DBSCAN, Gaussian Mixture Models     |
| **Dimensionality reduction** | Compress while preserving structure | PCA, t-SNE, UMAP, autoencoders               |
| **Density estimation**       | Model the distribution of the data  | Kernel density estimation, normalizing flows |
| **Anomaly detection**        | Find points that don't fit          | Isolation Forest, autoencoders               |
| **Generative modelling**     | Learn to produce new samples        | VAEs, GANs, diffusion models                 |

Applications in this course:
- Sensor anomaly detection — Week 11–12
- Representation learning — Week 11

**Mental model:** the model is a scientist given a pile of unlabelled specimens; it has to discover categories by itself.

---

### Reinforcement Learning (not covered this course, but placed on the map)

An **agent** takes **actions** in an **environment** and receives **rewards**. It learns a **policy** — a rule for choosing actions — to maximize cumulative reward.

Applications: robotics, game playing (AlphaGo), algorithmic trading, HVAC control.

Mention it here because it completes the map, but it requires a different mathematical framework. The concepts from this course (function approximation via neural networks, optimization) are prerequisites.

---

## The Algorithmic Family Tree

```
Machine Learning
├── Supervised
│   ├── Linear models (Week 03)
│   │   ├── Linear regression
│   │   └── Logistic regression
│   ├── Tree-based models (not in this course — sklearn docs cover them)
│   ├── Neural networks (Week 07–16)
│   │   ├── Fully connected (MLP)
│   │   ├── Convolutional (CNN) — Week 11
│   │   ├── Recurrent (RNN) — Week 13
│   │   └── Transformer — Week 14
│   └── Kernel methods / SVM (not in this course, but conceptually adjacent to Week 03–04)
│
├── Unsupervised
│   ├── Clustering (K-Means, GMM)
│   ├── Dimensionality reduction (PCA — connects to Week 04 via SVD)
│   └── Generative models (briefly in Week 06 via density estimation)
│
└── Reinforcement Learning (not in this course)
```

---

## The Training Loop (a preview)

Every supervised algorithm in this course follows this loop:

1. **Forward pass** — feed an input through the model, get a prediction.
2. **Loss computation** — measure how wrong the prediction is (Week 05 explains where loss functions come from probabilistically).
3. **Backward pass** — compute how each parameter contributed to the error (this is the gradient — Week 01/07 build this from scratch).
4. **Parameter update** — adjust parameters to reduce the error (gradient descent — the entire subject of Week 01–02).
5. **Repeat** until the model is good enough.

This loop is the same whether you are fitting a 2-parameter linear model or a 70-billion-parameter language model. **The scale changes; the structure does not.**

---

## What is a Model?

A model is a **parameterized function**.

- Linear regression: $\hat{y} = w_1 x_1 + w_2 x_2 + b$ — parameters are $\mathbf{w}$ and $b$.
- Neural network: a composition of many such functions with nonlinearities in between — parameters are millions of weights.

The art of machine learning is choosing:
1. The **architecture** (what function family to use)
2. The **loss** (what "wrong" means)
3. The **optimizer** (how to adjust parameters)
4. The **regularization** (how to prevent over-memorizing training data)

These four choices structure this entire course.

---

## Key Vocabulary

| Term                     | Plain definition                                                   | First used |
| ------------------------ | ------------------------------------------------------------------ | ---------- |
| **Feature / input**      | A measurable property of an example                                | Week 03    |
| **Label / target**       | The value we want to predict                                       | Week 03    |
| **Parameter / weight**   | A number the model learns                                          | Week 01    |
| **Hyperparameter**       | A setting chosen by the practitioner (learning rate, depth)        | Week 01    |
| **Loss / cost function** | A scalar measuring prediction error                                | Week 01    |
| **Gradient**             | Derivative of loss w.r.t. parameters; direction of steepest ascent | Week 01    |
| **Training set**         | Data used to learn parameters                                      | Week 03    |
| **Validation set**       | Data used to tune hyperparameters                                  | Week 04    |
| **Test set**             | Data used once, at the end, to estimate real-world performance     | Week 04    |
| **Overfitting**          | Model memorizes training data; fails on new data                   | Week 03–04 |
| **Underfitting**         | Model too simple; fails even on training data                      | Week 03    |
| **Epoch**                | One full pass over the training dataset                            | Week 01    |
| **Batch**                | A subset of the training set used for one update                   | Week 02    |
| **Inference**            | Using a trained model to make predictions on new inputs            | Week 16    |

---

## The Evaluation Mindset

Always ask three questions about any model:

1. **Does it fit the training data well?** (if not: underfitting — wrong architecture or loss)
2. **Does it generalize to unseen data?** (if not: overfitting — add data, regularization, or reduce complexity)
3. **Is the metric aligned with the actual problem?** (accuracy on imbalanced classes can be 99% while the model is useless)

Weeks 03 (bias-variance) and 04 (validation) formalize this. Keep it in your head from now on.

---

## The Landscape of Tools (used in this course)

| Tool                     | Role                                          | First used |
| ------------------------ | --------------------------------------------- | ---------- |
| NumPy                    | Numerical computing, array operations         | Week 01    |
| Matplotlib / Seaborn     | Visualization                                 | Week 01    |
| SciPy                    | Scientific computing, statistical tools       | Week 05    |
| scikit-learn             | Baseline ML models, preprocessing, evaluation | Week 03    |
| PyTorch                  | Deep learning framework, autograd             | Week 09    |
| HuggingFace Transformers | Pretrained models, fine-tuning                | Week 15    |
| FastAPI                  | Model serving                                 | Week 16    |

---

## Reading for this Week

- **Chapter 1** of Bishop's _Pattern Recognition and Machine Learning_ (PRML) — a map of the field.
- **Chapter 1** of Goodfellow et al. _Deep Learning_ (free online) — covers supervised, unsupervised, reinforcement.
- _The Hundred-Page Machine Learning Book_ by Andriy Burkov — Chapters 1–3 (concise and dense, excellent reference).
- Optional: 3Blue1Brown's "Neural Networks" series on YouTube for visual intuition (no equations needed yet).

---

## Deliverable

Write a one-page (or one-diagram) **map of ML** that shows:
- The three paradigms with two examples each
- Where regression, classification, and clustering live
- The four choices every practitioner makes (architecture, loss, optimizer, regularization)

This is not graded. It is a forcing function to make sure you have internalized the territory before entering it.

---

## Connection to Week 01

Week 01 starts immediately with gradient descent and loss surfaces. It assumes you know:
- What a loss function is and why we minimize it ✓ (defined above)
- What parameters are ✓ (defined above)
- That the goal is to adjust parameters to reduce loss ✓ (the training loop above)

If anything in that list feels uncertain, re-read the sections above before proceeding.
