# Week 00b — Math and Data Foundations

## Overview

Week 01 will ask you to compute gradients, visualize loss surfaces, and implement a learning loop in NumPy. This week closes any remaining gap between your programming background and the mathematical machinery you will need. Nothing here requires a university course — you need comfort, not mastery. The goal is that Week 01's equations look familiar, not foreign.

By the end of this week you should be able to:
- Work confidently with vectors and matrices in NumPy
- Understand what a derivative means geometrically and compute simple ones by hand
- Describe what a probability distribution is and what "fitting a distribution to data" means
- Handle a real dataset: load, inspect, clean, and do basic exploratory analysis

---

## Part 1 — Linear Algebra for ML

You do not need to prove theorems. You need to _interpret_ what operations mean.

### Vectors

A vector is a list of numbers. In ML, it is almost always a **data point** (a row of your dataset) or a **parameter vector** (the weights of a model).

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \in \mathbb{R}^3$$

The **dot product** measures similarity:

$$\mathbf{w} \cdot \mathbf{x} = w_1 x_1 + w_2 x_2 + w_3 x_3$$

Linear regression prediction for one sample: $\hat{y} = \mathbf{w} \cdot \mathbf{x} + b$. That's it.

The **L2 norm** (Euclidean length):

$$\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2 + x_3^2}$$

This will appear as a regularization penalty in Week 04.

### Matrices

A matrix is a grid of numbers. In ML it is usually a **dataset** (rows = samples, columns = features).

$$X \in \mathbb{R}^{n \times d}: \quad n \text{ samples}, \; d \text{ features}$$

**Matrix multiply** $X\mathbf{w}$ simultaneously computes the linear prediction for all $n$ samples — this is why vectorization is fast.

```python
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])   # 3 samples, 2 features
w = np.array([0.5, -0.3])                  # 2 weights
predictions = X @ w                        # shape (3,) — one number per sample
```

### What you must be able to do

| Operation        | NumPy syntax              | Used in                                   |
| ---------------- | ------------------------- | ----------------------------------------- |
| Dot product      | `a @ b` or `np.dot(a, b)` | Every forward pass (Week 01+)             |
| Matrix multiply  | `A @ B`                   | Linear regression (Week 03), NN (Week 07) |
| Transpose        | `A.T`                     | Normal equations (Week 03)                |
| Element-wise ops | `a * b`, `a ** 2`         | Loss computation (Week 01)                |
| Sum / mean       | `np.sum(x)`, `np.mean(x)` | Loss computation (Week 01)                |
| Reshape          | `x.reshape(n, 1)`         | Broadcasting fixes (every week)           |
| Broadcasting     | implicit                  | Essential for vectorized code             |

**Exercise:** implement mean squared error without a loop:

```python
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 1.8, 3.3])
mse = np.mean((y_true - y_pred) ** 2)   # should be ~0.047
```

---

## Part 2 — Calculus and Derivatives

### What a derivative means

The derivative $\frac{df}{dx}$ tells you: _if I increase $x$ by a tiny amount $\epsilon$, how much does $f$ change?_

$$\frac{df}{dx} \approx \frac{f(x + \epsilon) - f(x)}{\epsilon} \quad \text{(finite difference approximation)}$$

For $f(x) = x^2$: the derivative is $\frac{df}{dx} = 2x$. At $x = 3$, the derivative is $6$ — a small step right makes $f$ grow by approximately $6 \times$ that step.

**Geometric interpretation:** the derivative is the slope of the tangent line to the curve at a point.

In optimization: **the gradient points in the direction of steepest increase**. So to minimize $f$, you step in the **negative** gradient direction. This is the entire idea of gradient descent (Week 01).

### Partial derivatives

When $f$ depends on multiple variables, the **partial derivative** $\frac{\partial f}{\partial w_i}$ treats all variables except $w_i$ as constants.

$$f(w_1, w_2) = w_1^2 + 3 w_1 w_2$$

$$\frac{\partial f}{\partial w_1} = 2w_1 + 3w_2, \qquad \frac{\partial f}{\partial w_2} = 3w_1$$

The **gradient** is the vector of all partial derivatives:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial w_1} \\ \frac{\partial f}{\partial w_2} \end{bmatrix}$$

### The chain rule

If $f = g(h(x))$, then $\frac{df}{dx} = \frac{dg}{dh} \cdot \frac{dh}{dx}$.

This is the mathematical engine behind backpropagation (Week 07). Every layer in a neural network is one step of a chain rule application.

**Simple example:** $f = (w x - y)^2$. Let $u = wx - y$, so $f = u^2$.

$$\frac{df}{dw} = \frac{df}{du} \cdot \frac{du}{dw} = 2u \cdot x = 2(wx - y) \cdot x$$

You will derive this again in Week 03 for linear regression. Having seen it here removes the cognitive load.

### Rules to know by memory

| Function                         | Derivative                 |
| -------------------------------- | -------------------------- |
| $x^n$                            | $n x^{n-1}$                |
| $e^x$                            | $e^x$                      |
| $\ln x$                          | $1/x$                      |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ |
| $\max(0, x)$ (ReLU)              | $1$ if $x > 0$, else $0$   |

---

## Part 3 — Probability and Statistics

### Why probability matters in ML

Every ML model makes an implicit assumption about noise and uncertainty. Understanding probability lets you:
- Know where loss functions come from (Week 05)
- Know what "overfitting" means statistically (Week 04)
- Reason about prediction intervals (Week 06)

### Core concepts

**Random variable:** a variable whose value is determined by a random process. E.g., the true energy demand tomorrow.

**Probability distribution:** a function describing how likely each value is.
- Discrete: $P(X = k)$, sums to 1.
- Continuous: probability density function (PDF) $p(x)$, integrates to 1.

**Gaussian (Normal) distribution:**

$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Parameters: $\mu$ (mean, center) and $\sigma$ (standard deviation, spread). This is the most important distribution in this course.

**Expected value:** average value over the distribution.

$$E[X] = \int x \, p(x) \, dx \approx \frac{1}{n}\sum_{i=1}^n x_i$$

**Variance and standard deviation:** spread of the distribution.

$$\text{Var}(X) = E[(X - E[X])^2], \qquad \sigma = \sqrt{\text{Var}(X)}$$

### From probability to loss functions (preview)

Assuming that observations are drawn from a Gaussian centered at the model's prediction:

$$y_i = \hat{y}_i + \epsilon_i, \qquad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

Maximizing the likelihood of observing the data under this model (Week 05) leads directly to minimizing:

$$L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Mean squared error is Gaussian maximum likelihood.** This connection unlocks Week 05. Keep it in mind from Week 01 onwards.

### Practical: working with distributions in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# sample from a Gaussian
rng = np.random.default_rng(42)
data = rng.normal(loc=2.0, scale=1.5, size=1000)

# estimate parameters
print(data.mean(), data.std())   # should be close to 2.0, 1.5

# plot histogram + fitted PDF
x = np.linspace(-3, 7, 200)
plt.hist(data, bins=30, density=True, alpha=0.6)
plt.plot(x, stats.norm.pdf(x, data.mean(), data.std()))
plt.show()
```

---

## Part 4 — Exploratory Data Analysis (EDA)

Before training any model, understand the data. This habit prevents many silent failures.

### The EDA checklist

1. **Shape and types** — `df.shape`, `df.dtypes`. Know what you have.
2. **Missing values** — `df.isnull().sum()`. Decide: drop, impute, or flag.
3. **Distributions** — histogram each numeric feature. Note skew, outliers, bimodal shapes.
4. **Correlations** — `df.corr()`, heatmap. Which features move together?
5. **Target distribution** — is it skewed? Are there class imbalances? This determines your loss function.
6. **Time structure** — if there is a time index, plot the series. Seasonality, trends, and regime changes are invisible in aggregate statistics.

### The data-generating process

Always ask: _where did this data come from, and under what conditions?_ This determines:
- Whether train/test split is valid (random split invalid for time series — Week 04)
- Whether features would actually be available at prediction time (leakage)
- Whether the noise model is Gaussian or something heavier-tailed

### Practical: loading and inspecting a dataset

```python
import pandas as pd
import seaborn as sns

# load UCI energy efficiency dataset (used conceptually in Week 01)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
df = pd.read_excel(url)

print(df.head())
print(df.describe())
print(df.isnull().sum())

sns.pairplot(df.iloc[:, :5])   # scatter matrix of first 5 columns
```

---

## Part 5 — The NumPy Mindset

NumPy is the bedrock of every week in this course. Two habits make the difference:

### Think in shapes

Before writing any code, know the shape of every array: `(n,)`, `(n, 1)`, `(n, d)`, `(d, d)`. Shape errors are the most common bug in ML code.

```python
x = np.array([1, 2, 3])       # shape (3,)  — 1D array
x = x.reshape(3, 1)           # shape (3,1) — column vector
X = np.ones((5, 3))           # shape (5,3) — 5 samples, 3 features
```

### Avoid Python loops over samples

```python
# BAD: O(n) Python overhead
mse = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true))) / len(y_true)

# GOOD: vectorized
mse = np.mean((y_true - y_pred) ** 2)
```

Vectorized code is 10–1000× faster and cleaner. Build the habit now.

---

## Exercises

1. **Linear algebra warmup**
   - Generate a random `(100, 5)` matrix `X` and a random `(5,)` weight vector `w`.
   - Compute predictions `y_hat = X @ w`.
   - Add Gaussian noise to get `y`.
   - Compute MSE manually.

2. **Derivative by finite difference**
   - For $f(x) = x^3 - 2x$, compute the analytic derivative and verify it numerically at $x = 1.5$ using `(f(x+1e-5) - f(x)) / 1e-5`.

3. **Gaussian fitting**
   - Generate 500 samples from a Gaussian with $\mu = 5$, $\sigma = 2$.
   - Estimate $\mu$ and $\sigma$ from the samples.
   - Plot histogram with fitted PDF.

4. **EDA on a real dataset**
   - Load the `sklearn.datasets.fetch_california_housing` dataset.
   - Inspect shape, check for missing values, plot the target distribution, and compute the feature–target correlation matrix.

---

## Reading

- **NumPy Quickstart** — [numpy.org/doc/stable/user/quickstart.html](https://numpy.org/doc/stable/user/quickstart.html)
- **3Blue1Brown — Essence of Linear Algebra** (YouTube) — watch chapters 1–5 (vectors, linear transforms, dot product)
- **3Blue1Brown — Essence of Calculus** (YouTube) — watch chapters 1–4 (derivative, chain rule)
- **Khan Academy — Probability & Statistics** — "Random variables" and "Normal distributions" sections
- **Python Data Science Handbook**, Jake VanderPlas — Chapter 2 (NumPy) and Chapter 3 (pandas)

---

## Deliverable

A short notebook (or script) demonstrating:
- Matrix operations and shape management in NumPy
- A finite-differences derivative check for a function of your choice
- A sampled + fitted Gaussian plot
- EDA on the California housing dataset with at least 3 observations about the data

This is the foundation. Week 01 will assume fluency with all of it.

---

## Connection to Week 01

Week 01 opens with loss landscapes and gradient descent. When you see:

$$w \leftarrow w - \eta \cdot \nabla_w L(w)$$

you will read it as: _adjust each weight in the direction that reduces the loss, by a step proportional to how sensitive the loss is to that weight._ Every term in that equation is defined in this week.
