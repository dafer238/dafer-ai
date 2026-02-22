# Probability & Noise: Maximum Likelihood Estimation

## Table of Contents

- [Probability \& Noise: Maximum Likelihood Estimation](#probability--noise-maximum-likelihood-estimation)
  - [Table of Contents](#table-of-contents)
  - [1. Scope and Purpose](#1-scope-and-purpose)
  - [2. Probability Foundations: The Language of Uncertainty](#2-probability-foundations-the-language-of-uncertainty)
    - [2.1 Random Variables and Distributions](#21-random-variables-and-distributions)
    - [2.2 Joint, Marginal, and Conditional Distributions](#22-joint-marginal-and-conditional-distributions)
    - [2.3 Bayes' Theorem](#23-bayes-theorem)
    - [2.4 Key Distributions for This Week](#24-key-distributions-for-this-week)
      - [Gaussian (Normal)](#gaussian-normal)
      - [Laplace (Double Exponential)](#laplace-double-exponential)
      - [Student-_t_](#student-t)
      - [Bernoulli](#bernoulli)
      - [Poisson](#poisson)
  - [3. Likelihood: From Data to Models](#3-likelihood-from-data-to-models)
    - [3.1 The Likelihood Function](#31-the-likelihood-function)
    - [3.2 Log-Likelihood](#32-log-likelihood)
    - [3.3 Likelihood vs. Probability](#33-likelihood-vs-probability)
  - [4. Maximum Likelihood Estimation (MLE)](#4-maximum-likelihood-estimation-mle)
    - [4.1 The MLE Principle](#41-the-mle-principle)
    - [4.2 Properties of the MLE](#42-properties-of-the-mle)
    - [4.3 MLE for the Gaussian Distribution](#43-mle-for-the-gaussian-distribution)
  - [5. The MLE–Loss Function Connection](#5-the-mleloss-function-connection)
    - [5.1 Gaussian Noise → MSE](#51-gaussian-noise--mse)
    - [5.2 Bernoulli Noise → Binary Cross-Entropy](#52-bernoulli-noise--binary-cross-entropy)
    - [5.3 Poisson Noise → Poisson Deviance](#53-poisson-noise--poisson-deviance)
    - [5.4 The General Recipe](#54-the-general-recipe)
  - [6. MLE for Linear Regression (Full Derivation)](#6-mle-for-linear-regression-full-derivation)
    - [6.1 The Probabilistic Model](#61-the-probabilistic-model)
    - [6.2 Deriving the Log-Likelihood](#62-deriving-the-log-likelihood)
    - [6.3 MLE for $\\mathbf{w}$: Recovering OLS](#63-mle-for-mathbfw-recovering-ols)
    - [6.4 MLE for $\\sigma^2$](#64-mle-for-sigma2)
    - [6.5 Joint Optimisation in Practice](#65-joint-optimisation-in-practice)
  - [7. Robust Regression: Alternative Noise Models](#7-robust-regression-alternative-noise-models)
    - [7.1 Why Gaussian MLE Fails with Outliers](#71-why-gaussian-mle-fails-with-outliers)
    - [7.2 Laplace Noise → MAE (L1 Loss)](#72-laplace-noise--mae-l1-loss)
    - [7.3 Student-_t_ Noise → Heavy-Tailed Robustness](#73-student-t-noise--heavy-tailed-robustness)
    - [7.4 Huber Loss: A Practical Compromise](#74-huber-loss-a-practical-compromise)
    - [7.5 Comparison of Noise Models](#75-comparison-of-noise-models)
  - [8. Likelihood Surfaces and Optimisation](#8-likelihood-surfaces-and-optimisation)
    - [8.1 Convexity of the Gaussian NLL](#81-convexity-of-the-gaussian-nll)
    - [8.2 Non-Convex Likelihood Surfaces](#82-non-convex-likelihood-surfaces)
    - [8.3 Numerical Optimisation: scipy.optimize.minimize](#83-numerical-optimisation-scipyoptimizeminimize)
  - [9. From MLE to MAP: The Bridge to Regularisation](#9-from-mle-to-map-the-bridge-to-regularisation)
    - [9.1 Maximum A Posteriori Estimation](#91-maximum-a-posteriori-estimation)
    - [9.2 MAP = Regularised MLE](#92-map--regularised-mle)
  - [10. Model Selection: AIC and BIC](#10-model-selection-aic-and-bic)
  - [11. Connections to the Rest of the Course](#11-connections-to-the-rest-of-the-course)
  - [12. Notebook Reference Guide](#12-notebook-reference-guide)
  - [13. Symbol Reference](#13-symbol-reference)
  - [14. References](#14-references)

---

## 1. Scope and Purpose

[[Weeks 01](../../02_fundamentals/week01_optimization/theory.md)](../../02_fundamentals/week01_optimization/theory.md)–[06](../../02_fundamentals/week06_regularization/theory.md) took a **loss-function-first** approach: we defined MSE, minimised it, and observed that it works. This week answers the question _why_ it works — and when it does not.

The central insight is:

> **Every standard loss function is the negative log-likelihood of a probabilistic model.**

MSE corresponds to Gaussian noise. Binary cross-entropy corresponds to Bernoulli outcomes. Understanding this connection lets you:
1. **Choose** the right loss function by reasoning about the data-generating process.
2. **Design** new loss functions for non-standard problems (count data, heavy-tailed noise, heteroscedastic noise).
3. **Quantify uncertainty** in predictions ([Week 08](../week08_uncertainty/theory.md)).
4. **Justify regularisation** as Bayesian prior information (connecting back to [Week 06](../../02_fundamentals/week06_regularization/theory.md)).

This Week is the conceptual pivot of the entire course: everything before was optimisation of a given objective; everything after uses probability to _derive_ the objective.

**Prerequisites.** [Week 00b](../../01_intro/week00b_math_and_data/theory.md) (Gaussian distribution, preview of MLE), [Week 03](../../02_fundamentals/week03_linear_models/theory.md) (linear regression, MSE, normal equations), [Week 06](../../02_fundamentals/week06_regularization/theory.md) (Ridge as Bayesian MAP — the connection is completed here).

---

## 2. Probability Foundations: The Language of Uncertainty

### 2.1 Random Variables and Distributions

A **random variable** $Y$ is a quantity that can take different values with different probabilities. We write:
- $p(y)$ or $P(Y = y)$ for the **probability mass function (PMF)** (discrete $Y$).
- $f(y)$ or $p(y)$ for the **probability density function (PDF)** (continuous $Y$).

For a continuous random variable, $p(y)$ is not a probability — it is a **density**. The probability that $Y$ lies in an interval is:

$$P(a \leq Y \leq b) = \int_a^b p(y)\,dy$$

Key properties:
- $p(y) \geq 0$ for all $y$.
- $\int_{-\infty}^{\infty} p(y)\,dy = 1$ (normalisation).
- $p(y)$ can exceed 1 (it is a density, not a probability).

**Expectation** (mean):

$$\mathbb{E}[Y] = \int y\,p(y)\,dy$$

**Variance:**

$$\text{Var}[Y] = \mathbb{E}[(Y - \mathbb{E}[Y])^2] = \mathbb{E}[Y^2] - (\mathbb{E}[Y])^2$$

---

### 2.2 Joint, Marginal, and Conditional Distributions

For two random variables $X$ and $Y$:

| Concept          | Definition                           | Notation                       |
| ---------------- | ------------------------------------ | ------------------------------ |
| **Joint**        | Probability of both                  | $p(x, y)$                      |
| **Marginal**     | Integrate out the other              | $p(x) = \int p(x, y)\,dy$      |
| **Conditional**  | Given one, distribution of the other | $p(y \mid x) = p(x, y) / p(x)$ |
| **Independence** | Joint factorises                     | $p(x, y) = p(x)p(y)$           |

**The chain rule (product rule):**

$$p(x, y) = p(y \mid x)\,p(x) = p(x \mid y)\,p(y)$$

---

### 2.3 Bayes' Theorem

From the chain rule:

$$\boxed{p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})}}$$

| Term                         | Name                               | Role                                                                      |
| ---------------------------- | ---------------------------------- | ------------------------------------------------------------------------- |
| $p(\theta \mid \mathcal{D})$ | **Posterior**                      | Updated belief about $\theta$ after seeing data $\mathcal{D}$             |
| $p(\mathcal{D} \mid \theta)$ | **Likelihood**                     | How probable the observed data is under parameters $\theta$               |
| $p(\theta)$                  | **Prior**                          | Belief about $\theta$ before seeing data                                  |
| $p(\mathcal{D})$             | **Evidence (marginal likelihood)** | Normalising constant; $\int p(\mathcal{D} \mid \theta)p(\theta)\,d\theta$ |

Bayes' theorem is the foundation of Bayesian inference ([Week 08](../week08_uncertainty/theory.md)). For now, we focus on the **likelihood** $p(\mathcal{D} \mid \theta)$.

---

### 2.4 Key Distributions for This Week

#### Gaussian (Normal)

$$\mathcal{N}(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)$$

- **Mean:** $\mu$ (location).
- **Variance:** $\sigma^2$ (spread).
- **Symmetric,** thin tails, 95% of mass within $\mu \pm 1.96\sigma$.

#### Laplace (Double Exponential)

$$\text{Lap}(y \mid \mu, b) = \frac{1}{2b}\exp\!\left(-\frac{|y - \mu|}{b}\right)$$

- **Mean:** $\mu$.
- **Variance:** $2b^2$.
- **Sharper peak** at $\mu$ and **heavier tails** than Gaussian. The kink at $y = \mu$ (absolute value) connects to L1 loss (MAE) and Lasso ([Week 06](../../02_fundamentals/week06_regularization/theory.md)).

#### Student-_t_

$$t_\nu(y \mid \mu, \sigma) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\,\sigma}\left(1 + \frac{1}{\nu}\left(\frac{y - \mu}{\sigma}\right)^2\right)^{-(\nu+1)/2}$$

- **Degrees of freedom:** $\nu > 0$ (lower $\nu$ = heavier tails).
- $\nu \to \infty$: converges to Gaussian.
- $\nu = 1$: Cauchy distribution (so heavy-tailed that the mean does not exist).
- **Key property:** the $(\cdot)^{-(\nu+1)/2}$ power-law tail means outliers have bounded influence on the log-likelihood.

#### Bernoulli

$$\text{Ber}(y \mid p) = p^y(1-p)^{1-y}, \quad y \in \{0, 1\}$$

- **Mean:** $p$. **Variance:** $p(1-p)$.
- Used for binary classification (logistic regression, [Week 03](../../02_fundamentals/week03_linear_models/theory.md)).

#### Poisson

$$\text{Pois}(y \mid \lambda) = \frac{\lambda^y e^{-\lambda}}{y!}, \quad y \in \{0, 1, 2, \ldots\}$$

- **Mean = Variance:** $\lambda$.
- Used for count data (number of events in a fixed interval).

---

## 3. Likelihood: From Data to Models

### 3.1 The Likelihood Function

Given a dataset $\mathcal{D} = \{y_1, y_2, \ldots, y_n\}$ and a parametric model $p(y \mid \theta)$, the **likelihood** is the probability of the observed data as a function of the parameters:

$$\mathcal{L}(\theta) = p(\mathcal{D} \mid \theta)$$

If the data are **independent and identically distributed (i.i.d.)**:

$$\boxed{\mathcal{L}(\theta) = \prod_{i=1}^{n}p(y_i \mid \theta)}$$

> **Crucial distinction.** $p(y \mid \theta)$ and $\mathcal{L}(\theta)$ are the same mathematical expression, but viewed from different perspectives:
> - $p(y \mid \theta)$: fix $\theta$, vary $y$ → a probability distribution over data.
> - $\mathcal{L}(\theta)$: fix $y$ (the observed data), vary $\theta$ → a function of the parameters.

---

### 3.2 Log-Likelihood

Products of many small numbers cause numerical underflow. The **log-likelihood** converts products to sums:

$$\boxed{\ell(\theta) = \log\mathcal{L}(\theta) = \sum_{i=1}^{n}\log p(y_i \mid \theta)}$$

Properties:
- $\log$ is a **monotonically increasing** function, so maximising $\ell$ is equivalent to maximising $\mathcal{L}$.
- The sum structure makes differentiation straightforward (the gradient of a sum is the sum of gradients).
- In practice, we work exclusively with $\ell(\theta)$, never $\mathcal{L}(\theta)$.

---

### 3.3 Likelihood vs. Probability

| Aspect               | Probability                           | Likelihood                                                     |
| -------------------- | ------------------------------------- | -------------------------------------------------------------- |
| **Varies over**      | Data $y$                              | Parameters $\theta$                                            |
| **Fixed**            | $\theta$                              | Data $\mathcal{D}$                                             |
| **Integrates to 1?** | Yes ($\int p(y \mid \theta)\,dy = 1$) | **No** ($\int \mathcal{L}(\theta)\,d\theta \neq 1$ in general) |
| **Interpretation**   | "How probable is $y$?"                | "How well does $\theta$ explain the data?"                     |

The likelihood is **not** a probability distribution over $\theta$ (that would be the posterior $p(\theta \mid \mathcal{D})$, which requires Bayes' theorem and a prior — [Week 08](../week08_uncertainty/theory.md)).

---

## 4. Maximum Likelihood Estimation (MLE)

### 4.1 The MLE Principle

The **maximum likelihood estimator** (MLE) is the parameter value that maximises the likelihood (or, equivalently, the log-likelihood):

$$\boxed{\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta) = \arg\max_\theta \sum_{i=1}^{n}\log p(y_i \mid \theta)}$$

Equivalently, it minimises the **negative log-likelihood (NLL)**:

$$\hat{\theta}_{\text{MLE}} = \arg\min_\theta \left[-\ell(\theta)\right] = \arg\min_\theta \left[-\sum_{i=1}^{n}\log p(y_i \mid \theta)\right]$$

> **MLE as a loss function.** Minimising NLL is the same as minimising a loss function. This is the bridge between probabilistic modelling and optimisation-based ML.

---

### 4.2 Properties of the MLE

Under regularity conditions (smooth likelihood, identifiable model, data from the true distribution):

| Property                  | Statement                                                                                                                                                |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Consistency**           | $\hat{\theta}_{\text{MLE}} \xrightarrow{p} \theta^*$ as $n \to \infty$. The MLE converges to the true parameter.                                         |
| **Asymptotic normality**  | $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta^*) \xrightarrow{d} \mathcal{N}(\mathbf{0}, I(\theta^*)^{-1})$, where $I(\theta)$ is the Fisher information. |
| **Asymptotic efficiency** | The MLE achieves the **Cramér–Rao lower bound** asymptotically — no consistent estimator has smaller variance.                                           |
| **Equivariance**          | If $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ for any function $g$.                                         |
| **Bias**                  | The MLE is generally **biased** in finite samples (e.g., $\hat{\sigma}^2_{\text{MLE}}$ divides by $n$, not $n-1$). The bias vanishes as $n \to \infty$.  |

**Fisher Information:**

$$I(\theta) = -\mathbb{E}\!\left[\frac{\partial^2}{\partial\theta^2}\log p(Y \mid \theta)\right] = \mathbb{E}\!\left[\left(\frac{\partial}{\partial\theta}\log p(Y \mid \theta)\right)^2\right]$$

The Fisher information measures the **curvature** of the log-likelihood at the true parameter. Higher curvature = more information in the data = smaller variance of the MLE.

---

### 4.3 MLE for the Gaussian Distribution

Given i.i.d. samples $y_1, \ldots, y_n \sim \mathcal{N}(\mu, \sigma^2)$, find $\hat{\mu}$ and $\hat{\sigma}^2$:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mu)^2$$

**Setting $\partial\ell/\partial\mu = 0$:**

$$\frac{1}{\sigma^2}\sum_{i=1}^{n}(y_i - \mu) = 0 \implies \boxed{\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}y_i = \bar{y}}$$

The MLE of the mean is the sample mean (unbiased).

**Setting $\partial\ell/\partial\sigma^2 = 0$:**

$$-\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(y_i - \hat{\mu})^2 = 0 \implies \boxed{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

The MLE of the variance divides by $n$, not $n-1$. This is **biased** (underestimates true variance), but the bias is $\mathcal{O}(1/n)$ and vanishes asymptotically.

> The unbiased estimator $s^2 = \frac{1}{n-1}\sum(y_i - \bar{y})^2$ uses Bessel's correction. In ML with large $n$, the difference is negligible; in traditional statistics with small samples, $n-1$ is preferred.

---

## 5. The MLE–Loss Function Connection

This is the most important conceptual section of this week.

### 5.1 Gaussian Noise → MSE

**Model:** $y_i = f(\mathbf{x}_i; \theta) + \epsilon_i$, where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.

This means $y_i \mid \mathbf{x}_i \sim \mathcal{N}(f(\mathbf{x}_i; \theta), \sigma^2)$.

**NLL** (ignoring the constant $\frac{n}{2}\log(2\pi\sigma^2)$ and dividing by $n$):

$$-\frac{1}{n}\ell(\theta) = \frac{1}{2\sigma^2}\cdot\frac{1}{n}\sum_{i=1}^{n}(y_i - f(\mathbf{x}_i; \theta))^2 + \text{const}$$

Since $\sigma^2$ is a positive constant (it does not depend on $\theta$ for the purpose of optimising the model):

$$\boxed{\arg\min_\theta(-\ell) = \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n}(y_i - f(\mathbf{x}_i; \theta))^2 = \arg\min_\theta \text{MSE}}$$

**MSE is the MLE loss under Gaussian noise.**

---

### 5.2 Bernoulli Noise → Binary Cross-Entropy

**Model:** $y_i \in \{0, 1\}$, $p(y_i = 1 \mid \mathbf{x}_i) = \hat{p}_i = \sigma(f(\mathbf{x}_i; \theta))$, where $\sigma(\cdot)$ is the sigmoid.

**NLL:**

$$-\ell(\theta) = -\sum_{i=1}^{n}\left[y_i\log\hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]$$

$$\boxed{-\frac{1}{n}\ell = \text{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log\hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]}$$

**Binary cross-entropy is the MLE loss for logistic regression** ([Week 03](../../02_fundamentals/week03_linear_models/theory.md), Section 5.2).

---

### 5.3 Poisson Noise → Poisson Deviance

**Model:** $y_i \in \{0, 1, 2, \ldots\}$, $y_i \sim \text{Pois}(\lambda_i)$, where $\lambda_i = \exp(f(\mathbf{x}_i; \theta))$ (the $\exp$ ensures $\lambda_i > 0$).

**NLL:**

$$-\ell(\theta) = \sum_{i=1}^{n}\left[\lambda_i - y_i\log\lambda_i + \log(y_i!)\right]$$

Dropping the $\log(y_i!)$ term (independent of $\theta$):

$$\boxed{-\ell \propto \sum_{i=1}^{n}\left[\exp(f(\mathbf{x}_i; \theta)) - y_i \cdot f(\mathbf{x}_i; \theta)\right]}$$

This is the Poisson deviance loss. It is the correct loss for count data (e.g., number of clicks, number of defects, event rates).

---

### 5.4 The General Recipe

| Step | Action                                                                                               |
| ---- | ---------------------------------------------------------------------------------------------------- |
| 1    | **Choose a noise model** $p(y \mid \mu, \phi)$ matching your data (continuous? binary? count?)       |
| 2    | **Parameterise the mean:** $\mu_i = g(f(\mathbf{x}_i; \theta))$ for an appropriate link function $g$ |
| 3    | **Write the NLL:** $-\ell(\theta) = -\sum_i \log p(y_i \mid \mu_i, \phi)$                            |
| 4    | **Minimise NLL** using gradient descent or other optimiser                                           |

| Data type                | Distribution                          | Link function $g$           | Loss function    |
| ------------------------ | ------------------------------------- | --------------------------- | ---------------- |
| Continuous, symmetric    | Gaussian $\mathcal{N}(\mu, \sigma^2)$ | Identity ($\mu = f$)        | MSE              |
| Continuous, heavy-tailed | Laplace, Student-$t$                  | Identity                    | MAE, Huber       |
| Binary                   | Bernoulli                             | Sigmoid ($\mu = \sigma(f)$) | BCE              |
| Count                    | Poisson                               | Exp ($\mu = e^f$)           | Poisson deviance |
| Positive continuous      | Gamma                                 | Exp or log                  | Gamma deviance   |

> **This table unifies all loss functions encountered in the course.** Whenever you see a loss function, ask: "What noise model does this correspond to?" This perspective is more powerful than treating losses as ad-hoc choices.

---

## 6. MLE for Linear Regression (Full Derivation)

### 6.1 The Probabilistic Model

$$y_i = \mathbf{x}_i^\top\mathbf{w} + \epsilon_i, \quad \epsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$

Equivalently:

$$y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{x}_i^\top\mathbf{w}, \sigma^2)$$

Parameters to estimate: $\theta = (\mathbf{w}, \sigma^2)$.

---

### 6.2 Deriving the Log-Likelihood

$$p(y_i \mid \mathbf{x}_i, \mathbf{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y_i - \mathbf{x}_i^\top\mathbf{w})^2}{2\sigma^2}\right)$$

By i.i.d. assumption:

$$\ell(\mathbf{w}, \sigma^2) = \sum_{i=1}^{n}\log p(y_i \mid \mathbf{x}_i, \mathbf{w}, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top\mathbf{w})^2$$

In matrix form, with $\mathbf{r} = \mathbf{y} - X\mathbf{w}$ (residual vector):

$$\boxed{\ell(\mathbf{w}, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\mathbf{r}^\top\mathbf{r}}$$

---

### 6.3 MLE for $\mathbf{w}$: Recovering OLS

The only term that depends on $\mathbf{w}$ is $-\frac{1}{2\sigma^2}\mathbf{r}^\top\mathbf{r}$. Since $\sigma^2 > 0$:

$$\hat{\mathbf{w}}_{\text{MLE}} = \arg\max_\mathbf{w}\ell = \arg\min_\mathbf{w}\mathbf{r}^\top\mathbf{r} = \arg\min_\mathbf{w}\|\mathbf{y} - X\mathbf{w}\|^2$$

This is exactly the OLS objective. Taking the gradient and setting to zero (same as [Week 03](../../02_fundamentals/week03_linear_models/theory.md)):

$$\nabla_\mathbf{w}(\mathbf{r}^\top\mathbf{r}) = -2X^\top(\mathbf{y} - X\mathbf{w}) = \mathbf{0}$$

$$\boxed{\hat{\mathbf{w}}_{\text{MLE}} = (X^\top X)^{-1}X^\top\mathbf{y} = \hat{\mathbf{w}}_{\text{OLS}}}$$

> **The punchline.** OLS is not an arbitrary choice — it is the maximum likelihood estimator under the assumption of Gaussian noise. Every time you use MSE, you are implicitly assuming that the errors are Gaussian.

---

### 6.4 MLE for $\sigma^2$

Plugging $\hat{\mathbf{w}}$ into the log-likelihood and differentiating with respect to $\sigma^2$:

$$\frac{\partial\ell}{\partial\sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top\hat{\mathbf{w}})^2 = 0$$

$$\boxed{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}\hat{r}_i^2 = \frac{1}{n}\hat{\mathbf{r}}^\top\hat{\mathbf{r}} = \frac{\text{RSS}}{n}}$$

where $\hat{r}_i = y_i - \mathbf{x}_i^\top\hat{\mathbf{w}}$ is the residual and $\text{RSS} = \|\hat{\mathbf{r}}\|^2$ is the residual sum of squares.

**Interpretation.** $\hat{\sigma}^2_{\text{MLE}}$ is the average squared residual — the model's estimate of the noise variance. This is the number shown in the $\pm 2\sigma$ prediction band: the MLE says "the data scatter by approximately $\hat{\sigma}$ around the regression line."

> **Notebook reference.** Cell 5 jointly optimises $(\mathbf{w}, \sigma)$ using `scipy.optimize.minimize`, recovering the true parameters $\mathbf{w}_{\text{true}} = (2.0, 3.5)$, $\sigma_{\text{true}} = 1.0$. It also plots the $\pm 2\hat{\sigma}$ band.

---

### 6.5 Joint Optimisation in Practice

The notebook defines:

```python
def neg_log_likelihood_gaussian(params, X, y):
    w = params[:-1]; sigma = params[-1]
    if sigma <= 0: return 1e10
    mu = X @ w; residuals = y - mu
    return 0.5*n*np.log(2*np.pi*sigma**2) + np.sum(residuals**2)/(2*sigma**2)
```

This is the full NLL (including the normalisation constant), optimised over both $\mathbf{w}$ and $\sigma$ simultaneously. The `sigma <= 0` guard prevents invalid parameter values.

For Gaussian linear regression, closed-form solutions exist (Section 6.3–6.4), so numerical optimisation is unnecessary. But the same NLL template (`neg_log_likelihood_<distribution>`) generalises to any noise model where no closed form exists — the robust models in Section 7 use the same pattern.

---

## 7. Robust Regression: Alternative Noise Models

### 7.1 Why Gaussian MLE Fails with Outliers

The Gaussian log-density is proportional to $(y - \mu)^2$. An outlier with $|y - \mu| = 10\sigma$ contributes $100\sigma^2$ to the loss — **quadratically** in the deviation. This disproportionately large contribution pulls the estimated line toward the outlier.

Mathematically, the influence function of the Gaussian MLE is **unbounded**: the influence of a single observation on $\hat{\mathbf{w}}$ grows without limit as the observation moves further from the bulk of the data.

**The fix:** use a noise model with **heavier tails**, so that outliers have a bounded influence on the NLL.

---

### 7.2 Laplace Noise → MAE (L1 Loss)

**Model:** $\epsilon_i \sim \text{Laplace}(0, b)$.

**NLL:**

$$-\ell(\mathbf{w}, b) = n\log(2b) + \frac{1}{b}\sum_{i=1}^{n}|y_i - \mathbf{x}_i^\top\mathbf{w}|$$

For fixed $b$, minimising the NLL reduces to minimising the **Mean Absolute Error (MAE)**:

$$\hat{\mathbf{w}}_{\text{Laplace}} = \arg\min_\mathbf{w}\sum_{i=1}^{n}|y_i - \mathbf{x}_i^\top\mathbf{w}|$$

**Why this is robust:** the absolute value $|r_i|$ grows **linearly** with the residual (vs. quadratically for MSE). An outlier at $10\sigma$ contributes $10b$ instead of $100\sigma^2$ — a much smaller relative influence.

> **Connection to [Week 06](../../02_fundamentals/week06_regularization/theory.md).** The Laplace _prior_ on weights gives Lasso (L1 regularisation). The Laplace _noise model_ gives MAE loss. Both involve L1 norms, but applied to different quantities (weights vs. residuals).

---

### 7.3 Student-_t_ Noise → Heavy-Tailed Robustness

**Model:** $\epsilon_i \sim t_\nu(0, \sigma)$.

**NLL:**

$$-\ell(\mathbf{w}, \sigma) = -\sum_{i=1}^{n}\log t_\nu\!\left(\frac{y_i - \mathbf{x}_i^\top\mathbf{w}}{\sigma}\right) + n\log\sigma$$

At large residuals $|r_i| \to \infty$, the $t$-density decays as $|r_i|^{-(\nu+1)}$ (power law), so $-\log t_\nu(r_i) \propto (\nu+1)\log|r_i|$ — **logarithmic** growth. Compare:

| Noise model | NLL contribution of large residual $ | r                             | \to \infty$     | Influence on $\hat{\mathbf{w}}$                   |
| ----------- | ------------------------------------ | ----------------------------- |
| Gaussian    | $\propto r^2$ (quadratic)            | Unbounded (outliers dominate) |
| Laplace     | $\propto                             | r                             | $ (linear)      | Bounded constant                                  |
| Student-$t$ | $\propto \log                        | r                             | $ (logarithmic) | **Redescending** (decreases for extreme outliers) |

The Student-$t$ model is the **most robust**: extreme outliers are essentially ignored.

**Choosing $\nu$:**
- $\nu = 1$: Cauchy — extremely robust but very heavy tails (may be too robust).
- $\nu = 3$–$5$: good practical default for moderate outlier contamination.
- $\nu > 30$: nearly Gaussian.

> **Notebook reference.** Cell 9 fits Gaussian, Laplace, and Student-$t$ ($\nu = 3$) MLE on data with 5 injected outliers. The Gaussian line is visibly pulled toward the outliers; the Laplace and Student-$t$ lines stay close to the true model.

---

### 7.4 Huber Loss: A Practical Compromise

The **Huber loss** behaves quadratically for small residuals and linearly for large residuals:

$$L_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta|r| - \frac{1}{2}\delta^2 & |r| > \delta \end{cases}$$

| Residual $r$ | Behaviour | Why          |
| ------------ | --------- | ------------ |
| $            | r         | \leq \delta$ | Quadratic (like MSE) | Efficient for well-behaved data |
| $            | r         | > \delta$    | Linear (like MAE)    | Robust to outliers              |

The threshold $\delta$ controls the transition. Huber loss is:
- **Differentiable everywhere** (unlike MAE, which has a kink at 0).
- Used in `sklearn.linear_model.HuberRegressor`.
- The default choice when you suspect outliers but want to retain efficiency on clean data.

---

### 7.5 Comparison of Noise Models

| Property                   | Gaussian                  | Laplace              | Student-$t$ ($\nu = 3$)     | Huber           |
| -------------------------- | ------------------------- | -------------------- | --------------------------- | --------------- |
| **Loss**                   | $r^2$                     | $                    | r                           | $               | $\sim\log(1 + r^2/3)$ | Quadratic/linear |
| **Tail weight**            | Light (exponential decay) | Medium (exponential) | Heavy (polynomial)          | Medium          |
| **Robustness**             | None                      | Moderate             | Strong                      | Moderate        |
| **Sparsity in residuals?** | No                        | No                   | No                          | No              |
| **Differentiable?**        | Yes                       | No (kink at 0)       | Yes                         | Yes             |
| **MLE closed form?**       | Yes (OLS)                 | No (LP)              | No (iterative)              | No (iterative)  |
| **Best for**               | Clean data                | Moderate outliers    | Heavy outlier contamination | General default |

> **Suggested experiment.** Vary the number and magnitude of outliers in the notebook data. Observe the "breakdown point" — the fraction of outliers at which each estimator fails catastrophically. Gaussian breaks first, then Laplace, then Student-$t$.

---

## 8. Likelihood Surfaces and Optimisation

### 8.1 Convexity of the Gaussian NLL

For Gaussian linear regression, the NLL as a function of $\mathbf{w}$ (with $\sigma$ fixed) is:

$$-\ell(\mathbf{w}) \propto \|\mathbf{y} - X\mathbf{w}\|^2$$

The Hessian is $\nabla^2(-\ell) = \frac{1}{\sigma^2}X^\top X$, which is positive semi-definite (PSD). Therefore:

- The NLL is **convex** in $\mathbf{w}$ — every local minimum is global.
- The NLL is **strictly convex** when $X^\top X$ is positive definite ($X$ has full column rank).
- Gradient descent is guaranteed to converge to the global minimum ([Week 01](../../02_fundamentals/week01_optimization/theory.md)).

Joint optimisation over $(\mathbf{w}, \sigma^2)$ is also convex (the NLL is convex in $\sigma^2$ given $\mathbf{w}$, and vice versa; and the joint function is convex).

---

### 8.2 Non-Convex Likelihood Surfaces

For more complex models, the NLL can be **non-convex** with multiple local minima:

| Model                           | NLL convexity                                              |
| ------------------------------- | ---------------------------------------------------------- |
| Linear regression (Gaussian)    | Convex                                                     |
| Logistic regression (Bernoulli) | Convex                                                     |
| Gaussian mixtures (GMM)         | **Non-convex** (multiple modes from component relabelling) |
| Neural networks                 | **Non-convex** (many saddle points and local minima)       |

> **Notebook reference.** Cell 11 plots the NLL surface for Gaussian linear regression as both contour and 3D. The surface is a smooth, convex bowl with a single minimum at the MLE. For non-linear or mixture models, the surface would have multiple basins.

> **Forward pointer.** Non-convex optimisation landscapes are the central challenge of deep learning. [[Weeks 11](../../04_neural_networks/week11_nn_from_scratch/theory.md)](../../04_neural_networks/week11_nn_from_scratch/theory.md)–[14](../../05_deep_learning/week14_training_at_scale/theory.md) address this with techniques like learning rate schedules, momentum, batch normalisation, and careful initialisation.

---

### 8.3 Numerical Optimisation: scipy.optimize.minimize

When no closed form exists, MLE is solved numerically:

```python
from scipy.optimize import minimize
result = minimize(neg_log_likelihood, x0, args=(X, y), method='L-BFGS-B')
w_mle = result.x
```

| Method          | Description                           | When to use               |
| --------------- | ------------------------------------- | ------------------------- |
| `'L-BFGS-B'`    | Quasi-Newton with bounded constraints | Default for smooth NLLs   |
| `'Nelder-Mead'` | Derivative-free (simplex)             | Non-differentiable NLLs   |
| `'BFGS'`        | Quasi-Newton (no bounds)              | Unconstrained smooth NLLs |

> **Practical detail.** The notebook guards against $\sigma \leq 0$ by returning a large value (1e10). A cleaner approach is to **reparameterise**: optimise over $\log\sigma$ instead of $\sigma$, which maps $\mathbb{R} \to (0, \infty)$ automatically. This is a common trick in probabilistic modelling.

---

## 9. From MLE to MAP: The Bridge to Regularisation

### 9.1 Maximum A Posteriori Estimation

Bayes' theorem says:

$$p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\,p(\theta)$$

The **MAP estimator** maximises the posterior, or equivalently minimises the negative log-posterior:

$$\hat{\theta}_{\text{MAP}} = \arg\min_\theta\left[-\log p(\mathcal{D} \mid \theta) - \log p(\theta)\right] = \arg\min_\theta\left[-\ell(\theta) - \log p(\theta)\right]$$

---

### 9.2 MAP = Regularised MLE

| Prior $p(\theta)$                   | $-\log p(\theta)$                                    | MAP objective                     | Equivalent to                           |
| ----------------------------------- | ---------------------------------------------------- | --------------------------------- | --------------------------------------- |
| $\mathcal{N}(\mathbf{0}, \tau^2 I)$ | $\frac{1}{2\tau^2}\|\mathbf{w}\|_2^2 + \text{const}$ | NLL + $\lambda\|\mathbf{w}\|_2^2$ | **Ridge** ($\lambda = \sigma^2/\tau^2$) |
| $\text{Laplace}(\mathbf{0}, b)$     | $\frac{1}{b}\|\mathbf{w}\|_1 + \text{const}$         | NLL + $\lambda\|\mathbf{w}\|_1$   | **Lasso** ($\lambda = \sigma^2/b$)      |
| Uniform (improper)                  | $0$                                                  | NLL                               | **MLE** (no regularisation)             |

$$\boxed{\text{MLE} + \text{Prior} = \text{MAP} = \text{Regularised MLE}}$$

> **This closes the circle with [Week 06](../../02_fundamentals/week06_regularization/theory.md).** Ridge regression is not ad-hoc — it is the MAP estimate under Gaussian noise and a Gaussian prior on the weights. The regularisation strength $\lambda$ encodes the prior precision relative to the noise variance.

> **Notebook reference.** Exercise 2 (Cell 14) asks you to add an L2 prior to the Gaussian NLL and verify that MAP matches Ridge.
>
> **Suggested experiment.** Implement MAP with a Laplace prior and verify it matches Lasso. Then implement MAP with an Elastic Net prior (Gaussian + Laplace mixture) and compare coefficient paths.

---

## 10. Model Selection: AIC and BIC

When comparing models with different numbers of parameters (e.g., different noise assumptions, polynomial degrees, or number of mixture components), the log-likelihood alone is insufficient — more complex models always have higher likelihood. Information criteria penalise complexity.

**Akaike Information Criterion (AIC):**

$$\text{AIC} = -2\ell(\hat{\theta}) + 2p$$

**Bayesian Information Criterion (BIC):**

$$\text{BIC} = -2\ell(\hat{\theta}) + p\log n$$

| Symbol               | Meaning                   |
| -------------------- | ------------------------- |
| $\ell(\hat{\theta})$ | Maximised log-likelihood  |
| $p$                  | Number of free parameters |
| $n$                  | Number of data points     |

**Interpretation:**
- The first term ($-2\ell$) measures **goodness of fit** (lower = better fit).
- The second term ($2p$ or $p\log n$) penalises **model complexity**.
- Choose the model that **minimises** AIC or BIC.

**AIC vs. BIC:**
- BIC penalises complexity more heavily for $n > 8$ (since $\log n > 2$).
- AIC tends to select more complex models; BIC tends to select simpler models.
- BIC is **consistent**: as $n \to \infty$, it selects the true model (if it's in the candidate set).
- AIC is **efficient**: it selects the model with the best predictive performance (even if the true model is not in the set).

> **Connection to [Week 05](../../02_fundamentals/week05_clustering/theory.md).** BIC was used for GMM model selection (choosing the number of Gaussian components). The same criterion applies here for choosing between noise models or polynomial degrees.

---

## 11. Connections to the Rest of the Course

| Week                            | Connection                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------- |
| **[Week 00b](../../01_intro/week00b_math_and_data/theory.md) (Math)**             | Gaussian distribution and the MLE preview are formalised here                                  |
| **[Week 03](../../02_fundamentals/week03_linear_models/theory.md) (Linear Models)**     | MSE = NLL under Gaussian noise; OLS = MLE. BCE = NLL under Bernoulli.                          |
| **[Week 06](../../02_fundamentals/week06_regularization/theory.md) (Regularisation)**    | Ridge = MAP with Gaussian prior; Lasso = MAP with Laplace prior                                |
| **[Week 08](../week08_uncertainty/theory.md) (Uncertainty)**       | Full Bayesian inference goes beyond MAP to the full posterior $p(\mathbf{w} \mid \mathcal{D})$ |
| **[Week 09](../week09_time_series/theory.md) (Time Series)**       | Autoregressive likelihoods; MLE for AR/ARIMA models                                            |
| **[Week 10](../week10_surrogate_models/theory.md) (Surrogate Models)**  | Gaussian processes maximise marginal likelihood                                                |
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md) (NNs)**               | Neural network training = MLE (or MAP with weight decay). Cross-entropy loss = Bernoulli NLL.  |
| **[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md) (Training at Scale)** | Mixed-precision training affects NLL numerics; log-sum-exp tricks                              |
| **[Week 18](../../06_sequence_models/week18_transformers/theory.md) (Transformers)**      | Autoregressive language models maximise $\sum_t \log p(x_t \mid x_{<t})$ — sequential NLL      |

> **The unifying principle.** Every model in this course is defined by a probabilistic assumption (noise model) and an optimisation procedure (minimise NLL or regularised NLL). The choice of noise model determines the loss function; the choice of prior determines the regulariser; the choice of optimiser determines how fast you converge.

---

## 12. Notebook Reference Guide

| Cell                    | Section                      | What it demonstrates                                                                            | Theory reference      |
| ----------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------- | --------------------- |
| 5 (Gaussian MLE)        | MLE for linear regression    | Joint optimisation of $(\mathbf{w}, \sigma)$; recovers true parameters; $\pm 2\sigma$ band      | Section 6             |
| 7 (NLL vs. MSE)         | NLL $\propto$ MSE (Gaussian) | Normalised NLL and MSE curves overlap perfectly; correlation ≈ 1.0                              | Section 5.1           |
| 9 (Robust regression)   | Laplace and Student-$t$ MLE  | Three fitted lines on outlier-contaminated data; Gaussian is pulled; Laplace/Student-$t$ resist | Section 7             |
| 11 (Likelihood surface) | NLL contour + 3D surface     | Convex bowl for Gaussian linear; single global minimum at MLE                                   | Section 8.1           |
| Exercise 1 (Cell 13)    | Poisson regression           | Count-data MLE; exponential link function                                                       | Section 5.3           |
| Exercise 2 (Cell 14)    | MAP = Ridge                  | Adding Gaussian prior to NLL recovers Ridge                                                     | Section 9             |
| Exercise 4 (Cell 15)    | Heteroscedastic noise        | $\sigma(x) = \sigma_0 e^{0.3x}$; jointly model mean and variance                                | Section 7 (extension) |

**Suggested modifications:**

| Modification                                                                 | What it reveals                                                                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Plot NLL vs. both $w_0$ and $w_1$ (contour) while also varying $\sigma$      | $\sigma$ controls the "width" of the bowl: smaller $\sigma$ = sharper curvature = more confident |
| Add 20% outliers (not just 5) to the robust regression                       | At high contamination, even Laplace breaks down; Student-$t$ with $\nu = 1$ (Cauchy) persists    |
| Reparameterise $\sigma \to \log\sigma$ in `neg_log_likelihood_gaussian`      | Eliminates the `if sigma <= 0` guard; smoother optimisation                                      |
| Implement Huber loss MLE and compare to the three noise models               | Huber should match or exceed Laplace robustness while being differentiable                       |
| Repeat the NLL-MSE comparison for logistic regression                        | Show that BCE and Bernoulli NLL coincide, confirming the connection                              |
| Fit a degree-5 polynomial with Gaussian MLE and compute AIC/BIC vs. degree-1 | BIC should prefer degree 1 if the true relationship is linear                                    |

---

## 13. Symbol Reference

| Symbol                          | Name                             | Meaning                                                                     |
| ------------------------------- | -------------------------------- | --------------------------------------------------------------------------- |
| $y_i$                           | Observation                      | Observed target value for sample $i$                                        |
| $\mathbf{x}_i \in \mathbb{R}^d$ | Feature vector                   | Input features for sample $i$                                               |
| $\theta$                        | Parameters                       | Generic parameter vector (may be $\mathbf{w}$, $\sigma$, etc.)              |
| $\mathcal{D}$                   | Dataset                          | $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$                                           |
| $p(y \mid \theta)$              | Probability (density)            | Probability of $y$ given parameters $\theta$                                |
| $\mathcal{L}(\theta)$           | Likelihood                       | $\prod_i p(y_i \mid \theta)$ — function of $\theta$                         |
| $\ell(\theta)$                  | Log-likelihood                   | $\sum_i \log p(y_i \mid \theta)$                                            |
| $-\ell(\theta)$                 | Negative log-likelihood (NLL)    | The loss function to minimise                                               |
| $\hat{\theta}_{\text{MLE}}$     | MLE estimator                    | $\arg\max_\theta \ell(\theta)$                                              |
| $\hat{\theta}_{\text{MAP}}$     | MAP estimator                    | $\arg\max_\theta [\ell(\theta) + \log p(\theta)]$                           |
| $\mu$                           | Mean parameter                   | Location of the distribution                                                |
| $\sigma^2$                      | Variance                         | Noise variance (Gaussian)                                                   |
| $b$                             | Scale (Laplace)                  | Laplace distribution scale parameter                                        |
| $\nu$                           | Degrees of freedom (Student-$t$) | Controls tail heaviness; lower = heavier                                    |
| $\delta$                        | Huber threshold                  | Transition between quadratic and linear                                     |
| $I(\theta)$                     | Fisher information               | $-\mathbb{E}[\partial^2\ell/\partial\theta^2]$; curvature of log-likelihood |
| $\text{AIC}$                    | Akaike Information Criterion     | $-2\ell + 2p$                                                               |
| $\text{BIC}$                    | Bayesian Information Criterion   | $-2\ell + p\log n$                                                          |
| $p$ (in AIC/BIC)                | Number of parameters             | Free parameters in the model                                                |
| $\hat{p}_i$                     | Predicted probability            | $\sigma(f(\mathbf{x}_i; \theta))$ for logistic regression                   |
| $\lambda_i$                     | Poisson rate                     | $\exp(f(\mathbf{x}_i; \theta))$                                             |
| $\tau^2$                        | Prior variance                   | Width of the Gaussian prior on $\mathbf{w}$                                 |
| $\mathbf{r}$                    | Residual vector                  | $\mathbf{y} - X\mathbf{w}$                                                  |
| RSS                             | Residual sum of squares          | $\mathbf{r}^\top\mathbf{r} = \|\mathbf{y} - X\mathbf{w}\|^2$                |

---

## 14. References

1. Fisher, R. A. (1922). "On the Mathematical Foundations of Theoretical Statistics." *Philosophical Transactions of the Royal Society A*, 222, 309–368. — The original formulation of maximum likelihood.
2. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapters 3–5. MIT Press. — MLE, MAP, Bayesian inference; comprehensive and rigorous.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapters 1–3. Springer. — Probability distributions, MLE, Bayesian linear regression.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Chapter 8. Springer. — Likelihood-based inference and model assessment.
5. Casella, G. & Berger, R. L. (2002). *Statistical Inference*, 2nd ed. Duxbury. — Formal treatment of MLE, Fisher information, Cramér–Rao bound.
6. Huber, P. J. (1964). "Robust Estimation of a Location Parameter." *Annals of Mathematical Statistics*, 35(1), 73–101. — Huber loss; robust M-estimation.
7. Lange, K. L., Little, R. J. A., & Taylor, J. M. G. (1989). "Robust Statistical Modeling Using the $t$ Distribution." *JASA*, 84(408), 881–896. — Student-$t$ likelihood for robust regression.
8. Akaike, H. (1974). "A New Look at the Statistical Model Identification." *IEEE Transactions on Automatic Control*, 19(6), 716–723. — AIC.
9. Schwarz, G. (1978). "Estimating the Dimension of a Model." *Annals of Statistics*, 6(2), 461–464. — BIC.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 5: Machine Learning Basics. MIT Press. — MLE and its connection to deep learning loss functions.
11. SciPy documentation: `scipy.stats` and `scipy.optimize`. [https://docs.scipy.org/doc/scipy/](https://docs.scipy.org/doc/scipy/).
12. Wasserman, L. (2004). *All of Statistics*, Chapters 9–10. Springer. — Clear, concise treatment of MLE and its properties.
