# Uncertainty & Statistics

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Why Uncertainty Matters](#2-why-uncertainty-matters)
3. [Two Kinds of Uncertainty](#3-two-kinds-of-uncertainty)
    - 3.1 [Aleatoric Uncertainty (Data Noise)](#31-aleatoric-uncertainty-data-noise)
    - 3.2 [Epistemic Uncertainty (Model Ignorance)](#32-epistemic-uncertainty-model-ignorance)
    - 3.3 [Decomposing Total Uncertainty](#33-decomposing-total-uncertainty)
4. [Frequentist Uncertainty: Confidence Intervals](#4-frequentist-uncertainty-confidence-intervals)
    - 4.1 [Sampling Distributions](#41-sampling-distributions)
    - 4.2 [Confidence Intervals for the Mean](#42-confidence-intervals-for-the-mean)
    - 4.3 [Confidence Intervals for Regression Parameters](#43-confidence-intervals-for-regression-parameters)
5. [The Bootstrap](#5-the-bootstrap)
    - 5.1 [The Nonparametric Bootstrap](#51-the-nonparametric-bootstrap)
    - 5.2 [Bootstrap Confidence Intervals](#52-bootstrap-confidence-intervals)
    - 5.3 [When the Bootstrap Fails](#53-when-the-bootstrap-fails)
6. [Monte Carlo Methods](#6-monte-carlo-methods)
    - 6.1 [Monte Carlo Estimation](#61-monte-carlo-estimation)
    - 6.2 [Convergence Rate](#62-convergence-rate)
    - 6.3 [Monte Carlo for Uncertainty Propagation](#63-monte-carlo-for-uncertainty-propagation)
7. [Bayesian Inference](#7-bayesian-inference)
    - 7.1 [The Bayesian Framework](#71-the-bayesian-framework)
    - 7.2 [Prior, Likelihood, Posterior](#72-prior-likelihood-posterior)
    - 7.3 [Credible Intervals vs. Confidence Intervals](#73-credible-intervals-vs-confidence-intervals)
    - 7.4 [Conjugate Priors](#74-conjugate-priors)
8. [Bayesian Linear Regression](#8-bayesian-linear-regression)
    - 8.1 [The Model](#81-the-model)
    - 8.2 [The Posterior Over Weights](#82-the-posterior-over-weights)
    - 8.3 [The Posterior Predictive Distribution](#83-the-posterior-predictive-distribution)
    - 8.4 [Sequential Bayesian Updating](#84-sequential-bayesian-updating)
    - 8.5 [Connection to Ridge Regression](#85-connection-to-ridge-regression)
9. [Calibration](#9-calibration)
    - 9.1 [What Is Calibration?](#91-what-is-calibration)
    - 9.2 [Reliability Diagrams](#92-reliability-diagrams)
    - 9.3 [Standardised Residuals and Q-Q Plots](#93-standardised-residuals-and-q-q-plots)
    - 9.4 [Calibration for Classification](#94-calibration-for-classification)
10. [Prediction Intervals](#10-prediction-intervals)
    - 10.1 [Frequentist Prediction Intervals](#101-frequentist-prediction-intervals)
    - 10.2 [Bayesian Predictive Intervals](#102-bayesian-predictive-intervals)
    - 10.3 [Conformal Prediction (Distribution-Free)](#103-conformal-prediction-distribution-free)
11. [Ensemble Uncertainty](#11-ensemble-uncertainty)
12. [Connections to the Rest of the Course](#12-connections-to-the-rest-of-the-course)
13. [Notebook Reference Guide](#13-notebook-reference-guide)
14. [Symbol Reference](#14-symbol-reference)
15. [References](#15-references)

---

## 1. Scope and Purpose

[Week 07](../week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle) answered "what are the best parameter estimates?" using maximum likelihood. This week asks the follow-up question: **how confident should we be in those estimates, and in the predictions they produce?**

A model that says "tomorrow's temperature will be 22 °C" is less useful than one that says "22 °C ± 3 °C with 95% confidence." Quantifying uncertainty is essential for:
- **Decision-making under risk** — engineering, medicine, finance.
- **Identifying gaps in data** — epistemic uncertainty is high where you lack training data.
- **Detecting model failure** — an overconfident wrong prediction is more dangerous than an uncertain one.
- **Active learning** — querying the most uncertain examples to label next.

This week covers three complementary approaches to uncertainty:
1. **Frequentist** — confidence intervals, sampling distributions.
2. **Bootstrap** — resampling-based, distribution-free.
3. **Bayesian** — posterior distributions, credible intervals, predictive distributions.

And one cross-cutting diagnostic: **calibration** — checking that the stated uncertainties are honest.

**Prerequisites.** [Week 07](../week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle) (likelihood, MLE, Bayes' theorem), [Week 06](../../02_fundamentals/week06_regularization/theory.md#35-bayesian-interpretation) (Ridge as MAP = Bayesian connection), [Week 03](../../02_fundamentals/week03_linear_models/theory.md#3-linear-regression) (linear regression).

---

## 2. Why Uncertainty Matters

| Scenario           | Without uncertainty         | With uncertainty                                         |
| ------------------ | --------------------------- | -------------------------------------------------------- |
| Medical diagnosis  | "The patient has disease X" | "80% probability of X; recommend confirmatory test"      |
| Autonomous driving | "No obstacle ahead"         | "70% confident — slow down and use additional sensors"   |
| Financial model    | "Expected return is 8%"     | "8% ± 12% — the investment may lose money"               |
| ML deployment      | "Prediction: class A"       | "Prediction: class A (confidence 0.52) — defer to human" |

> **The failure mode of ignoring uncertainty.** A model that is 99% accurate but gives no uncertainty estimate will silently fail on the 1% of cases. A model that is 90% accurate but correctly flags its uncertain predictions can be far more valuable in practice because the uncertain cases can be routed to a human expert.

---

## 3. Two Kinds of Uncertainty

### 3.1 Aleatoric Uncertainty (Data Noise)

**Aleatoric** (from Latin _alea_ — dice) uncertainty arises from inherent randomness in the data-generating process. It is **irreducible** — no amount of additional data or a better model will eliminate it.

**Examples:**
- Measurement noise in sensors.
- Stochastic outcomes (coin flips, quantum phenomena).
- Unobserved confounders that act like random noise.

**Mathematically:** for the model $y = f(\mathbf{x}) + \epsilon$, $\text{Var}[\epsilon] = \sigma^2$ is the aleatoric uncertainty. In **homoscedastic** models, $\sigma^2$ is constant; in **heteroscedastic** models, $\sigma^2(\mathbf{x})$ varies with the input.

---

### 3.2 Epistemic Uncertainty (Model Ignorance)

**Epistemic** (from Greek _episteme_ — knowledge) uncertainty arises from **lack of knowledge**: limited data, model misspecification, parameter uncertainty. It is **reducible** — more data or a better model can decrease it.

**Examples:**
- Uncertainty in regression coefficients due to finite sample size.
- Predictions in regions with no training data (extrapolation).
- Choosing between model architectures.

**Mathematically:** in Bayesian linear regression, epistemic uncertainty is captured by the posterior variance of the weights $\text{Var}[\mathbf{w} \mid \mathcal{D}]$, which shrinks as $n$ increases.

---

### 3.3 Decomposing Total Uncertainty

For a prediction at a new point $\mathbf{x}_*$, the total predictive variance decomposes as:

$$\boxed{\underbrace{\text{Var}[y_* \mid \mathbf{x}_*]}_{\text{total}} = \underbrace{\text{Var}[\mathbb{E}[y_* \mid \mathbf{w}, \mathbf{x}_*]]}_{\text{epistemic (model)}} + \underbrace{\mathbb{E}[\text{Var}[y_* \mid \mathbf{w}, \mathbf{x}_*]]}_{\text{aleatoric (noise)}}}$$

This is the **law of total variance** (Eve's law). In Bayesian linear regression:

$$\text{Var}[y_*] = \underbrace{\mathbf{x}_*^\top\Sigma_{\text{post}}\mathbf{x}_*}_{\text{epistemic}} + \underbrace{\sigma^2}_{\text{aleatoric}}$$

| Property           | Aleatoric                          | Epistemic                           |
| ------------------ | ---------------------------------- | ----------------------------------- |
| Source             | Data noise                         | Limited data / model ignorance      |
| Reducible?         | No                                 | Yes (more data helps)               |
| Behaviour with $n$ | Constant                           | Shrinks as $n \to \infty$           |
| Where largest?     | High-noise regions                 | Data-sparse regions (extrapolation) |
| How to estimate?   | Residual variance $\hat{\sigma}^2$ | Posterior variance, ensemble spread |

> **Practical importance.** Knowing _which_ kind of uncertainty dominates tells you what to do. High aleatoric uncertainty → collect cleaner data or accept the noise floor. High epistemic uncertainty → collect more data in that region or use a more flexible model.

---

## 4. Frequentist Uncertainty: Confidence Intervals

### 4.1 Sampling Distributions

In the frequentist framework, parameters $\theta$ are fixed (unknown) constants. Uncertainty is quantified through the **sampling distribution** — the distribution of an estimator $\hat{\theta}$ across hypothetical repeated samples from the population.

If we could draw $B$ independent datasets $\mathcal{D}_1, \ldots, \mathcal{D}_B$ and compute $\hat{\theta}_1, \ldots, \hat{\theta}_B$, then:
- The mean of $\hat{\theta}_b$ estimates $\theta$ (if the estimator is unbiased).
- The spread of $\hat{\theta}_b$ quantifies **sampling variability**.

In practice, we have only one dataset. We either derive the sampling distribution analytically or approximate it via the bootstrap (Section 5).

---

### 4.2 Confidence Intervals for the Mean

Given i.i.d. samples $y_1, \ldots, y_n$ from a distribution with mean $\mu$ and variance $\sigma^2$, the Central Limit Theorem (CLT) gives:

$$\bar{y} \approx \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{for large } n$$

The **standard error** of the mean is $\text{SE}(\bar{y}) = \sigma / \sqrt{n}$ (estimated by $s / \sqrt{n}$ where $s$ is the sample standard deviation).

A $(1 - \alpha)$ confidence interval for $\mu$:

$$\boxed{\bar{y} \pm z_{1-\alpha/2}\cdot\frac{s}{\sqrt{n}}}$$

For unknown $\sigma$ and small $n$, replace $z$ with $t_{n-1}$:

$$\bar{y} \pm t_{n-1, 1-\alpha/2}\cdot\frac{s}{\sqrt{n}}$$

| $1-\alpha$ | $z_{1-\alpha/2}$ | Interpretation              |
| ---------- | ---------------- | --------------------------- |
| 90%        | 1.645            | 10% of intervals miss $\mu$ |
| 95%        | 1.960            | 5% of intervals miss $\mu$  |
| 99%        | 2.576            | 1% of intervals miss $\mu$  |

> **What a confidence interval means (frequentist interpretation).** If we repeated the experiment many times and computed a 95% CI each time, approximately 95% of those intervals would contain the true $\mu$. A _single_ interval either contains $\mu$ or it doesn't — we cannot assign a probability to that.

---

### 4.3 Confidence Intervals for Regression Parameters

For OLS estimator $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top\mathbf{y}$ under Gaussian noise:

$$\hat{\mathbf{w}} \sim \mathcal{N}\!\left(\mathbf{w}^*, \sigma^2(X^\top X)^{-1}\right)$$

The standard error for coefficient $j$ is:

$$\text{SE}(\hat{w}_j) = \hat{\sigma}\sqrt{[(X^\top X)^{-1}]_{jj}}$$

A 95% CI for $w_j$:

$$\hat{w}_j \pm t_{n-d-1, 0.975}\cdot\text{SE}(\hat{w}_j)$$

> **The $t$-test for significance.** If the CI contains zero, we cannot reject the null hypothesis $w_j = 0$ at the 5% level — the feature may not be useful. This is the basis of feature significance tests in `statsmodels`.

---

## 5. The Bootstrap

### 5.1 The Nonparametric Bootstrap

The **bootstrap** (Efron, 1979) estimates the sampling distribution of _any_ statistic by resampling from the observed data.

**Algorithm:**

1. **Input:** dataset $\mathcal{D} = \{y_1, \ldots, y_n\}$, statistic $T(\cdot)$, number of bootstrap samples $B$.
2. **For** $b = 1, \ldots, B$:
   - Draw a bootstrap sample $\mathcal{D}_b^*$ by sampling $n$ points from $\mathcal{D}$ **with replacement**.
   - Compute $T_b^* = T(\mathcal{D}_b^*)$.
3. **Output:** the bootstrap distribution $\{T_1^*, \ldots, T_B^*\}$.

```python
def bootstrap_stat(data, stat_fn, n_boot=1000, alpha=0.05):
    n = len(data)
    boot_stats = [stat_fn(np.random.choice(data, size=n, replace=True))
                  for _ in range(n_boot)]
    boot_stats = np.array(boot_stats)
    ci = np.percentile(boot_stats, [100*alpha/2, 100*(1-alpha/2)])
    return stat_fn(data), ci, boot_stats
```

**Key properties:**
- **No distributional assumptions.** Works for any statistic (mean, median, regression coefficients, AUC, etc.).
- **Approximates the sampling distribution** by treating the empirical distribution as a proxy for the true distribution.
- **Typical $B$:** $B = 1000$–$10000$.

> **Notebook reference.** Cell 7 implements the bootstrap for mean and median of exponential data ($n = 50$, $B = 2000$), producing histograms of the bootstrap distributions with 95% percentile intervals.

---

### 5.2 Bootstrap Confidence Intervals

Several methods exist:

**Percentile method** (simplest):

$$\text{CI}_{1-\alpha} = \left[T^*_{\alpha/2}, \, T^*_{1-\alpha/2}\right]$$

where $T^*_q$ is the $q$-quantile of the bootstrap distribution.

**Basic (pivotal) method:**

$$\text{CI}_{1-\alpha} = \left[2\hat{T} - T^*_{1-\alpha/2}, \, 2\hat{T} - T^*_{\alpha/2}\right]$$

This reflects the bootstrap distribution around the observed statistic, correcting for bias.

**BCa (bias-corrected and accelerated) method:** adjusts for both bias and skewness of the bootstrap distribution. It is the recommended default in most settings:

```python
from scipy.stats import bootstrap
result = bootstrap((data,), np.mean, confidence_level=0.95, method='BCa')
```

---

### 5.3 When the Bootstrap Fails

| Scenario                                           | Problem                                                         |
| -------------------------------------------------- | --------------------------------------------------------------- |
| **Extreme quantiles** (e.g., max, 99th percentile) | Bootstrap has difficulty sampling rare events                   |
| **Very small $n$** ($n < 10$)                      | Resampling the same few points ≠ resampling from the population |
| **Heavy dependence** (time series)                 | i.i.d. bootstrap breaks temporal structure; use block bootstrap |
| **Infinite variance** distributions                | CLT doesn't apply; bootstrap may not converge                   |

> **Suggested experiment.** Run the bootstrap on $n = 5$ vs. $n = 500$ samples from a Gaussian. Compare bootstrap CIs to the analytic $t$-interval. With $n = 5$, bootstrap CIs are noticeably different (and less reliable).

---

## 6. Monte Carlo Methods

### 6.1 Monte Carlo Estimation

**Problem:** compute $\mathbb{E}[f(X)]$ where $X \sim p(x)$.

**Monte Carlo estimate:** draw $N$ i.i.d. samples $x_1, \ldots, x_N \sim p(x)$ and approximate:

$$\boxed{\mathbb{E}[f(X)] \approx \hat{\mu}_N = \frac{1}{N}\sum_{i=1}^{N}f(x_i)}$$

By the law of large numbers, $\hat{\mu}_N \xrightarrow{a.s.} \mathbb{E}[f(X)]$ as $N \to \infty$.

---

### 6.2 Convergence Rate

By the CLT:

$$\hat{\mu}_N \sim \mathcal{N}\!\left(\mathbb{E}[f(X)], \frac{\text{Var}[f(X)]}{N}\right)$$

The **Monte Carlo standard error** is:

$$\text{SE} = \frac{\hat{\sigma}_f}{\sqrt{N}}$$

where $\hat{\sigma}_f^2 = \frac{1}{N-1}\sum_{i=1}^{N}(f(x_i) - \hat{\mu}_N)^2$.

**Key property:** the error decreases as $1/\sqrt{N}$ regardless of the dimensionality of $X$. This makes Monte Carlo the method of choice for high-dimensional integrals (where grid-based methods suffer the curse of dimensionality).

| Samples $N$  | Relative error reduction      |
| ------------ | ----------------------------- |
| $\times 4$   | Error halves ($\sqrt{4} = 2$) |
| $\times 100$ | Error reduces by 10×          |
| $10{,}000$   | Typically 1% precision        |

> **Notebook reference.** Cell 5 estimates $\mathbb{E}[X^2 + \sin(X)]$ for $X \sim \mathcal{N}(0,1)$ using 10,000 samples, producing the empirical distribution and 95% percentile interval.

---

### 6.3 Monte Carlo for Uncertainty Propagation

**Problem:** you know $\mathbf{w} \sim p(\mathbf{w} \mid \mathcal{D})$ (the posterior) and want to compute the distribution of predictions $y_* = f(\mathbf{x}_*; \mathbf{w})$.

**Procedure:**
1. Draw $\mathbf{w}_1, \ldots, \mathbf{w}_N \sim p(\mathbf{w} \mid \mathcal{D})$.
2. Compute $y_*^{(i)} = f(\mathbf{x}_*; \mathbf{w}_i)$ for each sample.
3. Summarise: mean, variance, percentiles of $\{y_*^{(i)}\}$.

This is how Bayesian uncertainty estimates are typically computed in practice (even when the posterior is not Gaussian): MCMC (Markov Chain Monte Carlo) or variational inference generates posterior samples, and predictions are propagated through the model.

> **Notebook reference.** Cell 9 draws 50 weight vectors from the BLR posterior $\mathcal{N}(\boldsymbol{\mu}_{\text{post}}, \Sigma_{\text{post}})$ and plots the corresponding regression lines. The spread of lines visualises epistemic uncertainty — wide where data is sparse.

---

## 7. Bayesian Inference

### 7.1 The Bayesian Framework

The Bayesian approach treats parameters $\theta$ as **random variables** with a probability distribution, not fixed unknowns. All inference follows from Bayes' theorem:

$$\boxed{p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})}}$$

| Step         | Name                                               | Meaning                                      |
| ------------ | -------------------------------------------------- | -------------------------------------------- |
| Start        | **Prior** $p(\theta)$                              | Encode beliefs before seeing data            |
| Observe data | **Likelihood** $p(\mathcal{D} \mid \theta)$        | How probable is the data under each $\theta$ |
| Update       | **Posterior** $p(\theta \mid \mathcal{D})$         | Updated belief after seeing data             |
| Predict      | **Posterior predictive** $p(y_* \mid \mathcal{D})$ | Integrate out $\theta$ to predict new data   |

**Comparison with MLE ([Week 07](../week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle)):**

|                        | MLE                                                          | Bayesian                                       |
| ---------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| **Output**             | Single point estimate $\hat{\theta}$                         | Full distribution $p(\theta \mid \mathcal{D})$ |
| **Uncertainty**        | Must be added via separate analysis (Fisher info, bootstrap) | Built in (posterior width)                     |
| **Regularisation**     | Added externally (Ridge, Lasso)                              | Built in (prior acts as regulariser)           |
| **Small data**         | Prone to overfitting                                         | Prior stabilises estimates                     |
| **Computational cost** | Low (one optimisation)                                       | Higher (integration/sampling)                  |

---

### 7.2 Prior, Likelihood, Posterior

**Prior** $p(\theta)$: encodes what you know (or assume) before seeing data.

| Prior type                     | Description                          | Example                                             |
| ------------------------------ | ------------------------------------ | --------------------------------------------------- |
| **Informative**                | Concentrates around plausible values | $w \sim \mathcal{N}(0, 1)$ (weights small)          |
| **Weakly informative**         | Broad but not flat                   | $w \sim \mathcal{N}(0, 10)$                         |
| **Non-informative (improper)** | Flat over the entire real line       | $p(w) \propto 1$ (improper; doesn't integrate to 1) |

> **Sensitivity to the prior.** With enough data, the posterior is dominated by the likelihood and the prior becomes irrelevant (the data "washes out" the prior). With little data, the prior has a strong effect. This is both the strength (regularisation) and the criticism (subjectivity) of Bayesian methods.

**Likelihood** $p(\mathcal{D} \mid \theta)$: same as in [Week 07](../week07_likelihood/theory.md#3-likelihood-from-data-to-models). The product of $n$ i.i.d. density evaluations.

**Posterior** $p(\theta \mid \mathcal{D})$: the complete solution to the inference problem. Every summary (point estimate, interval, prediction) is derived from it.

**Evidence (marginal likelihood)** $p(\mathcal{D})$:

$$p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\,p(\theta)\,d\theta$$

This integral is typically **intractable** (it's the reason Bayesian inference is computationally expensive). It is a normalising constant; for computing the posterior, it can often be ignored ($p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\,p(\theta)$). But it is useful for **model comparison** (comparing the evidence of different models).

---

### 7.3 Credible Intervals vs. Confidence Intervals

|                              | Frequentist CI                                                                    | Bayesian credible interval                                            |
| ---------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Definition**               | Procedure that captures $\theta$ in $(1-\alpha)$ fraction of repeated experiments | Set $C$ such that $P(\theta \in C \mid \mathcal{D}) = 1-\alpha$       |
| **Statement about $\theta$** | "$\theta$ is a fixed constant; the interval is random"                            | "$\theta$ has a probability distribution; $P(\theta \in C) = 0.95$"   |
| **Interpretation**           | "95% of intervals constructed this way contain $\theta$"                          | "There is a 95% probability that $\theta$ lies in $C$ given the data" |
| **Requires prior?**          | No                                                                                | Yes                                                                   |
| **With enough data**         | The two often coincide (Bernstein–von Mises theorem)                              | Same                                                                  |

The Bayesian credible interval has the intuitively appealing interpretation: "given what I've observed, $\theta$ is in this range with 95% probability." The frequentist CI does _not_ have this interpretation (it's about repeated experiments, not about the observed data).

**Highest Posterior Density (HPD) interval:** the narrowest credible interval for a given level. For unimodal symmetric posteriors (e.g., Gaussian), HPD = equal-tailed interval.

---

### 7.4 Conjugate Priors

A prior $p(\theta)$ is **conjugate** to a likelihood $p(\mathcal{D} \mid \theta)$ if the posterior $p(\theta \mid \mathcal{D})$ is in the same family as the prior. This gives a **closed-form posterior** (no numerical integration needed).

| Likelihood                    | Conjugate prior      | Posterior            |
| ----------------------------- | -------------------- | -------------------- |
| Gaussian (known $\sigma^2$)   | Gaussian             | Gaussian             |
| Gaussian (unknown $\sigma^2$) | Normal-Inverse-Gamma | Normal-Inverse-Gamma |
| Bernoulli                     | Beta                 | Beta                 |
| Poisson                       | Gamma                | Gamma                |
| Multinomial                   | Dirichlet            | Dirichlet            |

Bayesian linear regression (Section 8) uses the Gaussian-Gaussian conjugacy, which is why it has a closed-form posterior.

---

## 8. Bayesian Linear Regression

### 8.1 The Model

**Likelihood** (same as [Week 07](../week07_likelihood/theory.md#6-mle-for-linear-regression-full-derivation)):

$$y_i \mid \mathbf{x}_i, \mathbf{w} \sim \mathcal{N}(\mathbf{x}_i^\top\mathbf{w}, \sigma^2) \implies p(\mathbf{y} \mid X, \mathbf{w}) = \mathcal{N}(\mathbf{y} \mid X\mathbf{w}, \sigma^2 I)$$

**Prior on weights:**

$$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1}I)$$

where $\alpha > 0$ is the **prior precision** (inverse prior variance). Larger $\alpha$ = tighter prior = more regularisation.

**Known quantities:** we treat $\sigma^2$ (noise variance) and $\alpha$ (prior precision) as known hyperparameters. (Estimating them from data is possible but requires more advanced techniques — empirical Bayes or full hierarchical models.)

Let $\beta = 1/\sigma^2$ be the **noise precision**, so the likelihood precision is $\beta$.

---

### 8.2 The Posterior Over Weights

Applying Bayes' theorem with the Gaussian-Gaussian conjugacy:

$$p(\mathbf{w} \mid \mathcal{D}) = \mathcal{N}(\mathbf{w} \mid \boldsymbol{\mu}_{\text{post}}, \Sigma_{\text{post}})$$

$$\boxed{\Sigma_{\text{post}} = (\alpha I + \beta X^\top X)^{-1}}$$

$$\boxed{\boldsymbol{\mu}_{\text{post}} = \beta\,\Sigma_{\text{post}}\,X^\top\mathbf{y}}$$

**Derivation sketch.** The log-posterior is:

$$\log p(\mathbf{w} \mid \mathcal{D}) = \log p(\mathbf{y} \mid X, \mathbf{w}) + \log p(\mathbf{w}) + \text{const}$$

$$= -\frac{\beta}{2}\|\mathbf{y} - X\mathbf{w}\|^2 - \frac{\alpha}{2}\|\mathbf{w}\|^2 + \text{const}$$

$$= -\frac{1}{2}\mathbf{w}^\top\underbrace{(\alpha I + \beta X^\top X)}_{=\Sigma_{\text{post}}^{-1}}\mathbf{w} + \mathbf{w}^\top\underbrace{\beta X^\top\mathbf{y}}_{=\Sigma_{\text{post}}^{-1}\boldsymbol{\mu}_{\text{post}}} + \text{const}$$

This is a quadratic form in $\mathbf{w}$, so the posterior is Gaussian. Completing the square yields $\boldsymbol{\mu}_{\text{post}}$ and $\Sigma_{\text{post}}$.

> **Notebook reference.** Cell 9 implements `BayesianLinearRegression` with `alpha=0.1, beta=4.0`. For $n = 20$ training points with true weights $(2, 0.5)$ and $\sigma = 0.5$, the posterior mean closely recovers the true parameters.

---

### 8.3 The Posterior Predictive Distribution

To predict $y_*$ at a new point $\mathbf{x}_*$, integrate out the weight uncertainty:

$$p(y_* \mid \mathbf{x}_*, \mathcal{D}) = \int p(y_* \mid \mathbf{x}_*, \mathbf{w})\,p(\mathbf{w} \mid \mathcal{D})\,d\mathbf{w}$$

Since both factors are Gaussian, the result is Gaussian:

$$\boxed{p(y_* \mid \mathbf{x}_*, \mathcal{D}) = \mathcal{N}\!\left(y_* \mid \boldsymbol{\mu}_{\text{post}}^\top\mathbf{x}_*, \, \sigma_*^2\right)}$$

where:

$$\sigma_*^2 = \underbrace{\frac{1}{\beta}}_{\text{aleatoric}} + \underbrace{\mathbf{x}_*^\top\Sigma_{\text{post}}\mathbf{x}_*}_{\text{epistemic}}$$

| Term                                                | Source                            | Behaviour with $n$          |
| --------------------------------------------------- | --------------------------------- | --------------------------- |
| $1/\beta = \sigma^2$                                | Noise variance (aleatoric)        | Constant                    |
| $\mathbf{x}_*^\top\Sigma_{\text{post}}\mathbf{x}_*$ | Parameter uncertainty (epistemic) | Decreases as $n \to \infty$ |

**The prediction interval is widest where:**
1. The model is extrapolating (far from training data).
2. The features are in a direction poorly constrained by the data (small eigenvalues of $X^\top X$).

> **Notebook reference.** Cell 9 plots $\pm 2\sigma_*$ bands around the posterior mean. Notice how the bands widen outside the training range $[0, 10]$ — this is epistemic uncertainty increasing during extrapolation. Cell 9 also draws 50 posterior weight samples and plots the corresponding lines, visually showing the same uncertainty.

---

### 8.4 Sequential Bayesian Updating

A fundamental property of Bayesian inference: **the posterior after seeing data $\mathcal{D}_1$ becomes the prior for the next batch $\mathcal{D}_2$**.

If $\mathcal{D} = \mathcal{D}_1 \cup \mathcal{D}_2$:

$$p(\mathbf{w} \mid \mathcal{D}) \propto p(\mathcal{D}_2 \mid \mathbf{w})\cdot\underbrace{p(\mathbf{w} \mid \mathcal{D}_1)}_{\text{new prior}} \propto p(\mathcal{D}_2 \mid \mathbf{w})\cdot p(\mathcal{D}_1 \mid \mathbf{w})\cdot p(\mathbf{w})$$

For BLR, this means after observing the first $k$ points:

$$\Sigma_k^{-1} = \alpha I + \beta\sum_{i=1}^{k}\mathbf{x}_i\mathbf{x}_i^\top, \qquad \boldsymbol{\mu}_k = \beta\,\Sigma_k\sum_{i=1}^{k}y_i\mathbf{x}_i$$

To add point $k+1$, apply a rank-1 update:

$$\Sigma_{k+1}^{-1} = \Sigma_k^{-1} + \beta\,\mathbf{x}_{k+1}\mathbf{x}_{k+1}^\top$$

**Interpretation:** with each new data point, the posterior precision increases (the ellipsoid shrinks), and the posterior mean shifts toward the MLE. The uncertainty band narrows.

> **Notebook reference.** Exercise 1 (Cell 14) asks you to implement sequential updating: feed points one at a time and plot how the posterior predictive interval shrinks from a wide prior to a narrow posterior.

---

### 8.5 Connection to Ridge Regression

The posterior mean of BLR is:

$$\boldsymbol{\mu}_{\text{post}} = (\alpha I + \beta X^\top X)^{-1}\beta X^\top\mathbf{y}$$

Compare with the Ridge solution ([Week 06](../../02_fundamentals/week06_regularization/theory.md#32-closed-form-solution)):

$$\hat{\mathbf{w}}_{\text{Ridge}} = (X^\top X + \lambda I)^{-1}X^\top\mathbf{y}$$

Setting $\lambda = \alpha/\beta = \alpha\sigma^2$:

$$\boxed{\boldsymbol{\mu}_{\text{post}} = \hat{\mathbf{w}}_{\text{Ridge}} \quad \text{with} \quad \lambda = \frac{\alpha}{\beta} = \frac{\sigma^2}{\tau^2}}$$

where $\tau^2 = 1/\alpha$ is the prior variance.

**Ridge gives the point estimate; BLR gives point estimate + uncertainty.** The posterior covariance $\Sigma_{\text{post}}$ is the "bonus" of the Bayesian approach — it tells you how confident to be in each coefficient.

> **The circle is complete.** [Week 03](../../02_fundamentals/week03_linear_models/theory.md#4-statistical-interpretation-of-linear-regression) introduced MSE. [Week 06](../../02_fundamentals/week06_regularization/theory.md#3-ridge-regression-l2-regularisation) added L2 penalty (Ridge). [Week 07](../week07_likelihood/theory.md#9-from-mle-to-map-the-bridge-to-regularisation) showed MSE = Gaussian NLL and Ridge = Gaussian MAP. Now Week 08 goes beyond MAP to the full posterior, providing uncertainty estimates that MAP alone cannot.

---

## 9. Calibration

### 9.1 What Is Calibration?

A probabilistic model is **well-calibrated** if its stated confidence levels match empirical coverage. Formally:

For a regression model that predicts $\mu(\mathbf{x})$ and $\sigma(\mathbf{x})$ and claims a $(1-\alpha)$ prediction interval $[\mu \pm z_{1-\alpha/2}\sigma]$:

$$\text{Well-calibrated:} \quad P\!\left(y \in [\mu - z\sigma, \, \mu + z\sigma]\right) \approx 1 - \alpha \quad \text{for all } \alpha$$

**Miscalibration types:**

| Type                | Symptom                                 | Consequence                                          |
| ------------------- | --------------------------------------- | ---------------------------------------------------- |
| **Overconfident**   | Stated 95% interval covers <95% of data | Dangerous: model is more wrong than it admits        |
| **Underconfident**  | Stated 95% interval covers >95% of data | Wasteful: intervals too wide, decisions too cautious |
| **Well-calibrated** | Coverage ≈ stated level                 | Trustworthy                                          |

> **Why calibration matters.** A model that says "95% confidence" but is only right 70% of the time is **misleading**. In high-stakes domains (medicine, safety), this can lead to catastrophic decisions. Calibration is the minimum requirement for trustworthy uncertainty estimates.

---

### 9.2 Reliability Diagrams

A **reliability diagram** (calibration plot) checks calibration across multiple confidence levels:

1. Choose a grid of confidence levels: e.g., $1-\alpha \in \{0.1, 0.2, \ldots, 0.99\}$.
2. For each level, compute the predicted interval and check what fraction of test points fall inside.
3. Plot **predicted confidence** ($x$-axis) vs. **empirical coverage** ($y$-axis).
4. A perfectly calibrated model lies on the **diagonal** $y = x$.

```python
confidence_levels = np.linspace(0.1, 0.99, 20)
empirical_coverages = []
for level in confidence_levels:
    z_crit = stats.norm.ppf((1 + level) / 2)
    in_interval = np.abs(z_scores) <= z_crit
    empirical_coverages.append(in_interval.mean())

plt.plot(confidence_levels, empirical_coverages, 'o-')
plt.plot([0, 1], [0, 1], 'r--', label='Perfect')
```

> **Notebook reference.** Cell 11 constructs a full calibration assessment: standardised residual histogram (should look $\mathcal{N}(0,1)$), Q-Q plot, and reliability diagram. For BLR on the synthetic data, the model should be well-calibrated (close to the diagonal).

---

### 9.3 Standardised Residuals and Q-Q Plots

**Standardised (z-score) residual** for prediction $i$:

$$z_i = \frac{y_i - \mu_i}{\sigma_i}$$

where $\mu_i$ and $\sigma_i$ are the predicted mean and standard deviation at $\mathbf{x}_i$.

If the model is well-specified and well-calibrated:
- $z_i \overset{\text{approx}}{\sim} \mathcal{N}(0, 1)$.
- $\bar{z} \approx 0$, $\text{std}(z) \approx 1$.
- The Q-Q plot (quantiles of $\{z_i\}$ vs. quantiles of $\mathcal{N}(0,1)$) is approximately linear.

**Diagnostic checks:**

| Observation             | Implication                                                   |
| ----------------------- | ------------------------------------------------------------- |
| $\text{std}(z) < 1$     | Model is **underconfident** (intervals too wide)              |
| $\text{std}(z) > 1$     | Model is **overconfident** (intervals too narrow)             |
| $\bar{z} \neq 0$        | Systematic bias in predictions                                |
| Heavy tails in Q-Q plot | Noise is not Gaussian (outliers) → use robust model ([Week 07](../week07_likelihood/theory.md#7-robust-regression-alternative-noise-models)) |
| Curvature in Q-Q plot   | Misspecified distribution (e.g., skewed noise)                |

---

### 9.4 Calibration for Classification

For classification with predicted probabilities $\hat{p}_i$:

$$\text{Well-calibrated:} \quad P(Y = 1 \mid \hat{p} = q) = q \quad \text{for all } q \in [0, 1]$$

"Among all examples where the model says 70% probability, approximately 70% should actually be positive."

**Reliability diagram (classification):**
- Bin predictions by $\hat{p}$ (e.g., bins $[0, 0.1), [0.1, 0.2), \ldots$).
- For each bin, compute the fraction of positives (empirical probability).
- Plot bin midpoint ($x$) vs. empirical fraction ($y$).

**Expected Calibration Error (ECE):**

$$\text{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

where $B_m$ is the $m$-th bin, $\text{acc}$ is the accuracy in the bin, and $\text{conf}$ is the mean predicted probability.

**Post-hoc calibration methods:**
- **Platt scaling:** fit a logistic regression on a held-out set to map uncalibrated logits to calibrated probabilities.
- **Temperature scaling:** divide logits by a scalar $T > 0$ before softmax; $T > 1$ softens predictions (reduces overconfidence).
- **Isotonic regression:** non-parametric monotone mapping.

> **Forward pointer.** Neural networks are notoriously poorly calibrated (Guo et al., 2017). Temperature scaling becomes critical in [Weeks 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#8-the-training-loop)–[18](../../06_sequence_models/week18_transformers/theory.md#9-training-considerations).

---

## 10. Prediction Intervals

### 10.1 Frequentist Prediction Intervals

A **prediction interval** for a new $y_*$ at $\mathbf{x}_*$ (under OLS with Gaussian noise):

$$\hat{y}_* \pm t_{n-d-1, 1-\alpha/2}\cdot\hat{\sigma}\sqrt{1 + \mathbf{x}_*^\top(X^\top X)^{-1}\mathbf{x}_*}$$

The $\sqrt{1 + \ldots}$ includes both the noise variance (the "1") and the parameter estimation uncertainty (the $\mathbf{x}_*^\top(X^\top X)^{-1}\mathbf{x}_*$ term). This is wider than the confidence interval for the mean, which is $\hat{y}_* \pm t \cdot \hat{\sigma}\sqrt{\mathbf{x}_*^\top(X^\top X)^{-1}\mathbf{x}_*}$.

| Interval type                                                     | Captures                                | Width                         |
| ----------------------------------------------------------------- | --------------------------------------- | ----------------------------- |
| **Confidence interval (for $\mathbb{E}[y_* \mid \mathbf{x}_*]$)** | Uncertainty in the mean prediction      | Narrower (epistemic only)     |
| **Prediction interval (for $y_*$)**                               | Uncertainty in a single new observation | Wider (epistemic + aleatoric) |

---

### 10.2 Bayesian Predictive Intervals

The Bayesian posterior predictive (Section 8.3) directly gives:

$$y_* \mid \mathbf{x}_*, \mathcal{D} \sim \mathcal{N}\!\left(\boldsymbol{\mu}_{\text{post}}^\top\mathbf{x}_*, \, \sigma^2 + \mathbf{x}_*^\top\Sigma_{\text{post}}\mathbf{x}_*\right)$$

A 95% predictive interval:

$$\boldsymbol{\mu}_{\text{post}}^\top\mathbf{x}_* \pm 1.96\sqrt{\sigma^2 + \mathbf{x}_*^\top\Sigma_{\text{post}}\mathbf{x}_*}$$

This is essentially the same as the frequentist prediction interval (they coincide for Gaussian BLR with known $\sigma^2$), but the Bayesian version has a cleaner interpretation: "95% probability that $y_*$ lies in this interval."

---

### 10.3 Conformal Prediction (Distribution-Free)

Both frequentist and Bayesian prediction intervals assume specific distributional forms (Gaussian noise). **Conformal prediction** provides **distribution-free** coverage guarantees:

**Split conformal prediction:**

1. Split the data into training and calibration sets.
2. Fit the model on the training set.
3. Compute residuals $r_i = |y_i - \hat{y}_i|$ on the calibration set.
4. Let $q$ be the $(1-\alpha)(1 + 1/n_{\text{cal}})$-quantile of $\{r_i\}$.
5. For a new $\mathbf{x}_*$: the prediction interval is $[\hat{y}_* - q, \, \hat{y}_* + q]$.

**Guarantee:** $P(y_* \in [\hat{y}_* - q, \, \hat{y}_* + q]) \geq 1 - \alpha$ under the only assumption that the data are **exchangeable** (i.i.d. is a special case).

**Advantages:** no distributional assumptions, finite-sample guarantee, works with any base model (including neural networks).

**Limitation:** constant-width intervals (the basic version doesn't adapt to heteroscedastic regions). Adaptive conformal methods exist but are more involved.

> **Notebook reference.** Exercise 3 asks you to implement conformal prediction and compare interval width to BLR intervals.

---

## 11. Ensemble Uncertainty

Training multiple models and examining the spread of their predictions is a simple but effective way to estimate epistemic uncertainty:

**Procedure:**
1. Train $M$ models (e.g., $M = 10$), each on a different bootstrap subsample of the training data.
2. For a test point $\mathbf{x}_*$, compute predictions $\hat{y}_*^{(1)}, \ldots, \hat{y}_*^{(M)}$.
3. **Ensemble mean:** $\bar{y}_* = \frac{1}{M}\sum_m \hat{y}_*^{(m)}$ (improved prediction via averaging).
4. **Ensemble variance:** $\hat{\sigma}_{\text{ens}}^2 = \frac{1}{M-1}\sum_m (\hat{y}_*^{(m)} - \bar{y}_*)^2$ (epistemic uncertainty estimate).

**Why it works:** each bootstrap subsample produces a slightly different model. Where all models agree, epistemic uncertainty is low. Where they disagree, the model is uncertain — typically in data-sparse regions.

**Connection to BLR:** for linear models, the ensemble spread converges to the posterior variance $\mathbf{x}_*^\top\Sigma_{\text{post}}\mathbf{x}_*$ as $M \to \infty$ (both are measuring the same thing — parameter uncertainty propagated through the model).

> **Notebook reference.** Exercise 2 (Cell 15) trains 10 bootstrap linear models and compares ensemble error bars to BLR posterior predictive bands.
>
> **Suggested experiment.** Plot ensemble variance as a function of $x$ and overlay it with BLR epistemic uncertainty $\mathbf{x}_*^\top\Sigma_{\text{post}}\mathbf{x}_*$. They should be nearly proportional.

> **Forward pointer.** Ensembles of neural networks ([Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#9-regularisation-in-neural-networks)+) are one of the most practical methods for uncertainty estimation in deep learning — "deep ensembles" (Lakshminarayanan et al., 2017).

---

## 12. Connections to the Rest of the Course

| Week                            | Connection                                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| **[Week 03](../../02_fundamentals/week03_linear_models/theory.md#3-linear-regression) (Linear Models)**     | OLS = MLE point estimate; BLR adds the posterior around it                                 |
| **[Week 06](../../02_fundamentals/week06_regularization/theory.md#35-bayesian-interpretation) (Regularisation)**    | Ridge = posterior mean of BLR; $\lambda = \alpha\sigma^2$                                  |
| **[Week 07](../week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle) (Likelihood)**        | MLE = the "no prior" special case; NLL is the likelihood term in the posterior             |
| **[Week 09](../week09_time_series/theory.md#6-autoregressive-models-arp) (Time Series)**       | Bayesian AR models; uncertainty in multi-step forecasts grows over time                    |
| **[Week 10](../week10_surrogate_models/theory.md#5-gp-posterior-regression) (Surrogate Models)**  | Gaussian processes are infinite-dimensional BLR; posterior predictive = GP mean + variance |
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md#9-regularisation-in-neural-networks) (NNs)**               | Weight decay = Gaussian prior; Bayesian neural networks extend BLR to non-linear models    |
| **[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md#4-learning-rate-schedules) (Training at Scale)** | MC Dropout as approximate Bayesian inference; temperature scaling for calibration          |
| **[Week 16](../../05_deep_learning/week16_regularization_dl/theory.md#3-dropout) (Regularisation DL)** | Dropout at test time produces ensemble-like uncertainty estimates                          |
| **[Week 20](../../08_deployment/week20_deployment/theory.md) (Deployment)**        | Calibration and uncertainty reporting are requirements for responsible deployment          |

---

## 13. Notebook Reference Guide

| Cell                 | Section                                     | What it demonstrates                                                       | Theory reference |
| -------------------- | ------------------------------------------- | -------------------------------------------------------------------------- | ---------------- |
| 5 (Monte Carlo)      | MC estimation of $\mathbb{E}[X^2 + \sin X]$ | 10,000 samples; histogram of $f(X)$; 95% percentile interval               | Section 6        |
| 7 (Bootstrap)        | Bootstrap for mean and median               | 2,000 resamples of exponential data ($n = 50$); percentile CIs             | Section 5        |
| 9 (BLR)              | Bayesian linear regression                  | Closed-form posterior; $\pm 2\sigma$ predictive band; 50 posterior samples | Section 8        |
| 11 (Calibration)     | Reliability diagram                         | Standardised residuals, Q-Q plot, coverage vs. confidence                  | Section 9        |
| Exercise 1 (Cell 14) | Sequential updating                         | Feed points one at a time; watch the posterior shrink                      | Section 8.4      |
| Exercise 2 (Cell 15) | Ensemble uncertainty                        | 10 bootstrap models; error bars vs. BLR bands                              | Section 11       |
| Exercise 4 (Cell 16) | Real-data calibration                       | Ridge + prediction intervals on a real dataset; reliability diagram        | Section 9.2      |

**Suggested modifications:**

| Modification                                                      | What it reveals                                                                         |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Set BLR `alpha=10.0` (strong prior) vs `alpha=0.001` (weak prior) | Prior dominance vs. data dominance; strong prior → underfitting, weak → near-OLS        |
| Reduce training data to $n = 5$                                   | Epistemic uncertainty becomes dominant; posterior width is large                        |
| Add heteroscedastic noise: $\sigma(x) = 0.2 + 0.3x$               | Homoscedastic model's intervals are too wide for small $x$ and too narrow for large $x$ |
| Change to non-Gaussian noise (e.g., Student-$t$)                  | Q-Q plot reveals heavy tails; standardised residuals fail the $\mathcal{N}(0,1)$ check  |
| Compare $B = 100$ vs $B = 10{,}000$ bootstrap samples             | With $B = 100$ the CI is noisy; with $B = 10{,}000$ it stabilises                       |
| Implement temperature scaling on a classifier                     | Post-hoc calibration improves reliability diagram dramatically                          |

---

## 14. Symbol Reference

| Symbol                                  | Name                           | Meaning                                                                     |
| --------------------------------------- | ------------------------------ | --------------------------------------------------------------------------- |
| $\theta$                                | Parameters                     | Generic parameter vector                                                    |
| $\mathcal{D}$                           | Dataset                        | Observed data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$                             |
| $p(\theta)$                             | Prior                          | Belief about $\theta$ before seeing data                                    |
| $p(\mathcal{D} \mid \theta)$            | Likelihood                     | Probability of data given parameters                                        |
| $p(\theta \mid \mathcal{D})$            | Posterior                      | Updated belief after seeing data                                            |
| $p(y_* \mid \mathbf{x}_*, \mathcal{D})$ | Posterior predictive           | Prediction distribution at $\mathbf{x}_*$                                   |
| $p(\mathcal{D})$                        | Evidence (marginal likelihood) | $\int p(\mathcal{D} \mid \theta)p(\theta)d\theta$                           |
| $\alpha$                                | Prior precision                | $= 1/\tau^2$; precision of the Gaussian weight prior                        |
| $\beta$                                 | Noise precision                | $= 1/\sigma^2$; precision of the observation noise                          |
| $\sigma^2$                              | Noise variance                 | Aleatoric uncertainty                                                       |
| $\tau^2$                                | Prior variance                 | $= 1/\alpha$; controls regularisation strength                              |
| $\boldsymbol{\mu}_{\text{post}}$        | Posterior mean                 | $= \beta\Sigma_{\text{post}}X^\top\mathbf{y}$                               |
| $\Sigma_{\text{post}}$                  | Posterior covariance           | $= (\alpha I + \beta X^\top X)^{-1}$                                        |
| $\sigma_*^2$                            | Predictive variance            | $\sigma^2 + \mathbf{x}_*^\top\Sigma_{\text{post}}\mathbf{x}_*$              |
| $z_i$                                   | Standardised residual          | $(y_i - \mu_i)/\sigma_i$                                                    |
| $I(\theta)$                             | Fisher information             | Curvature of log-likelihood; $-\mathbb{E}[\partial^2\ell/\partial\theta^2]$ |
| $\text{SE}(\hat{\theta})$               | Standard error                 | Estimated std of the sampling distribution                                  |
| $t_{n-d-1, q}$                          | Student-$t$ quantile           | $q$-quantile with $n - d - 1$ degrees of freedom                            |
| $B$                                     | Number of bootstrap samples    | Typically 1000–10,000                                                       |
| $T_b^*$                                 | Bootstrap statistic            | Value of the statistic on the $b$-th bootstrap sample                       |
| $N$                                     | Number of MC samples           | Typically 1000–100,000                                                      |
| $M$                                     | Number of ensemble models      | Typically 5–20                                                              |
| $\text{ECE}$                            | Expected Calibration Error     | Weighted average of per-bin calibration gaps                                |
| $q$                                     | Conformal quantile             | Quantile of calibration residuals                                           |

---

## 15. References

1. Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." *Annals of Statistics*, 7(1), 1–26. — The original bootstrap paper.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 3. Springer. — Bayesian linear regression derivation; posterior predictive; sequential updating.
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapters 3, 5, 7. MIT Press. — Comprehensive Bayesian inference; conjugate priors; BLR.
4. Gelman, A., Carlin, J. B., Stern, H. S., et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press. — The standard reference for Bayesian methods; PPCs; hierarchical models.
5. Wasserman, L. (2004). *All of Statistics*, Chapters 6, 8, 11. Springer. — Clear treatment of confidence intervals, bootstrap, and Bayesian methods.
6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." *ICML*. — Neural networks are miscalibrated; temperature scaling.
7. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*. — Deep ensembles for uncertainty.
8. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. — Conformal prediction theory.
9. Angelopoulos, A. N. & Bates, S. (2023). "Conformal Prediction: A Gentle Introduction." *Foundations and Trends in ML*. — Modern, accessible introduction to conformal methods.
10. Davidson-Pilon, C. (2015). *Bayesian Methods for Hackers*. Addison-Wesley. — Practical PyMC examples; excellent for building intuition.
11. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Chapters 7–8. Springer. — Cross-validation, bootstrap, and model assessment.
12. Bernstein–von Mises theorem. See van der Vaart, A. (1998). *Asymptotic Statistics*, Chapter 10. Cambridge University Press. — Formal result on posterior convergence to the sampling distribution.
