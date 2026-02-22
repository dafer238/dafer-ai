# Time-Series Fundamentals: Seasonality & Autocorrelation

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Anatomy of a Time Series](#2-anatomy-of-a-time-series)
   - 2.1 [Trend](#21-trend)
   - 2.2 [Seasonality](#22-seasonality)
   - 2.3 [Cyclical Variation](#23-cyclical-variation)
   - 2.4 [Irregular Component (Noise)](#24-irregular-component-noise)
3. [Decomposition](#3-decomposition)
   - 3.1 [Additive Decomposition](#31-additive-decomposition)
   - 3.2 [Multiplicative Decomposition](#32-multiplicative-decomposition)
   - 3.3 [Choosing Between Additive and Multiplicative](#33-choosing-between-additive-and-multiplicative)
   - 3.4 [STL Decomposition](#34-stl-decomposition)
4. [Autocorrelation](#4-autocorrelation)
   - 4.1 [Autocorrelation Function (ACF)](#41-autocorrelation-function-acf)
   - 4.2 [Partial Autocorrelation Function (PACF)](#42-partial-autocorrelation-function-pacf)
   - 4.3 [Reading ACF and PACF Plots](#43-reading-acf-and-pacf-plots)
5. [Stationarity](#5-stationarity)
   - 5.1 [Definition and Importance](#51-definition-and-importance)
   - 5.2 [Augmented Dickey–Fuller Test](#52-augmented-dickeyfuller-test)
   - 5.3 [Achieving Stationarity by Differencing](#53-achieving-stationarity-by-differencing)
6. [Autoregressive Models — AR(p)](#6-autoregressive-models--arp)
   - 6.1 [Definition](#61-definition)
   - 6.2 [Stationarity Conditions](#62-stationarity-conditions)
   - 6.3 [ACF and PACF Signatures](#63-acf-and-pacf-signatures)
7. [Moving Average Models — MA(q)](#7-moving-average-models--maq)
   - 7.1 [Definition](#71-definition)
   - 7.2 [Invertibility](#72-invertibility)
   - 7.3 [ACF and PACF Signatures](#73-acf-and-pacf-signatures-1)
8. [ARMA and ARIMA](#8-arma-and-arima)
   - 8.1 [ARMA(p, q)](#81-armap-q)
   - 8.2 [ARIMA(p, d, q)](#82-arimap-d-q)
   - 8.3 [The Box–Jenkins Methodology](#83-the-boxjenkins-methodology)
   - 8.4 [Information Criteria for Model Selection](#84-information-criteria-for-model-selection)
9. [Seasonal ARIMA — SARIMA](#9-seasonal-arima--sarima)
   - 9.1 [The Model](#91-the-model)
   - 9.2 [Seasonal Differencing](#92-seasonal-differencing)
   - 9.3 [Identifying Seasonal Orders](#93-identifying-seasonal-orders)
10. [Forecast Evaluation](#10-forecast-evaluation)
    - 10.1 [Walk-Forward Cross-Validation](#101-walk-forward-cross-validation)
    - 10.2 [Forecast Error Metrics](#102-forecast-error-metrics)
    - 10.3 [Residual Diagnostics](#103-residual-diagnostics)
    - 10.4 [Forecast Uncertainty](#104-forecast-uncertainty)
11. [Connections to the Rest of the Course](#11-connections-to-the-rest-of-the-course)
12. [Notebook Reference Guide](#12-notebook-reference-guide)
13. [Symbol Reference](#13-symbol-reference)
14. [References](#14-references)

---

## 1. Scope and Purpose

[Week 08](../week08_uncertainty/theory.md) established how to quantify uncertainty in static settings (bootstrap, Bayesian intervals, calibration). This week applies those ideas to **data ordered in time**, where consecutive observations are no longer independent.

Most real-world data streams — sensor readings, financial returns, sales figures, climate records — are time series. The key observation is that **neighbouring values are correlated**: today's temperature is similar to yesterday's, and knowing this correlation allows us to forecast tomorrow's. The classical statistical toolkit for exploiting this structure is the **ARIMA** family.

**Goals for this week:**
1. Decompose a time series into trend, seasonality, and residual.
2. Diagnose temporal correlation via ACF and PACF.
3. Test for and achieve **stationarity** — the prerequisite for ARIMA models.
4. Identify, fit, and diagnose ARIMA and SARIMA models.
5. Evaluate forecasts honestly using walk-forward cross-validation.

**Prerequisites.** [Week 03](../../02_fundamentals/week03_linear_models/theory.md) (linear regression — AR models generalise it), [Week 06](../../02_fundamentals/week06_regularization/theory.md) (time-series cross-validation), [[Week 07](../week07_likelihood/theory.md)](../week07_likelihood/theory.md)–[08](../week08_uncertainty/theory.md) (likelihood, residual diagnostics, calibration).

---

## 2. Anatomy of a Time Series

A time series $\{Y_t\}_{t=1}^{T}$ is a sequence of observations indexed by time. Its structure is typically described in terms of four components.

### 2.1 Trend

A long-term increase or decrease in the level.

$$T_t = \text{smooth, slowly varying function of } t$$

Examples: GDP growth, global temperature rise, cumulative CO₂ emissions.

A **linear trend** is $T_t = a + bt$. More flexible trends can be estimated with moving averages or LOESS (locally weighted scatterplot smoothing).

---

### 2.2 Seasonality

A **fixed, periodic** pattern that repeats with a known period $s$:

$$S_t = S_{t+s} \quad \text{for all } t$$

| Period              | Example                                       |
| ------------------- | --------------------------------------------- |
| $s = 12$ (monthly)  | Airline passengers peak in summer             |
| $s = 7$ (daily)     | Retail sales peak on weekends                 |
| $s = 24$ (hourly)   | Electricity demand peaks mid-afternoon        |
| $s = 4$ (quarterly) | Government spending spikes at fiscal year-end |

The seasonal component sums to approximately zero (additive) or averages to approximately one (multiplicative) over each period:

$$\sum_{j=1}^{s} S_j \approx 0 \quad (\text{additive}), \qquad \frac{1}{s}\sum_{j=1}^{s} S_j \approx 1 \quad (\text{multiplicative})$$

---

### 2.3 Cyclical Variation

Fluctuations that are **not of fixed period** — their length varies from cycle to cycle. Business cycles (expansion → recession → recovery) are the classic example. Because the period is unknown and variable, cycles are much harder to model than seasonality.

> **Seasonality vs. cycles.** Seasonality has a fixed, known period (e.g., 12 months). Cycles have variable, often long and unknown periods (e.g., 3–10 years for business cycles). In practice, cycles are often absorbed into the trend or modelled separately with domain-specific models.

---

### 2.4 Irregular Component (Noise)

The residual after removing trend, seasonality, and cycles:

$$R_t = Y_t - T_t - S_t \quad (\text{additive})$$

If the model is well-specified, $R_t$ should be approximately **white noise**: zero mean, constant variance, no autocorrelation.

---

## 3. Decomposition

### 3.1 Additive Decomposition

$$\boxed{Y_t = T_t + S_t + R_t}$$

Assumes the seasonal fluctuations have **constant amplitude** regardless of the level of the series. Appropriate when the series has roughly the same seasonal swing whether the overall level is high or low.

**Classical method** (moving average):
1. Estimate the trend $\hat{T}_t$ with a centred moving average of length $s$.
2. Detrend: $Y_t - \hat{T}_t$.
3. Estimate seasonal factors $\hat{S}_j$ ($j = 1, \ldots, s$) by averaging the detrended values for each season.
4. Residual: $\hat{R}_t = Y_t - \hat{T}_t - \hat{S}_t$.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(series, model='additive', period=12)
```

---

### 3.2 Multiplicative Decomposition

$$\boxed{Y_t = T_t \times S_t \times R_t}$$

Assumes seasonal fluctuations are **proportional to the level**: when the series is higher, the seasonal swings are larger. This is extremely common in economic and financial data.

**Log trick.** Taking logarithms converts multiplicative to additive:

$$\log Y_t = \log T_t + \log S_t + \log R_t$$

This is why the notebook works with $\log(\text{AirPassengers})$ — it stabilises the variance and makes the additive model appropriate.

```python
result = seasonal_decompose(series, model='multiplicative', period=12)
```

---

### 3.3 Choosing Between Additive and Multiplicative

| Diagnostic         | Additive                        | Multiplicative                                  |
| ------------------ | ------------------------------- | ----------------------------------------------- |
| Seasonal amplitude | Constant over time              | Grows with level                                |
| Visual check       | Parallel seasonal bands         | Fanning-out bands                               |
| Residual variance  | Compare $\text{Var}(\hat{R}_t)$ | Choose the model with smaller residual variance |
| Log transformation | Not needed                      | Equivalent to additive on $\log Y_t$            |

> **Notebook reference.** Cell 7 decomposes AirPassengers both ways. The multiplicative residuals are more uniform, confirming that seasonal amplitude grows with the passenger count. Exercise 1 asks you to compute residual variance for both and confirm this quantitatively.

---

### 3.4 STL Decomposition

**STL** (Seasonal-Trend decomposition using LOESS; Cleveland et al., 1990) uses local regression instead of simple moving averages. Advantages over classical decomposition:

- **Robust to outliers** — LOESS uses iterative reweighting.
- **Allows the seasonal component to change slowly over time** (controlled by `seasonal` parameter).
- **The trend smoother bandwidth is adjustable** (`trend` parameter).

```python
from statsmodels.tsa.seasonal import STL
stl = STL(series, period=12, robust=True)
result = stl.fit()
```

---

## 4. Autocorrelation

### 4.1 Autocorrelation Function (ACF)

The **autocorrelation** at lag $k$ measures the linear correlation between $Y_t$ and $Y_{t-k}$:

$$\boxed{\rho_k = \frac{\text{Cov}(Y_t, Y_{t-k})}{\text{Var}(Y_t)} = \frac{\gamma_k}{\gamma_0}}$$

where $\gamma_k = \text{Cov}(Y_t, Y_{t-k})$ is the **autocovariance** at lag $k$.

Properties:
- $\rho_0 = 1$ (always).
- $|\rho_k| \leq 1$.
- $\rho_k = \rho_{-k}$ (symmetric in $k$ for stationary series).

**Interpretation.** If $\rho_1 = 0.8$, each value explains 80% of the variance of the next value (in terms of linear correlation). High ACF at lag 12 for monthly data indicates seasonality.

**Significance bands.** Under the null hypothesis of white noise:

$$\rho_k \approx 0 \pm \frac{1.96}{\sqrt{T}}$$

Lags where $|\hat{\rho}_k|$ exceeds this band are **statistically significant** (at the 5% level).

---

### 4.2 Partial Autocorrelation Function (PACF)

The **partial autocorrelation** at lag $k$, denoted $\phi_{kk}$, measures the correlation between $Y_t$ and $Y_{t-k}$ **after removing the linear effect of all intermediate lags** $Y_{t-1}, Y_{t-2}, \ldots, Y_{t-k+1}$.

Formally, $\phi_{kk}$ is the coefficient of $Y_{t-k}$ in the regression:

$$Y_t = \phi_{k1}Y_{t-1} + \phi_{k2}Y_{t-2} + \cdots + \phi_{kk}Y_{t-k} + \epsilon_t$$

**Why PACF matters.** The ACF at lag 2 for an AR(1) process is non-zero (because $Y_t$ correlates with $Y_{t-2}$ via the chain $Y_t \to Y_{t-1} \to Y_{t-2}$). The PACF strips out this indirect effect: $\phi_{22} = 0$ for AR(1), revealing the **true lag structure**.

---

### 4.3 Reading ACF and PACF Plots

This is the critical diagnostic skill for time-series model identification.

| Process            | ACF pattern                                  | PACF pattern                  |
| ------------------ | -------------------------------------------- | ----------------------------- |
| **AR(p)**          | Decays gradually (exponential or sinusoidal) | **Cuts off after lag $p$**    |
| **MA(q)**          | **Cuts off after lag $q$**                   | Decays gradually              |
| **ARMA(p, q)**     | Decays after lag $q$                         | Decays after lag $p$          |
| **White noise**    | All within significance bands                | All within significance bands |
| **Seasonal AR/MA** | Spikes at multiples of $s$                   | Spikes at multiples of $s$    |

"Cuts off" means the values drop to zero (within the significance band) after a specific lag. "Decays" means the values taper off gradually.

**Visual example (from the notebook):**

| Generated process                     | Expected ACF                                      | Expected PACF                        |
| ------------------------------------- | ------------------------------------------------- | ------------------------------------ |
| AR(2): $\phi = [0.7, -0.3]$           | Damped oscillation (because of negative $\phi_2$) | Significant at lags 1, 2; zero after |
| MA(1): $\theta = 0.8$                 | Significant at lag 1 only; zero after             | Gradual exponential decay            |
| ARMA(1,1): $\phi = 0.6, \theta = 0.4$ | Gradual decay starting from lag 1                 | Gradual decay starting from lag 1    |

> **Notebook reference.** Cell 9 plots ACF/PACF for raw AirPassengers (strong seasonal spikes at lags 12, 24, 36) and for the log-differenced series (much cleaner). Cell 13 generates the three synthetic processes above and shows their ACF/PACF signatures.

---

## 5. Stationarity

### 5.1 Definition and Importance

A time series $\{Y_t\}$ is **(weakly) stationary** if:

1. **Constant mean:** $\mathbb{E}[Y_t] = \mu$ for all $t$.
2. **Constant variance:** $\text{Var}(Y_t) = \sigma^2$ for all $t$.
3. **Autocovariance depends only on lag:** $\text{Cov}(Y_t, Y_{t-k}) = \gamma_k$ (not on $t$).

> **Why stationarity matters.** The ARIMA family assumes that the (differenced) series is stationary. If the mean or variance changes over time, the estimated coefficients are meaningless — they describe a process that no longer exists. Non-stationarity leads to **spurious correlations** and unreliable forecasts.

**Common sources of non-stationarity:**

| Source                | Effect on series         | Fix                                                  |
| --------------------- | ------------------------ | ---------------------------------------------------- |
| **Trend**             | Mean changes over time   | Differencing $\nabla Y_t = Y_t - Y_{t-1}$            |
| **Seasonality**       | Periodic shifts in level | Seasonal differencing $\nabla_s Y_t = Y_t - Y_{t-s}$ |
| **Changing variance** | Variance grows/shrinks   | Log transform, Box–Cox                               |
| **Structural break**  | Abrupt regime change     | Split series or include intervention variables       |

---

### 5.2 Augmented Dickey–Fuller Test

The **ADF test** is the standard test for stationarity.

**Null hypothesis $H_0$:** the series has a **unit root** (is non-stationary).
**Alternative $H_1$:** the series is stationary.

The test fits the regression:

$$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p}\delta_i\Delta Y_{t-i} + \epsilon_t$$

and tests $H_0{:}\ \gamma = 0$ (unit root) against $H_1{:}\ \gamma < 0$ (stationary).

**Decision rule:** reject $H_0$ if the **ADF statistic** is more negative than the critical value at the chosen significance level, or equivalently if **p-value < 0.05**.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series.dropna(), autolag='AIC')
adf_stat, p_value = result[0], result[1]
print(f'ADF={adf_stat:.3f}, p={p_value:.4f}')
# p < 0.05 → STATIONARY (reject H0)
# p ≥ 0.05 → NON-STATIONARY (fail to reject H0)
```

> **Notebook reference.** Cell 11 applies `adf_test` to raw AirPassengers (non-stationary), log(Passengers) (non-stationary), first-differenced log (stationary), and double-differenced log (stationary). The p-value drops below 0.05 after first-order differencing.

---

### 5.3 Achieving Stationarity by Differencing

**First-order differencing** removes a linear trend:

$$\nabla Y_t = Y_t - Y_{t-1}$$

**Seasonal differencing** (period $s$) removes seasonal patterns:

$$\nabla_s Y_t = Y_t - Y_{t-s}$$

For AirPassengers with $s = 12$:

$$\nabla_{12}\nabla\log Y_t = (\log Y_t - \log Y_{t-1}) - (\log Y_{t-12} - \log Y_{t-13})$$

This removes both the trend (first difference) and the seasonality (seasonal difference), producing a stationary series suitable for ARIMA modelling.

**How many differences?**
- Apply ADF test after each difference.
- Typically $d \leq 2$ regular differences and $D \leq 1$ seasonal differences suffice.
- **Over-differencing** introduces artificial negative autocorrelation — watch for a large negative spike at lag 1 in the ACF.

---

## 6. Autoregressive Models — AR(p)

### 6.1 Definition

An **autoregressive model of order $p$** expresses the current value as a linear combination of its past $p$ values plus noise:

$$\boxed{Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \epsilon_t}$$

where $\epsilon_t \sim \text{WN}(0, \sigma^2)$ (white noise) and $\phi_1, \ldots, \phi_p$ are the AR coefficients.

**Using the backshift operator** $B$ (where $BY_t = Y_{t-1}$):

$$\Phi(B)Y_t = c + \epsilon_t \qquad \text{where} \quad \Phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$$

**Interpretation.** An AR(1) model $Y_t = \phi_1 Y_{t-1} + \epsilon_t$ is a first-order linear recurrence with noise. With $|\phi_1| < 1$, any perturbation decays geometrically — the process has "memory" but reverts to the mean. This is the time-series analogue of simple linear regression, where the predictor is the series' own past.

---

### 6.2 Stationarity Conditions

An AR($p$) process is stationary if and only if all roots of the characteristic polynomial $\Phi(z) = 1 - \phi_1 z - \cdots - \phi_p z^p$ lie **outside the unit circle** (i.e., $|z| > 1$).

**For AR(1):** stationarity requires $|\phi_1| < 1$.

**For AR(2):** stationarity requires:

$$\phi_1 + \phi_2 < 1, \quad \phi_2 - \phi_1 < 1, \quad |\phi_2| < 1$$

If the roots are complex, the ACF has a **damped sinusoidal** pattern (oscillatory behaviour). Real roots give a purely exponential decay.

---

### 6.3 ACF and PACF Signatures

| AR(p) property      | ACF                                              | PACF                               |
| ------------------- | ------------------------------------------------ | ---------------------------------- |
| **Shape**           | Gradual decay (exponential or damped sinusoidal) | **Cuts off after lag $p$**         |
| **Identifying $p$** | Not directly readable                            | Last significant lag in PACF = $p$ |

> **The PACF "cut-off" is the signature of AR models.** If you see the PACF drop to zero after lag 2 while the ACF decays gradually, the process is AR(2).

---

## 7. Moving Average Models — MA(q)

### 7.1 Definition

A **moving average model of order $q$** expresses the current value as a linear combination of current and past noise terms:

$$\boxed{Y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q}}$$

Using the backshift operator:

$$Y_t = \mu + \Theta(B)\epsilon_t \qquad \text{where} \quad \Theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$$

**Interpretation.** An MA(1) model says today's value is a weighted average of today's shock and yesterday's shock. The process has a **finite memory** of exactly $q$ time steps — $\text{Cov}(Y_t, Y_{t-k}) = 0$ for $k > q$.

---

### 7.2 Invertibility

An MA($q$) process is **invertible** if all roots of $\Theta(z) = 0$ lie outside the unit circle. Invertibility ensures that the MA process can be rewritten as an infinite-order AR process:

$$\epsilon_t = \sum_{j=0}^{\infty}\pi_j Y_{t-j}$$

Invertibility is necessary for unique parameterisation (different MA coefficients can produce the same autocorrelation structure; invertibility selects the canonical one).

**For MA(1):** invertibility requires $|\theta_1| < 1$.

---

### 7.3 ACF and PACF Signatures

| MA(q) property      | ACF                               | PACF                                             |
| ------------------- | --------------------------------- | ------------------------------------------------ |
| **Shape**           | **Cuts off after lag $q$**        | Gradual decay (exponential or damped sinusoidal) |
| **Identifying $q$** | Last significant lag in ACF = $q$ | Not directly readable                            |

> **The ACF "cut-off" is the signature of MA models.** If you see the ACF drop to zero after lag 1 while the PACF decays, the process is MA(1).

---

## 8. ARMA and ARIMA

### 8.1 ARMA(p, q)

Combining AR and MA gives the **ARMA(p, q)** model:

$$\boxed{\Phi(B)Y_t = c + \Theta(B)\epsilon_t}$$

$$Y_t = c + \phi_1 Y_{t-1} + \cdots + \phi_p Y_{t-p} + \epsilon_t + \theta_1\epsilon_{t-1} + \cdots + \theta_q\epsilon_{t-q}$$

**Key property.** Both the ACF and PACF decay gradually for ARMA — neither cuts off cleanly. This makes order identification harder than for pure AR or MA. In practice, use information criteria (AIC/BIC) to compare candidate models.

**Stationarity and invertibility.** ARMA(p,q) is stationary if the AR polynomial $\Phi(z)$ has all roots outside the unit circle, and invertible if the MA polynomial $\Theta(z)$ has all roots outside the unit circle.

---

### 8.2 ARIMA(p, d, q)

Most real-world series are not stationary in levels. **ARIMA** (Autoregressive Integrated Moving Average) models the **differenced** series:

$$\boxed{\Phi(B)\nabla^d Y_t = c + \Theta(B)\epsilon_t}$$

where $\nabla^d = (1 - B)^d$ is the $d$-th order differencing operator.

**The three parameters:**

| Parameter | Name              | Purpose                                       |
| --------- | ----------------- | --------------------------------------------- |
| $p$       | AR order          | Number of lagged $Y$ terms                    |
| $d$       | Integration order | Number of differences to achieve stationarity |
| $q$       | MA order          | Number of lagged $\epsilon$ terms             |

**Special cases:**

| Model          | Equivalent to                            |
| -------------- | ---------------------------------------- |
| ARIMA(0, 0, 0) | White noise                              |
| ARIMA(1, 0, 0) | AR(1)                                    |
| ARIMA(0, 0, 1) | MA(1)                                    |
| ARIMA(0, 1, 0) | Random walk $Y_t = Y_{t-1} + \epsilon_t$ |
| ARIMA(0, 1, 1) | Exponential smoothing (SES)              |
| ARIMA(1, 1, 1) | Often a good "default" starting model    |

---

### 8.3 The Box–Jenkins Methodology

The classical procedure (Box & Jenkins, 1970) for building an ARIMA model:

**Step 1. Identification.**
1. Plot the series. Look for trend, seasonality, changing variance.
2. Transform if needed (log, Box–Cox) to stabilise variance.
3. Difference until stationary (ADF test confirms).
4. Plot ACF and PACF of the stationary series.
5. Use the ACF/PACF signatures (Section 4.3) to propose candidate orders $(p, q)$.

**Step 2. Estimation.**
Fit the model using maximum likelihood (MLE) or conditional least squares:

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(series, order=(p, d, q))
result = model.fit()
```

**Step 3. Diagnostics.**
Check that the residuals are white noise:
- **Ljung–Box test:** $H_0$: residuals are uncorrelated up to lag $K$. p > 0.05 → good.
- **Residual ACF:** all lags within significance bands.
- **Q-Q plot:** residuals approximately normal.
- **Residual vs. time:** no patterns, constant variance.

If diagnostics fail, return to Step 1 and try different orders.

**Step 4. Forecast.**
Use the fitted model to produce point forecasts and prediction intervals.

> **Notebook reference.** Cell 15 fits ARIMA(1,1,1) on $\log(\text{AirPassengers})$ and runs `plot_diagnostics()` — the four-panel display includes standardised residuals, histogram + KDE, Q-Q plot, and correlogram (ACF of residuals).

---

### 8.4 Information Criteria for Model Selection

When the ACF/PACF don't give clear guidance, compare multiple models using:

**Akaike Information Criterion (AIC):**

$$\text{AIC} = -2\log\hat{L} + 2k$$

**Bayesian Information Criterion (BIC):**

$$\text{BIC} = -2\log\hat{L} + k\log n$$

where $\hat{L}$ is the maximised likelihood and $k$ is the number of parameters.

| Criterion | Penalty for complexity          | Tendency                                                    |
| --------- | ------------------------------- | ----------------------------------------------------------- |
| AIC       | $2k$ (lighter)                  | Selects model with best predictive accuracy; may overfit    |
| BIC       | $k\log n$ (heavier for $n > 7$) | Selects true model (if in candidate set); more parsimonious |

**Rule:** lower is better. For forecasting, AIC is generally preferred; for model identification, BIC.

> **Practical tip.** `auto_arima` from the `pmdarima` library automates the search over $(p, d, q)$ by trying many combinations and selecting by AIC:
> ```python
> import pmdarima as pm
> model = pm.auto_arima(series, seasonal=True, m=12, stepwise=True)
> ```

---

## 9. Seasonal ARIMA — SARIMA

### 9.1 The Model

**SARIMA extends ARIMA** by adding seasonal AR, MA, and differencing operators with period $s$:

$$\boxed{\text{SARIMA}(p, d, q)(P, D, Q)_s}$$

$$\Phi_p(B)\,\tilde{\Phi}_P(B^s)\,\nabla^d\nabla_s^D\,Y_t = c + \Theta_q(B)\,\tilde{\Theta}_Q(B^s)\,\epsilon_t$$

where:
- $\Phi_p(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ — non-seasonal AR
- $\tilde{\Phi}_P(B^s) = 1 - \tilde{\phi}_1 B^s - \cdots - \tilde{\phi}_P B^{Ps}$ — seasonal AR
- $\Theta_q(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ — non-seasonal MA
- $\tilde{\Theta}_Q(B^s) = 1 + \tilde{\theta}_1 B^s + \cdots + \tilde{\theta}_Q B^{Qs}$ — seasonal MA
- $\nabla^d = (1-B)^d$ — regular differences
- $\nabla_s^D = (1-B^s)^D$ — seasonal differences $D$ times

**The seven parameters:**

| Parameter | Domain       | Role                                   |
| --------- | ------------ | -------------------------------------- |
| $p$       | Non-seasonal | AR order                               |
| $d$       | Non-seasonal | Differencing order                     |
| $q$       | Non-seasonal | MA order                               |
| $P$       | Seasonal     | Seasonal AR order                      |
| $D$       | Seasonal     | Seasonal differencing order            |
| $Q$       | Seasonal     | Seasonal MA order                      |
| $s$       | Seasonal     | Seasonal period (e.g., 12 for monthly) |

---

### 9.2 Seasonal Differencing

For monthly data with period $s = 12$:

$$\nabla_{12}Y_t = Y_t - Y_{t-12}$$

This removes the seasonal pattern by subtracting the value from the same month last year. For AirPassengers, the combination $\nabla_{12}\nabla\log Y_t$ (one regular + one seasonal difference of the log series) produces a stationary series.

---

### 9.3 Identifying Seasonal Orders

After differencing, examine the ACF/PACF at **seasonal lags** ($s, 2s, 3s, \ldots$):

| Pattern at seasonal lags                                              | Indicates                           |
| --------------------------------------------------------------------- | ----------------------------------- |
| ACF decays at $s, 2s, 3s, \ldots$; PACF has one seasonal spike at $s$ | Seasonal AR(1): $P = 1$             |
| ACF has one spike at $s$; PACF decays at seasonal lags                | Seasonal MA(1): $Q = 1$             |
| Both decay at seasonal lags                                           | Seasonal ARMA: $P \geq 1, Q \geq 1$ |

> **The AirPassengers benchmark.** SARIMA(1,1,1)(1,1,1,12) on $\log(\text{AirPassengers})$ is the standard textbook specification. The notebook fits this model in Cell 17.

**Implementation:**

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(log_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False)
result = model.fit(disp=False)
print(result.summary())
```

> **Notebook reference.** Cell 17 fits the SARIMA model and runs four-panel residual diagnostics. The residuals should pass the Ljung–Box test and appear approximately Gaussian in the Q-Q plot.

---

## 10. Forecast Evaluation

### 10.1 Walk-Forward Cross-Validation

**Standard $k$-fold CV cannot be used for time series** because it shuffles the data and breaks the temporal order. Instead, use **walk-forward (rolling-origin) cross-validation:**

**Algorithm:**

1. Fix a test window of $h$ steps (e.g., $h = 12$ months).
2. Set the initial training set $\mathcal{T}_0 = \{Y_1, \ldots, Y_{T-h}\}$.
3. **For** $i = 0, 1, \ldots, h-1$:
   a. Fit the model on $\mathcal{T}_i$.
   b. Forecast the next step: $\hat{Y}_{T-h+i+1}$.
   c. Expand the training set: $\mathcal{T}_{i+1} = \mathcal{T}_i \cup \{Y_{T-h+i+1}\}$.
4. Collect all one-step-ahead forecasts and compare with actuals.

```
|--- training ---|-- test --|
|================|o         |  forecast step 1
|=================|o        |  forecast step 2
|==================|o       |  forecast step 3
...
|=========================|o|  forecast step h
```

**Variant: expanding window** (above) vs. **sliding window** (fixed training size — drop oldest observation as you add the newest).

> **Why refit at each step?** The model parameters may change as more data becomes available. Refitting ensures the forecast uses all available information.

> **Notebook reference.** Cell 19 implements `walk_forward_cv` with SARIMA, forecasting the last 12 months of AirPassengers. It reports RMSE and MAPE and plots forecasts vs. actuals.

---

### 10.2 Forecast Error Metrics

| Metric    | Formula                                       | Properties                                     |
| --------- | --------------------------------------------- | ---------------------------------------------- |
| **MAE**   | $\frac{1}{h}\sum\|y_t - \hat{y}_t\|$          | Robust to outliers; same units as $y$          |
| **RMSE**  | $\sqrt{\frac{1}{h}\sum(y_t - \hat{y}_t)^2}$   | Penalises large errors more; same units as $y$ |
| **MAPE**  | $\frac{100}{h}\sum\frac{\|y_t - \hat{y}_t\|}{ | y_t                                            | }$ | Scale-free (%); undefined if $y_t = 0$ |
| **sMAPE** | $\frac{200}{h}\sum\frac{\|y_t - \hat{y}_t\|}{ | y_t                                            | +  | \hat{y}_t                              | }$ | Symmetric; bounded; still problematic near zero |

**Which to use?**
- **Within a single series:** RMSE or MAE (interpretable units).
- **Comparing across series of different scales:** MAPE or sMAPE.
- **For statistical tests:** use the raw errors and apply the Diebold–Mariano test to compare two forecasting methods.

---

### 10.3 Residual Diagnostics

After fitting, verify that the residuals $\hat{\epsilon}_t = Y_t - \hat{Y}_t$ are white noise:

**1. Ljung–Box test:**

$$Q(K) = T(T+2)\sum_{k=1}^{K}\frac{\hat{\rho}_k^2}{T-k}$$

Under $H_0$ (no autocorrelation), $Q(K) \sim \chi^2_{K-p-q}$. Reject $H_0$ if p-value < 0.05 (residuals are correlated → model misspecified).

**2. Residual ACF:** plot ACF of residuals. All lags should be within the significance band.

**3. Q-Q plot:** residuals should lie on the diagonal if normally distributed.

**4. Residuals vs. fitted / vs. time:** look for patterns (heteroscedasticity, trends in residuals).

> **Notebook reference.** The `plot_diagnostics()` call in Cell 15 and Cell 17 produces all four diagnostic plots.

---

### 10.4 Forecast Uncertainty

ARIMA models produce prediction intervals. For a one-step-ahead forecast:

$$\hat{Y}_{T+1} \pm z_{1-\alpha/2}\cdot\hat{\sigma}$$

For multi-step forecasts, uncertainty **accumulates**:

$$\text{Var}(\hat{Y}_{T+h}) = \sigma^2\sum_{j=0}^{h-1}\psi_j^2$$

where $\psi_j$ are the coefficients in the infinite MA representation $Y_t = \sum_{j=0}^{\infty}\psi_j\epsilon_{t-j}$ (Wold's theorem).

**Key insight:** forecast variance grows with horizon $h$. For a random walk ($\psi_j = 1$ for all $j$):

$$\text{Var}(\hat{Y}_{T+h}) = h\sigma^2$$

The prediction interval widens as $\sqrt{h}$. This is why long-horizon forecasts are fundamentally more uncertain.

> **Connection to [Week 08](../week08_uncertainty/theory.md).** This is the time-series version of uncertainty propagation. Multi-step forecasts compound epistemic uncertainty at each step, analogous to how Bayesian predictive variance is wider where data is sparse.

> **Notebook reference.** Exercise 5 asks you to forecast $h = 1, 3, 6, 12$ steps ahead and plot RMSE vs. horizon — you should see RMSE increase with $h$.

---

## 11. Connections to the Rest of the Course

| Week                           | Connection                                                                                                            |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **[Week 03](../../02_fundamentals/week03_linear_models/theory.md) (Linear Models)**    | AR(p) is a linear regression where the features are lagged values of $Y$                                              |
| **[Week 06](../../02_fundamentals/week06_regularization/theory.md) (Regularisation)**   | Walk-forward CV was introduced; Ridge applies to AR models to prevent overfitting with many lags                      |
| **[Week 07](../week07_likelihood/theory.md) (Likelihood)**       | ARIMA coefficients are estimated by MLE; AIC/BIC are likelihood-based model selection criteria                        |
| **[Week 08](../week08_uncertainty/theory.md) (Uncertainty)**      | Prediction intervals for multi-step forecasts; residual calibration; Bayesian time-series models                      |
| **[Week 10](../week10_surrogate_models/theory.md) (Surrogate Models)** | Gaussian Processes use autocovariance kernels (e.g., Matérn, periodic) for temporal modelling                         |
| **[Week 11](../../04_neural_networks/week11_nn_from_scratch/theory.md) (Neural Networks)**  | Recurrent architectures (RNNs, LSTMs) learn non-linear autoregressive dynamics                                        |
| **[Week 17](../../06_sequence_models/week17_attention/theory.md) (Attention)**        | Attention replaces fixed-lag structure; the model learns which past timesteps matter                                  |
| **[Week 18](../../06_sequence_models/week18_transformers/theory.md) (Transformers)**     | Positional encodings play the role of time indices; Transformer-based forecasting (e.g., Temporal Fusion Transformer) |

---

## 12. Notebook Reference Guide

| Cell              | Section                   | What it demonstrates                                             | Theory reference |
| ----------------- | ------------------------- | ---------------------------------------------------------------- | ---------------- |
| 5 (Load Data)     | AirPassengers             | Monthly passengers 1949–1960; trend + multiplicative seasonality | Section 2        |
| 7 (Decomposition) | Additive & multiplicative | Side-by-side decomposition; log transform equivalence            | Section 3        |
| 9 (ACF/PACF)      | Autocorrelation           | Raw series seasonal spikes; differenced series clean structure   | Section 4        |
| 11 (ADF Test)     | Stationarity              | ADF on raw, log, diff(1), diff(1)+diff(12)                       | Section 5        |
| 13 (Synthetic)    | AR/MA processes           | ACF/PACF signatures for AR(2), MA(1), ARMA(1,1)                  | Section 4.3      |
| 15 (ARIMA)        | ARIMA(1,1,1)              | Fit + residual diagnostics on log(AirPassengers)                 | Section 8        |
| 17 (SARIMA)       | SARIMA(1,1,1)(1,1,1,12)   | Seasonal model fit + diagnostics                                 | Section 9        |
| 19 (Walk-Forward) | Forecast evaluation       | 12-month walk-forward CV; RMSE, MAPE, forecast plot              | Section 10       |
| Ex. 1             | Decomposition             | Compare residual variance: additive vs. multiplicative           | Section 3.3      |
| Ex. 2             | ACF/PACF reading          | Identify orders from synthetic process plots                     | Section 4.3      |
| Ex. 3             | Sunspots                  | ADF test on a different dataset                                  | Section 5.2      |
| Ex. 4             | ARIMA order selection     | Propose orders from ACF/PACF; compare AIC/BIC                    | Section 8.4      |
| Ex. 5             | Multi-horizon CV          | RMSE vs. forecast horizon $h$                                    | Section 10.4     |

**Suggested modifications:**

| Modification                                                       | What it reveals                                                               |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| Fit ARIMA(2,1,0) and ARIMA(0,1,2) — compare AIC                    | Pure AR vs. pure MA on the same data; information criteria decide             |
| Double-difference ($d=2$) and check ACF lag 1                      | Over-differencing creates artificial negative autocorrelation                 |
| Try `seasonal=False` in `auto_arima` on AirPassengers              | Ignoring seasonality produces poor residuals and inflated RMSE                |
| Forecast 24 months ahead and plot prediction intervals             | Intervals fan out — uncertainty grows with horizon                            |
| Replace AirPassengers with a non-seasonal dataset (e.g., sunspots) | ARIMA without seasonal component; cyclical patterns require different $p$     |
| Use STL instead of classical decomposition                         | Observe how STL handles outliers and time-varying seasonality more gracefully |

---

## 13. Symbol Reference

| Symbol       | Name                           | Meaning                                                |
| ------------ | ------------------------------ | ------------------------------------------------------ |
| $Y_t$        | Observation                    | Value of the time series at time $t$                   |
| $T_t$        | Trend                          | Long-term level component                              |
| $S_t$        | Seasonal                       | Periodic component with period $s$                     |
| $R_t$        | Residual                       | Irregular (noise) component                            |
| $\gamma_k$   | Autocovariance                 | $\text{Cov}(Y_t, Y_{t-k})$                             |
| $\rho_k$     | Autocorrelation (ACF)          | $\gamma_k / \gamma_0$                                  |
| $\phi_{kk}$  | Partial autocorrelation (PACF) | Correlation at lag $k$ removing intermediate effects   |
| $B$          | Backshift operator             | $BY_t = Y_{t-1}$; $B^k Y_t = Y_{t-k}$                  |
| $\nabla$     | Difference operator            | $\nabla Y_t = (1 - B)Y_t = Y_t - Y_{t-1}$              |
| $\nabla_s$   | Seasonal difference            | $(1 - B^s)Y_t = Y_t - Y_{t-s}$                         |
| $\phi_i$     | AR coefficients                | Weights on lagged values in AR model                   |
| $\theta_j$   | MA coefficients                | Weights on lagged noise terms in MA model              |
| $p$          | AR order                       | Number of AR lags                                      |
| $d$          | Integration order              | Number of regular differences                          |
| $q$          | MA order                       | Number of MA lags                                      |
| $P, D, Q$    | Seasonal orders                | Seasonal AR, differencing, MA orders                   |
| $s$          | Seasonal period                | 12 for monthly, 7 for daily, etc.                      |
| $\Phi(B)$    | AR polynomial                  | $1 - \phi_1 B - \cdots - \phi_p B^p$                   |
| $\Theta(B)$  | MA polynomial                  | $1 + \theta_1 B + \cdots + \theta_q B^q$               |
| $\psi_j$     | MA($\infty$) coefficients      | Wold decomposition: $Y_t = \sum \psi_j \epsilon_{t-j}$ |
| $\sigma^2$   | Innovation variance            | $\text{Var}(\epsilon_t)$                               |
| $\text{AIC}$ | Akaike info criterion          | $-2\log\hat{L} + 2k$                                   |
| $\text{BIC}$ | Bayesian info criterion        | $-2\log\hat{L} + k\log n$                              |
| $Q(K)$       | Ljung–Box statistic            | Tests white noise of residuals up to lag $K$           |

---

## 14. References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley. — The foundational reference for ARIMA/SARIMA and the Box–Jenkins methodology.
2. Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts. Available free online at [otexts.com/fpp3](https://otexts.com/fpp3). — Excellent modern treatment; covers decomposition, ARIMA, ETS, and evaluation.
3. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. — Graduate-level reference; rigorous treatment of stationarity, unit roots, and spectral methods.
4. Shumway, R. H. & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*, 4th ed. Springer. — Applied focus with R examples; good for ACF/PACF intuition.
5. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on LOESS." *Journal of Official Statistics*, 6(1), 3–73. — The original STL paper.
6. Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *JASA*, 74(366), 427–431. — The ADF test.
7. Ljung, G. M. & Box, G. E. P. (1978). "On a Measure of Lack of Fit in Time Series Models." *Biometrika*, 65(2), 297–303. — The Ljung–Box test for residual autocorrelation.
8. Wold, H. (1938). *A Study in the Analysis of Stationary Time Series*. Almqvist & Wiksell. — Wold's decomposition theorem: any stationary process = deterministic + MA($\infty$).
9. Hyndman, R. J. & Khandakar, Y. (2008). "Automatic Time Series Forecasting: The forecast Package for R." *Journal of Statistical Software*, 27(3). — The algorithm behind `auto.arima` / `pmdarima.auto_arima`.
10. Seabold, S. & Perktold, J. (2010). "statsmodels: Econometric and Statistical Modeling with Python." *Proc. SciPy Conf.* — Reference for the `statsmodels.tsa` module used in the notebook.
