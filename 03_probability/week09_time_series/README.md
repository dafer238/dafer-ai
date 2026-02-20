# Week 07 — Time-Series Fundamentals: Seasonality & Autocorrelation

## Prerequisites

- **Week 00b** — basic statistics (mean, variance, correlation).
- **Week 03 (linear models)** — linear regression is the base for ARIMA's AR component.
- **Week 04 (regularization)** — time-series cross-validation (walk-forward) was introduced there.

## What this week delivers

Most real-world data is ordered in time: sensor readings, sales, financial returns, climate records. Understanding time-series structure — trend, seasonality, autocorrelation, and stationarity — is a prerequisite for any sequential model, including the RNNs and Transformers introduced later. This week gives you the classical statistical toolkit (decomposition, ACF/PACF, ARIMA/SARIMA) before deep-learning approaches are introduced in Weeks 13–14.

## Overview

Characterise time-series structure through visual and statistical tools, test for and achieve stationarity, and build forecasting models from ARIMA to SARIMA. Evaluate forecasts rigorously with walk-forward cross-validation.

## Study

- Time-series anatomy: trend, seasonality, cyclical variation, irregular noise
- Additive vs multiplicative decomposition; STL (Seasonal-Trend decomposition using LOESS)
- Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
- Stationarity: ADF (Augmented Dickey-Fuller) test; differencing (`diff`)
- ARIMA(p,d,q) model identification via ACF/PACF rules
- SARIMA(p,d,q)(P,D,Q,s) — seasonal extension
- Walk-forward (rolling-origin) forecast evaluation and MAE/RMSE/MAPE

## Practical libraries & tools

- `statsmodels.tsa.seasonal.seasonal_decompose`, `STL`
- `statsmodels.graphics.tsaplots.plot_acf`, `plot_pacf`
- `statsmodels.tsa.stattools.adfuller`
- `statsmodels.tsa.arima.model.ARIMA`
- `statsmodels.tsa.statespace.sarimax.SARIMAX`
- NumPy, Pandas, Matplotlib

## Datasets & examples

- **AirPassengers** — classic monthly international airline passengers 1949–1960 (strong trend + seasonality)
- **Sunspots** (`statsmodels.datasets.sunspots`) — cyclical pattern, no clear trend
- Synthetic AR(2) / MA(1) processes for ACF/PACF intuition

## Exercises

1. **Decomposition** — decompose AirPassengers into trend/seasonal/residual using both additive and multiplicative `seasonal_decompose`. Compare residuals; identify which model fits better.

2. **ACF/PACF reading** — generate synthetic AR(2), MA(1), and ARMA(1,1) processes with known parameters; plot ACF and PACF; identify the model orders from the plots.

3. **Stationarity** — apply ADF test to raw AirPassengers and after first-order differencing and seasonal differencing. Confirm p-value drops below 0.05.

4. **ARIMA fitting** — use ACF/PACF to propose (p,d,q); fit with `statsmodels.tsa.arima.model.ARIMA`; inspect residual diagnostics.

5. **SARIMA forecasting with walk-forward CV** — fit `SARIMAX(p,d,q)(P,D,Q,12)` on AirPassengers; run a 12-step walk-forward CV; report RMSE and plot forecasts vs actuals.

## Code hints

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=''):
    result = adfuller(series.dropna())
    print(f'{name}  ADF stat={result[0]:.4f}  p={result[1]:.4f}  '
          f'{"STATIONARY" if result[1] < 0.05 else "NON-STATIONARY"}')

def fit_arima(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    return model.fit()
```

## Deliverables

- [ ] Decomposition plots (trend / seasonal / residual) for AirPassengers.
- [ ] ACF/PACF plots for synthetic AR, MA, ARMA processes with annotations.
- [ ] ADF test before/after differencing.
- [ ] ARIMA residual diagnostics (Ljung-Box, Q-Q plot).
- [ ] SARIMA walk-forward CV results: RMSE, 12-month ahead forecast plot.

## What comes next

- **Week 13 (attention)** — attention-based sequence modelling is the deep-learning counterpart to ARIMA.
- **Week 14 (transformers)** — positional encodings play a role analogous to trend/seasonality removal.
- **Week 07 (surrogate models)** — Gaussian Processes also model temporal correlation and can serve as a basis for GP-based time-series forecasting.
