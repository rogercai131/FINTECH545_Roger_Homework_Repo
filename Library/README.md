# Library — Quant Risk Calculation Package

A personal Python package of quantitative risk-management functions, organized
so each file covers one type of calculation, and each function can adapt to
different scenarios (missing data, distribution choice, fitting method, etc.)
through keyword options rather than needing a separate function per variant.

## Design

- **One file per calculation type.** `covariance.py` handles all
  covariance/correlation calculations, `option_pricing.py` handles all option
  pricing, and so on.
- **Functions accept and return `pandas` objects** (`DataFrame`/`Series`) so
  columns/index labels (asset names, dates) are never lost. Internally, the
  heavy math is done with `numpy` for speed; pandas is only the interface.
- **Scenario flexibility via option-string keyword arguments**, not booleans —
  e.g. `missing_strategy="drop_row"` or `"pairwise"`, `method="near_psd"` or
  `"higham"`. This makes it easy to add a new option later (e.g. a new
  missing-data strategy) without changing any function's signature.
- **Shared logic lives in `utils.py`** (missing-data handling, AICc) so it
  isn't duplicated across calculation files.
- **`Risk_Library.py` is the aggregator.** It just imports the public
  functions from every other file, so you can use either import style shown
  below.

## Getting started

Every module here is a plain Python file, not a namespaced sub-package, so
the only setup needed is having the `Library/` folder itself on your
`sys.path` — e.g. run your script from inside `Library/`, or add it explicitly:

```python
import sys
sys.path.append("/path/to/Library")
```

Then import functions either **standalone**, straight from the file that
defines them:

```python
from covariance import covariance_matrix
```

or **aggregated**, from `Risk_Library`, which re-exports everything:

```python
from Risk_Library import covariance_matrix
```

Both give you the exact same function — use whichever reads better for the
script you're writing. The examples below use the standalone style, one
module at a time.

---

## Module reference

### `utils.py` — shared helpers

Not usually called directly from your own code — these back the option-string
dispatch and shared statistics used across the other files:

- `handle_missing_data(data, missing_strategy="drop_row")` — the
  `"drop_row"`/`"pairwise"` dispatcher used by `covariance.py`.
- `aicc(log_likelihood, num_params, num_obs)` — corrected Akaike Information
  Criterion, used by `distributions.py` and `copula.py`.

### `covariance.py` — covariance & correlation matrices

```python
import pandas as pd
import numpy as np
from covariance import covariance_matrix

data = pd.DataFrame({
    "A": [0.01, 0.02, np.nan, -0.01, 0.03],
    "B": [0.015, 0.018, 0.02, -0.005, 0.025],
})

covariance_matrix(data, method="covariance", missing_strategy="drop_row")
```
```
          A         B
A  0.000292  0.000216
B  0.000216  0.000166
```

Also in this file:
- `ew_covariance_matrix(data, lambda_=0.97, method="covariance", lambda_corr=None)`
  — exponentially-weighted covariance/correlation, optionally mixing a
  different decay factor for variance vs. correlation.
- `kendall_correlation_matrix(data, psd_fix_method="near_psd")` — a
  rank-based (Kendall's tau) correlation matrix, repaired to be PSD.

`method` picks covariance vs. correlation; `missing_strategy` picks
`"drop_row"` (listwise deletion) vs. `"pairwise"` (each column pair uses its
own complete rows, via pandas' native pairwise handling).

### `psd_fix.py` — repairing a non-PSD matrix

A covariance/correlation matrix built from pairwise-complete data (or from
independently-estimated pairwise correlations, like Kendall's tau) isn't
always positive semi-definite. `fix_psd_matrix` repairs it:

```python
import pandas as pd
import numpy as np
from psd_fix import fix_psd_matrix

bad_corr = pd.DataFrame(
    [[1.00, -0.99, 0.99],
     [-0.99, 1.00, 0.46],
     [0.99, 0.46, 1.00]],
    index=["A", "B", "C"], columns=["A", "B", "C"],
)
np.linalg.eigvalsh(bad_corr.values)   # -> has a negative eigenvalue, not valid

fixed = fix_psd_matrix(bad_corr, method="near_psd")
np.linalg.eigvalsh(fixed.values)      # -> all eigenvalues >= 0 now
```
```
eigenvalues before fix: [-0.9206  1.9000  2.0206]
eigenvalues after fix:  [ 0.0000  1.4609  1.5391]
```

`method="near_psd"` (Rebonato & Jackel eigenvalue clipping, fast) or
`"higham"` (alternating projections, slower but the true nearest PSD matrix).
Auto-detects covariance vs. correlation input from the diagonal.

### `cholesky.py` — factorizing a PSD matrix

```python
from cholesky import cholesky_psd

cholesky_psd(fixed)   # `fixed` from the psd_fix example above
```
```
          A         B    C
A  1.000000  0.000000  0.0
B -0.519188  0.854660  0.0
C  0.519188  0.854660  0.0
```

Like a standard Cholesky factorization (`L` such that `L @ L.T == matrix`),
but tolerant of positive *semi*-definite input (zero-variance directions come
out as all-zero rows) instead of raising an error.

### `simulation.py` — Monte Carlo simulation from a covariance matrix

```python
import pandas as pd
from simulation import simulate_normal

cov = pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0009]], index=["A", "B"], columns=["A", "B"])
draws = simulate_normal(cov, num_simulations=50_000, seed=1)
draws.cov()   # should be close to `cov`, since we just drew from it
```
```
          A         B
A  0.000396  0.000106
B  0.000106  0.000899
```

`method="direct"` (default) factors the full covariance matrix; `method="pca"`
keeps only the top principal components explaining `explained_variance` of
total variance. `psd_fix="near_psd"`/`"higham"` repairs the input first if
it isn't already PSD.

### `returns.py` — prices to returns

```python
import pandas as pd
from returns import return_calculate

prices = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=4),
    "AAPL": [100, 102, 101, 105],
})
return_calculate(prices, method="arithmetic")
```
```
        Date      AAPL
0 2024-01-02  0.020000
1 2024-01-03 -0.009804
2 2024-01-04  0.039604
```

`method="arithmetic"` (`P_t/P_{t-1} - 1`) or `"log"` (`ln(P_t/P_{t-1})`).
`date_column` (default `"Date"`) is carried along untouched; pass `None` if
every column is a price series.

### `distributions.py` — fitting a distribution to data

```python
import pandas as pd
from distributions import fit_normal_distribution, fit_t_distribution

sample = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
fit_normal_distribution(sample)
```
```
mu       0.003333
sigma    0.015384
dtype: float64
```

```python
fit_t_distribution(sample)   # heavier-tailed sample -> a real dataset with fat tails
```
```
mu       -0.000103
sigma     0.020957
nu       10.322929
dtype: float64
```

Also in this file: `fit_t_regression(y, x)` (linear regression with
t-distributed errors), `t_distribution_aicc(data, fitted=None)`, and
`fit_nig_distribution(data, method="moments"/"mle")` (Normal Inverse Gaussian).

### `risk_metrics.py` — VaR and Expected Shortfall

```python
from risk_metrics import calculate_var, calculate_es

calculate_var(sample, distribution="normal", alpha=0.05)   # `sample` from above
```
```
VaR Absolute          0.021971
VaR Diff from Mean    0.025304
dtype: float64
```

`distribution="normal"`/`"t"` picks the fitted distribution;
`method="analytical"` (closed-form quantile) or `"simulation"` (draw from the
fitted distribution and take the empirical quantile). `calculate_es` has the
same signature, returning `"ES Absolute"`/`"ES Diff from Mean"`.

### `portfolio_risk.py` — portfolio VaR/ES via copula simulation

```python
import pandas as pd
from portfolio_risk import calculate_portfolio_var_es

portfolio = pd.DataFrame({
    "Stock": ["A", "B"],
    "Holding": [100, 50],
    "Starting Price": [20, 30],
    "Distribution": ["Normal", "T"],   # each asset can have its own marginal
})
# `returns` = a DataFrame of historical returns with columns "A", "B"
calculate_portfolio_var_es(portfolio, returns, alpha=0.05, seed=1)
```
```
   Stock      VaR95        ES95  VaR95_Pct  ES95_Pct
0      A  55.384313   70.641705   0.027692  0.035321
1      B  81.681487  110.809664   0.054454  0.073873
2  Total  99.280509  131.082222   0.028366  0.037452
```

Fits each asset's own marginal (per the `Distribution` column), links them
with a Gaussian copula (correlation estimated from history), simulates joint
scenarios, and reports dollar and percentage VaR/ES per holding and in total.
Column names scale with `alpha` (e.g. `alpha=0.01` gives `VaR99`/`ES99`).

### `portfolio_optimization.py` — constructing portfolio weights

```python
import pandas as pd
from portfolio_optimization import calculate_risk_parity_weights

cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["Stock", "Bond"], columns=["Stock", "Bond"])
calculate_risk_parity_weights(cov)
```
```
Stock    0.6
Bond     0.4
Name: W, dtype: float64
```

(Each holding contributes equally to portfolio risk: at these weights, both
Stock and Bond contribute exactly 0.0168 to portfolio variance.) Pass
`risk_budgets=[...]` for unequal risk budgets. `calculate_max_sharpe_weights(cov,
expected_returns, risk_free_rate, lower_bound, upper_bound)` finds the
weights maximizing the Sharpe ratio, with per-asset bounds.

### `attribution.py` — ex-post return/risk attribution

```python
import pandas as pd
from attribution import return_attribution

returns = pd.DataFrame({"A": [0.02, -0.01, 0.03], "B": [0.01, 0.02, -0.01]})
return_attribution(returns, weights=[0.6, 0.4])
```
```
                           A         B  Portfolio
Value
TotalReturn         0.040094  0.019898   0.032016
Return Attribution  0.023997  0.008019   0.032016
Vol Attribution     0.011545 -0.003974   0.007571
```

`TotalReturn` is each holding's own standalone compounded return;
`Return Attribution`/`Vol Attribution` are each holding's share of the
*portfolio's* total return/volatility (they always sum exactly to the
`Portfolio` column). Pass `factor_returns=`/`betas=` to attribute to
systematic factors plus a residual `"Alpha"` instead of to holdings directly.

### `option_pricing.py` — pricing a single option

```python
from option_pricing import price_european_option

price_european_option("Call", S=100, K=100, T=0.5, r=0.03, sigma=0.2)
```
```
Value     6.371028
Delta     0.570158
Gamma     0.027772
Vega     27.772132
Rho      25.322391
Theta    -7.073770
dtype: float64
```

Closed-form Generalized Black-Scholes-Merton pricing and Greeks (`q` for a
continuous dividend yield). `price_american_option(...)` prices via a
binomial tree instead, supporting early exercise, and either a continuous
yield `q` or discrete dollar dividends via `dividend_days`/`dividend_amounts`.

### `multivariate_t.py` — fitting a multivariate t distribution

```python
from multivariate_t import fit_multivariate_t

# `returns` = a DataFrame of historical returns, one column per asset
fitted = fit_multivariate_t(returns)
fitted["mu"]              # fitted mean vector (Series)
fitted["scale"]            # fitted scale matrix S (DataFrame)
fitted["nu"]               # fitted degrees of freedom (float)
fitted["log_likelihood"]   # log-likelihood at the fit (float)
```

Note the implied covariance of the fit is `nu / (nu - 2) * scale`, not
`scale` itself.

### `copula.py` — copula dependence modeling

```python
from copula import compare_copulas, simulate_t_copula_portfolio_var_es

compare_copulas(returns)   # AICc/BIC comparison of a Gaussian vs. a T copula
```

Also in this file: `gaussian_copula_log_likelihood(data)`,
`fit_t_copula(data)`, `t_copula_tail_dependence(correlation, nu)` (pairwise
lower tail dependence under a fitted t copula), and
`simulate_t_copula_portfolio_var_es(portfolio, returns, alpha, num_simulations, seed)`
— the t-copula counterpart to `portfolio_risk.calculate_portfolio_var_es`,
for a `portfolio` given as `Stock`/`currentValue` columns.

---

## Tests

The `tests/` folder is a `pytest` suite validating (almost) every function
above against the reference input/output CSV pairs listed in `Tests.xlsx`
and stored in `test_files/`. Run it from inside `Library/`:

```bash
pytest tests/
```

A few things worth knowing before reading a test file or a failure:

- **Deterministic calculations are checked for an exact (or near-machine-precision)
  match** against the reference CSV — this covers most of the suite:
  covariance/correlation, PSD repair, Cholesky, returns, distribution fits,
  VaR/ES formulas, portfolio optimization, attribution, and option pricing.
- **Monte Carlo simulations can't be matched exactly.** Tests that simulate
  random draws (`test_simulation.py`, the `method="simulation"` cases in
  `test_risk_metrics.py`, `test_portfolio_risk.py`, and
  `test_copula.py::test_simulate_t_copula_portfolio_var_es`) instead simulate
  their own sample with a fixed seed and check it lands close to the
  reference value, within a tolerance sized to the sampling noise expected at
  that sample size — a fresh random draw from the same distribution will
  never exactly reproduce another one's covariance/quantile/tail-mean.
- **A few numerically-optimized fits use a looser tolerance for the same
  reason a calculator and a spreadsheet can disagree in the 10th decimal
  place**: `test_multivariate_t.py` and parts of `test_copula.py` fit
  parameters (like a distribution's degrees of freedom) via a 1-D numerical
  optimizer, so the last few digits can depend on the optimizer's
  convergence tolerance even when the method is identical.
- **`test_option_pricing.py`** flags one specific case explicitly: the
  binomial-tree Greeks for an at-the-money American put sitting very close to
  its early-exercise boundary are more sensitive to the finite-difference
  bump size than the other Greeks/options, so that one Delta uses a looser
  tolerance than the rest of the test.
