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
`sys.path`.

**One-time setup (recommended):** register the folder with the repo's
`.venv` via a `.pth` file. Python reads every `*.pth` file in
`site-packages` at startup and adds each path in it to `sys.path`, so after
this, any notebook or script using `.venv` — in any folder — can import the
Library with no `sys.path` line at all. Run from the repo root with the venv
active:

```bash
python -c "import site, pathlib; (pathlib.Path(site.getsitepackages()[0]) / 'risk_library.pth').write_text(str(pathlib.Path('Library').resolve()))"
```

Restart any running notebook kernel afterwards (`.pth` files are only read
when Python starts). The file lives inside `.venv/` (gitignored), so re-run
this if you ever rebuild the venv or move the repo.

**Fallback** (a different Python environment without the `.pth` file): add
the folder at the top of the script/notebook instead:

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

### Editor autocomplete (VS Code)

VS Code's autocomplete/hover docs (Pylance) read code without running it, so
they never see the `.pth` file or a runtime `sys.path` line. The repo's
`.vscode/settings.json` points Pylance at this folder instead:

```json
{ "python.analysis.extraPaths": ["./Library"] }
```

With that in place, `from Risk_Library import ` shows a dropdown of every
function (the names listed in `Risk_Library.__all__`), and hovering a
function shows its docstring. This is an editor-only setting — actually
importing the Library when code runs is handled by the `.pth` file (or the
fallback `sys.path` line). (Reload the VS Code window after changing it.)

---

## Module reference

### `utils.py` — shared helpers

Not usually called directly from your own code — these back the option-string
dispatch and shared statistics used across the other files:

- `handle_missing_data(data, missing_strategy="drop_row")` — the
  `"drop_row"`/`"pairwise"` dispatcher used by `covariance.py` and
  `moments.py`.
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
eigenvalues before fix: [-0.6488  1.4600  2.1888]
eigenvalues after fix:  [ 0.0000  1.2284  1.7716]
```

`method="near_psd"` (Rebonato & Jackel eigenvalue clipping, fast) or
`"higham"` (alternating projections, slower but the true nearest PSD matrix).
Auto-detects covariance vs. correlation input from the diagonal.

Also in this file: `classify_psd` diagnoses a matrix *without* changing it,
reporting PD / PSD / non-PSD plus the eigenvalue evidence:

```python
from psd_fix import classify_psd

classify_psd(bad_corr)["summary"]
classify_psd(fixed)["summary"]
```
```
non-PSD: 1 eigenvalue(s) below -threshold (2.19e-08): [-0.6488].
PSD: all eigenvalues are above -threshold (1.77e-08); 1 eigenvalue(s) within threshold of zero; 1 small negative eigenvalue(s) were treated as zero.
```

Returns a dict (`"classification"`, `"summary"`, `"eigenvalues"`,
`"adjusted_eigenvalues"`, `"threshold"`, the indices/values of small negative
eigenvalues treated as zero and of real negatives, min/max eigenvalue, and
symmetry info). The tolerance is scale-aware — `threshold = atol + rtol *
max|eigenvalue|`, defaults `atol=1e-12`, `rtol=1e-8` — so the verdict is a
*numerical* one that depends on the tolerance, not a proof of PSD-ness.
`asymmetry_strategy="report"` (default) refuses to classify a non-symmetric
matrix; `"symmetrize"` classifies `(A + A^T)/2` and says so in the summary.

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

`method="classical"` (`P_t - P_{t-1}`), `"arithmetic"` (`P_t/P_{t-1} - 1`), or
`"log"`/`"geometric"` (`ln(P_t/P_{t-1})`). `date_column` (default `"Date"`) is
carried along untouched; pass `None` if every column is a price series.

Also in this file, the reverse direction — prices from returns — under the
three return models, plus the closed-form mean/std of the next price when
`r ~ N(mu, sigma^2)`:

```python
import numpy as np
import pandas as pd
from returns import price_from_returns, expected_price_moments

r = np.random.default_rng(1).normal(0.0, 0.2, 100_000)   # simulated returns
rows = {}
for m in ["classical", "arithmetic", "geometric"]:
    p = price_from_returns(r, start_price=100, method=m)
    e = expected_price_moments(100, sigma=0.2, method=m)
    rows[m] = [p.mean(), p.std(), e["Mean"], e["Std"]]
pd.DataFrame(rows, index=["Sim Mean", "Sim Std", "Expected Mean", "Expected Std"]).T
```
```
              Sim Mean    Sim Std  Expected Mean  Expected Std
classical    99.999082   0.199308     100.000000      0.200000
arithmetic   99.908189  19.930763     100.000000     20.000000
geometric   101.914446  20.550435     102.020134     20.609778
```

`method="classical"`
(`P_t = P_{t-1} + r_t`), `"arithmetic"` (`P_t = P_{t-1}(1 + r_t)`), or
`"geometric"` (`P_t = P_{t-1}e^{r_t}`). `price_from_returns` takes
`horizon="one_step"` (every return is a separate scenario for the same next
period) or `"path"` (returns are consecutive periods, compounded in order —
the exact inverse of `return_calculate`).

### `moments.py` — mean, variance, skewness, kurtosis

```python
import pandas as pd
from moments import first_four_moments

data = pd.DataFrame({
    "A": [0.01, -0.02, 0.015, 0.005, -0.01, 0.02],
    "B": [0.03, -0.01, 0.00, -0.04, 0.02, 0.01],
})
first_four_moments(data, method="unbiased", kurtosis_type="excess")
```
```
                 A         B
Mean      0.003333  0.001667
Variance  0.000237  0.000617
Skewness -0.704959 -0.870689
Kurtosis -0.929875  0.735427
```

`method="unbiased"` (sample estimators: `n-1` variance, adjusted skew/kurtosis —
same as pandas/Excel) or `"biased"` (population formulas, same as
`scipy.stats.skew/kurtosis(bias=True)`). `kurtosis_type="excess"` (normal = 0)
or `"raw"` (normal = 3). A Series returns a Series; a DataFrame returns one
column per numeric input column, with `missing_strategy="drop_row"`/`"pairwise"`.

Notes on getting spec number from output:
m = first_four_moments(df["x"])   # Series
m["Skewness"]                     # a single float

m = first_four_moments(df)        # DataFrame
m.loc["Kurtosis", "x"]            # a single float
m["x"]                            # Series of all four moments for column x

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

Every fit function also takes `output="params"` (default, the Series above),
`"distribution"` (a frozen `scipy.stats` distribution built from the fit), or
`"both"` (a `(params, distribution)` tuple):

```python
norm_fit = fit_normal_distribution(sample, output="distribution")   # = stats.norm(loc=mu, scale=sigma)
norm_fit.ppf(0.05)              # 5% quantile
norm_fit.cdf(0.0)               # P(X <= 0)
norm_fit.mean(), norm_fit.std()
```
```
-0.021971
0.414231
(0.003333, 0.015384)
```

The frozen distributions are `stats.norm(loc=mu, scale=sigma)`,
`stats.t(df=nu, loc=mu, scale=sigma)`, the error term
`stats.t(df=nu, loc=0, scale=sigma)` for `fit_t_regression`, and
`stats.norminvgauss(a=alpha*delta, b=beta*delta, loc=mu, scale=delta)` for
NIG (scipy's scaled parameterization). For the t, note `.std()` is
`sigma * sqrt(nu / (nu - 2))`, not `sigma`.

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
`method="analytical"` (closed-form quantile), `"simulation"` (draw from the
fitted distribution and take the empirical quantile), or `"historical"`
(historical simulation: no distribution is fitted — VaR is minus the
empirical alpha-quantile of the observed returns, ES minus the average of the
returns at or below it; `distribution` is ignored). `calculate_es` has the
same signature, returning `"ES Absolute"`/`"ES Diff from Mean"`.

```python
calculate_var(returns, method="historical", alpha=0.05)
```

On the 1,000 returns of a fat-tailed stock (`Midterm/Practice/practice1/problem2.csv`):

| 5% alpha | VaR Absolute | ES Absolute |
|---|---|---|
| Normal, analytical | 0.036586 | 0.045792 |
| t, analytical | 0.036102 | 0.048230 |
| Historical | 0.034963 | 0.046824 |

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

**Simulating joint returns from a copula:**

```python
from copula import simulate_copula

# `returns` = a DataFrame of historical returns, one column per asset
sims, fit = simulate_copula(returns, copula="t", marginals="t", seed=1, output="both")
sims.head(3)
fit["nu"]
```
```
         A1        A2        A3        A4        A5
0  0.003080 -0.017871  0.007903 -0.012124  0.008206
1  0.005695  0.018715  0.009990  0.005764  0.009014
2  0.004238 -0.006460 -0.006232  0.034638  0.011710
5.9464
```

Each column keeps its own fitted marginal distribution and the copula links
them: `copula="gaussian"` (no tail dependence) or `"t"` (joint extremes more
likely; its `nu` is fitted, or pass `nu=` yourself). `marginals="normal"`/`"t"`
for every column, or a dict like `{"A": "normal", "B": "t"}` to choose per
column. `correlation="normal_scores"` (Gaussian default, same as
`portfolio_risk`), `"spearman"`, or `"kendall"` (t default, same as
`fit_t_copula`). `output="simulations"` (default) or `"both"` to also get the
fit: `{"marginals": frozen scipy distributions, "correlation", "nu"}`.
Turn the simulated returns into portfolio P&L/VaR yourself, or use the
ready-made `portfolio_risk.calculate_portfolio_var_es` /
`simulate_t_copula_portfolio_var_es` below.

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
- **`price_from_returns` / `expected_price_moments`** and the `"classical"`
  method of `return_calculate` (in `test_returns.py`) are checked against
  their closed-form definitions, a prices → returns → prices round trip, and
  a fixed-seed simulation of the next price's mean/std.
- **`method="historical"`** of `calculate_var`/`calculate_es` (in
  `test_risk_metrics.py`) has no reference CSV, so it's checked against a
  small hand-built sample, against numpy's quantile/tail-mean directly, and
  for converging to the analytical Normal answer on a large normal sample.
- **`simulate_copula`** (in `test_copula.py`) has no reference CSV, so it's
  checked against what it should reproduce: with the same seed, its Gaussian
  copula gives exactly Test 9.1's `calculate_portfolio_var_es` VaR, and its
  t copula's fitted `nu`/correlation match `fit_t_copula`. It also checks the
  simulated means, standard deviations and correlation, and that a low-`nu`
  t copula gives more joint extremes than a Gaussian one.
- **`output="distribution"`/`"both"`** of the fit functions (in
  `test_distributions.py`) is checked by confirming each frozen scipy
  distribution's mean/median/std match its fitted parameters through that
  distribution's own formulas (and, for the NIG method-of-moments fit, that
  it reproduces the sample mean and variance).
- **`classify_psd`** (in `test_psd_fix.py`) has no reference CSV either, so
  it's checked against small hand-built matrices with known eigenvalues (PD,
  singular PSD, PSD with injected rounding error, non-PSD, non-symmetric).
- **`test_moments.py`** has no reference CSV in `Tests.xlsx`, so it checks
  `first_four_moments` against pandas (`.var()`/`.skew()`/`.kurt()`) for
  `method="unbiased"` and `scipy.stats` (`bias=True`) for `method="biased"`.
