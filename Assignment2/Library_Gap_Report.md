# Library Gap Report: what Assignment 2 needed that `Library/` did not provide

This report lists every calculation in `Assignment2/Code.ipynb` that had to be written inline, plus the places where an existing
Library function was used but did not fully fit. Each item says where it came up, what the notebook did, and a suggested home in
`Library/` that follows the package conventions:
- one file per topic
- DataFrame in and DataFrame out, with numpy inside
- option strings rather than booleans
- shared helpers in `utils.py`
- every function re-exported in `Risk_Library.py`

Priority means how often this kind of calculation comes up in the course:
- **High:** needed in several problems, or a basic building block.
- **Medium:** needed once, but standard.
- **Low:** specific to this assignment.

---

## 1. Missing calculations

### 1.1 Risk measures (`risk_metrics.py`)

| Calculation | Used in | Notebook helper | Priority | Suggestion |
|---|---|---|---|---|
| Historical VaR (empirical α-quantile) | P2(d), P3(c,e), P4(e), P5(c) | `historical_var` | **High** | Add `distribution="historical"` to `calculate_var`. It returns `-np.quantile(x, alpha)` and needs no fitting step. |
| Historical ES (mean of outcomes at or below the quantile) | P3(c,e), P4(e) | `historical_es` | **High** | Same option on `calculate_es`. Use the same quantile as the VaR so that ES ≥ VaR. |
| Normal VaR for a **given σ** (no fit from data) | P2(d), P5(d) | `normal_var` | **High** | `var_from_params(sigma, mu=0, distribution="normal", nu=None, alpha)`. Today `calculate_var` always fits σ itself, so it cannot use an EW σ or a portfolio σ from w′Σw. |
| VaR/ES of a raw P&L vector (dollars, many columns) | P3, P4, P5 | `historical_var_es_table` | Medium | A `var_es_table(pnl_df, alphas=[0.05, 0.01])` helper. Most problems report several positions at two α levels. |

### 1.2 Covariance and matrix tools (`covariance.py` / `psd_fix.py` / `utils.py`)

| Calculation | Used in | Notebook helper | Priority | Suggestion |
|---|---|---|---|---|
| Joint-observation count table for missing data | P1(Predict) | `joint_observation_counts` | Medium | `covariance.joint_observation_counts(data)`, i.e. `O.T @ O` on the not-missing indicator. Useful next to `missing_strategy="pairwise"`. |
| Eigenvalues (sorted) / PSD check | P1(d,f) | `eigenvalues` | Medium | `psd_fix.is_psd(matrix, tol)` plus `psd_fix.eigenvalues(matrix)`. |
| Frobenius distance between matrices | P1(f,i) | `frobenius_distance` | Medium | `psd_fix.frobenius_distance(a, b)`. This is the objective Higham minimises, so it belongs next to the repair. |
| Correlation + σ → covariance | P1(e,f) | `corr_to_cov` | **High** | `covariance.corr_to_cov(corr, std)` and its inverse `cov_to_corr(cov)`. `psd_fix` already does this conversion internally (`_to_correlation` / `_from_correlation`), so expose it. |
| Portfolio variance w′Σw | P1(e,f), P5(d,e) | `portfolio_variance` | **High** | `portfolio_risk.portfolio_variance(cov, weights)`, with weights matched to the covariance labels. |

### 1.3 Exponential weighting (`covariance.py`)

| Calculation | Used in | Notebook helper | Priority | Suggestion |
|---|---|---|---|---|
| EW weights as a public function | P2(b) | `ew_weights_recent_first` | Medium | Make `_ew_weights` public, or add `ew_weights(n, lambda_, order="oldest_first"\|"recent_first")`. The private version is oldest-first, which makes "weight on the last k days" awkward to compute. |
| Effective sample size 1/Σw² | P2(b,g) | `effective_sample_size` | Medium | `ew_effective_sample_size(lambda_, n=None)`. With n=None return the closed form (1+λ)/(1−λ). |
| Half life ln 0.5 / ln λ | P2(b) | `half_life` | Medium | `ew_half_life(lambda_)`. |
| EW **volatility** of a single series | P2(d) | `ew_std_zero_mean` | **High** | See item 2.2: needs a centering option. |

### 1.4 Descriptive statistics (`distributions.py` or new `moments.py`)

| Calculation | Used in | Notebook helper | Priority | Suggestion |
|---|---|---|---|---|
| First four moments (mean, std, skew, excess kurtosis) | P2(c), P3(a), P4(a) | `four_moments` | **High** | `moments(data, bias=False)`, DataFrame in, one row per column out. Needed in almost every assignment. |
| AICc for a fitted **normal** | P4(c) | inline in the AICc cell | **High** | `normal_distribution_aicc` to match `t_distribution_aicc`. Better, a `compare_distributions(data, candidates=["normal","t"])` that returns LL / AICc / chosen. |
| Two-regime volatility break scan | P2(b) | `two_regime_loglik` | Low | Assignment-specific; leave it inline. |

### 1.5 Copulas (`copula.py`)

| Calculation | Used in | Notebook helper | Priority | Suggestion |
|---|---|---|---|---|
| Pseudo-observations (rank/(n+1)) and PIT through **chosen** margins | P4(b,d) | inline | **High** | `to_uniforms(data, margins="empirical"\|dict_of_fitted)`. |
| Per-observation copula log density (Gaussian and t) | P4(d,i) | `gaussian_copula_logpdf`, `t_copula_logpdf` | **High** | Public `gaussian_copula_logpdf(U, R)` and `t_copula_logpdf(U, R, nu)` that work on **uniforms**. The existing fit/compare functions can then be thin wrappers around them. |
| Gaussian copula simulation | P4(e) | inline | **High** | `simulate_copula(R, n, copula="gaussian"\|"t", nu=None, seed)` returning the **uniforms**. |
| Map simulated uniforms through margins → P&L | P4(e) | `uniforms_to_portfolio_pnl` | Medium | `uniforms_to_returns(U, margins)`. |
| Joint-tail counts (data and simulated) | P4(b,f) | `joint_tail_counts`, `implied_tail_days` | Low | `joint_tail_counts(U, q=0.025, tail="lower"\|"upper"\|"both")`. |
| Copula information criteria with an explicit parameter count | P4(d) | `copula_ic` | Low | See item 2.4. |

### 1.6 Regression and factor models (new `factor_model.py`)

| Calculation | Used in | Notebook helper | Priority | Suggestion |
|---|---|---|---|---|
| OLS regression (α, β, residuals) | P5(b) | `ols_market_model` | **High** | `fit_ols_regression(y, x)` returning α, β, residuals and residual σ, to match `fit_t_regression`. |
| Factor-model covariance ββ′σ²_M + Σ_ε | P5(d,f) | `model_implied_cov` | Medium | `factor_model_covariance(betas, factor_cov, resid_cov, resid_structure="full"\|"diagonal")`. |
| Factor-model simulation | P5(c) | `factor_cov`, `simulate_factor_model_var` | Medium | `simulate_factor_model(betas, factor_cov, resid_cov, n, resid_structure=..., seed)` → simulated asset returns. |

---

## 2. Existing functions that did not fully fit

| # | Function | Issue found | Effect in Assignment 2 | Suggested fix |
|---|---|---|---|---|
| 2.1 | `cholesky.cholesky_psd` | On a **non-PSD** input it does not raise an error. It takes `sqrt(negative)` and silently returns `NaN` (with a RuntimeWarning). | P1(d): the pairwise matrix gave a NaN factor instead of a clear failure. | Raise `ValueError` when the pivot is below `-tolerance`, or add `on_negative="raise"\|"nan"\|"zero"`. |
| 2.2 | `covariance.ew_covariance_matrix` | Always centers on the **EW-weighted mean**. There is no option to center at zero or at the sample mean. | P2: the assignment says "remove the sample mean", which gives σ = 2.35% at λ=0.94; the Library gives 2.21%. Inline code was needed. | Add `center="ew_mean"\|"sample_mean"\|"zero"` (default unchanged). |
| 2.3 | `copula.*` (`_fit_t_margins`, `gaussian_copula_log_likelihood`, `fit_t_copula`, `simulate_t_copula_portfolio_var_es`) | Margins are **hard-coded to t**. You cannot pass margins chosen by AICc, or already-computed uniforms. | P4: X1 was chosen normal. The Library still matched only because X1's t fit drove ν → 10⁹, which is effectively normal. | Accept `margins=` (dict of fitted distributions or `"t"`/`"normal"` per column) or a `uniforms=` argument. |
| 2.4 | `copula.compare_copulas` | Gaussian AICc is plain −2LL (K=0), t uses K=1; correlation parameters are not counted. Only totals are returned. | P4(d): I counted the 3 correlations for both copulas (K=3 vs 4). The ranking is the same but the reported AICc/BIC values differ. | Add `count_correlation_params=True`; return per-observation log likelihood as well. |
| 2.5 | `copula.simulate_t_copula_portfolio_var_es` | Returns only VaR/ES at **one α**, never the simulated draws. There is no Gaussian counterpart with the same interface (`portfolio_risk.calculate_portfolio_var_es` needs holdings × price, a different input). | P4(e,f): the simulation had to be rewritten to report both α levels and count joint tails. | Split it into `simulate_copula` → `uniforms_to_returns` → `var_es_table`; keep the current function as a wrapper. |
| 2.6 | `risk_metrics.calculate_var` / `calculate_es` | Only `"normal"`/`"t"`; no historical option, and no way to pass in σ. | P2, P3, P4, P5. | See 1.1. |
| 2.7 | `utils.aicc` | Not re-exported by `Risk_Library`, so callers must use `from utils import aicc`. | P4(c,d). | Add it to `Risk_Library.py`. |
| 2.8 | `distributions.fit_t_distribution` | ν is unbounded: on normal data it returns ν ≈ 10⁹. This is not wrong, but it is awkward to display and use. | P4(c): X1. | Optionally cap ν (e.g. 1,000) or flag "ν at bound → use normal". |
| 2.9 | `psd_fix.fix_psd_matrix(method="higham")` | Converges to a smallest eigenvalue of −4.9e−10 (within tolerance), so a strict `np.linalg.cholesky` could still fail on the output. | P1(f): harmless here, because `cholesky_psd` tolerates it. | Optionally clip the final eigenvalues at 0 (or `epsilon`) after the last projection. |

---

## 3. Library functions that were used as-is

`covariance_matrix` (both missing strategies), `fix_psd_matrix` (both methods), `cholesky_psd`, `return_calculate`,
`ew_covariance_matrix` (for comparison), `fit_normal_distribution`, `fit_t_distribution`, `t_distribution_aicc`, `calculate_var`
(normal/t), `kendall_correlation_matrix`, `gaussian_copula_log_likelihood` and `fit_t_copula` (as cross-checks: they matched the
inline versions to 0.01 log-likelihood units and 0.002 in ν), `t_copula_tail_dependence`, and `simulate_normal`.

## 4. Suggested order of work

1. **Risk measures:** historical VaR/ES, parameter-based VaR, and a VaR/ES table (1.1, 2.6).
2. **Basic tools:** `four_moments`, `portfolio_variance`, `corr_to_cov`/`cov_to_corr`, normal AICc, and exporting `aicc` (1.2, 1.4, 2.7).
3. **Fixes:** `cholesky_psd` failure handling and the EW centering option (2.1, 2.2).
4. **Copula restructure:** uniforms-based log densities, a generic simulator, and margins chosen by the caller (1.5, 2.3–2.5).
5. **Factor-model module:** OLS, model covariance, and simulation (1.6).
6. EW helpers and the remaining low-priority items.

Each new function should get a pytest in `Library/tests/`. Where no course test file exists, the Assignment 2 numbers in this report
can serve as regression values: for example P1 pairwise smallest eigenvalue −0.01107, P2 EW(0.94) σ 0.023532, P4 Gaussian copula
LL 350.454, and P5 sample cov = model cov to 1e−19.
