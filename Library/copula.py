"""
Copula-based dependence modeling: fitting each variable's own marginal
distribution separately from how the variables move together (their
"copula"), comparing a Gaussian vs. a Student's t copula, tail dependence,
simulating joint returns from a Gaussian or t copula (`simulate_copula`),
and simulating a multi-asset portfolio's VaR/ES under a t copula.

Covers: FINTECH545 Tests 13.5 - 13.9.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist, chi2
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

from distributions import fit_normal_distribution, fit_t_distribution
from covariance import kendall_correlation_matrix
from cholesky import cholesky_psd
from psd_fix import fix_psd_matrix
from utils import aicc as _aicc


def _fit_t_margins(data: pd.DataFrame):
    """
    Fit each column's own ("generalized", i.e. all 3 parameters free)
    Student's t distribution, and map each column's observations to their
    corresponding uniform (0, 1) score via that column's own fitted CDF.

    Returns
    -------
    (marginals, uniform_scores) : tuple
        `marginals` is a dict {column_label: {"mu", "sigma", "nu"}}.
        `uniform_scores` is a same-shaped np.ndarray of the probability
        integral transform of `data` under each column's own fit.
    """
    labels = data.columns
    values = data.values
    marginals = {}
    uniform_scores = np.zeros_like(values)

    for i, label in enumerate(labels):
        fitted = fit_t_distribution(values[:, i])
        marginals[label] = fitted
        uniform_scores[:, i] = t_dist.cdf(
            values[:, i], df=fitted["nu"], loc=fitted["mu"], scale=fitted["sigma"]
        )

    return marginals, uniform_scores


def _mvt_log_density_zero_mean(z: np.ndarray, scale_inv: np.ndarray, log_det_scale: float, nu: float, k: int):
    """Log-density of a zero-mean multivariate t (scale matrix implied by
    `scale_inv`/`log_det_scale`, degrees of freedom `nu`) at each row of `z`."""
    mahalanobis_sq = np.einsum("ij,jk,ik->i", z, scale_inv, z)
    log_norm_const = gammaln((nu + k) / 2) - gammaln(nu / 2) - (k / 2) * np.log(nu * np.pi) - 0.5 * log_det_scale
    return log_norm_const - ((nu + k) / 2) * np.log(1 + mahalanobis_sq / nu)


def gaussian_copula_log_likelihood(data: pd.DataFrame) -> float:
    """
    Log-likelihood of the Gaussian copula fit to `data`, with each
    column's own marginal fit as a ("generalized") Student's t
    distribution.

    Expected input
    ---------------
    data : pd.DataFrame
        Rows are observations, columns are variables. Should not contain
        NaN values.

    Output
    ------
    float
        The Gaussian copula's log-likelihood (this is the log-likelihood
        of the *dependence structure* alone -- it does not include the
        marginal distributions' own log-likelihood).
    """
    _, uniform_scores = _fit_t_margins(data)
    normal_scores = norm.ppf(uniform_scores)

    correlation = kendall_correlation_matrix(data).values
    correlation_inv = np.linalg.inv(correlation)
    _, log_det_correlation = np.linalg.slogdet(correlation)
    k = correlation.shape[0]

    # The Gaussian copula density simplifies to this closed form (the
    # standard-normal marginal densities cancel out of the ratio).
    mahalanobis_sq = np.einsum("ij,jk,ik->i", normal_scores, correlation_inv, normal_scores)
    identity_term = np.einsum("ij,ij->i", normal_scores, normal_scores)  # z' I z
    log_density = -0.5 * log_det_correlation - 0.5 * (mahalanobis_sq - identity_term)

    return float(np.sum(log_density))


def fit_t_copula(data: pd.DataFrame) -> dict:
    """
    Fit a Student's t copula to `data`, with each column's own marginal
    fit as a ("generalized") Student's t distribution.

    Expected input
    ---------------
    data : pd.DataFrame
        Rows are observations, columns are variables. Should not contain
        NaN values.

    Output
    ------
    dict with keys:
      - "correlation": pd.DataFrame, the (Kendall's-tau-based, PSD-repaired)
        correlation matrix used by the copula.
      - "nu": float, the copula's own fitted degrees of freedom (found by
        1-D profile likelihood -- this is a property of the *dependence
        structure* and is generally different from any individual
        column's own marginal `nu`).
      - "log_likelihood": float, the t copula's log-likelihood at the
        fitted `nu` (again, the dependence structure's log-likelihood
        alone, not including the marginals').
    """
    _, uniform_scores = _fit_t_margins(data)

    correlation_df = kendall_correlation_matrix(data)
    nu, log_likelihood = _fit_t_copula_nu(uniform_scores, correlation_df.values)

    return {
        "correlation": correlation_df,
        "nu": nu,
        "log_likelihood": log_likelihood,
    }


def _fit_t_copula_nu(uniform_scores: np.ndarray, correlation: np.ndarray):
    """
    Fit a t copula's degrees of freedom by 1-D profile likelihood, holding
    its correlation matrix fixed.

    Returns
    -------
    (nu, log_likelihood) : tuple of floats
        The fitted degrees of freedom and the copula log-likelihood there.
    """
    correlation_inv = np.linalg.inv(correlation)
    _, log_det_correlation = np.linalg.slogdet(correlation)
    k = correlation.shape[0]

    def copula_log_likelihood(nu):
        z = t_dist.ppf(uniform_scores, df=nu)
        joint_log_density = _mvt_log_density_zero_mean(z, correlation_inv, log_det_correlation, nu, k)
        marginal_log_density = t_dist.logpdf(z, df=nu).sum(axis=1)
        # The t copula density is the joint density divided by the
        # product of each (standard) univariate t marginal density.
        return np.sum(joint_log_density - marginal_log_density)

    result = minimize_scalar(
        lambda nu: -copula_log_likelihood(nu), bounds=(2.0001, 1000), method="bounded",
        options={"xatol": 1e-10},
    )
    return result.x, -result.fun


def _simulate_copula_uniforms(correlation: np.ndarray, copula: str, nu, num_simulations: int, rng) -> np.ndarray:
    """
    Draw correlated uniform (0, 1) scores from a Gaussian or t copula.

    Gaussian: Z = L @ standard normals (L = Cholesky factor of the
    correlation), U = Phi(Z).
    t: the same correlated normals divided by sqrt(chi2_nu / nu) -- which
    makes them jointly multivariate-t -- then U = t_nu CDF of those.

    Returns an array of shape (num_assets, num_simulations).
    """
    factor = cholesky_psd(pd.DataFrame(correlation)).values
    normal_draws = factor @ rng.standard_normal((correlation.shape[0], num_simulations))
    if copula == "gaussian":
        return norm.cdf(normal_draws)
    chi_square_draws = chi2.rvs(nu, size=num_simulations, random_state=rng)
    t_draws = normal_draws / np.sqrt(chi_square_draws / nu)
    return t_dist.cdf(t_draws, df=nu)


def compare_copulas(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fit both a Gaussian and a Student's t copula to `data` and compare
    them by AICc and BIC (lower is better for both).

    Expected input
    ---------------
    data : pd.DataFrame
        Rows are observations, columns are variables. Should not contain
        NaN values.

    Output
    ------
    pd.DataFrame
        Indexed by ["Gaussian", "T"], columns ["LL", "K", "AICC", "BIC"].
        `K` is the number of *extra* parameters relative to the Gaussian
        copula's dependence structure (0 for Gaussian; 1 for the t copula,
        its extra degrees-of-freedom parameter) -- the correlation matrix
        itself isn't counted, since both copulas use the same one.
    """
    num_obs = len(data)

    gaussian_ll = gaussian_copula_log_likelihood(data)
    t_fit = fit_t_copula(data)
    t_ll = t_fit["log_likelihood"]

    rows = {}
    for name, log_likelihood, num_params in [("Gaussian", gaussian_ll, 0), ("T", t_ll, 1)]:
        rows[name] = {
            "LL": log_likelihood,
            "K": num_params,
            "AICC": _aicc(log_likelihood, num_params, num_obs) if num_params > 0 else -2 * log_likelihood,
            "BIC": -2 * log_likelihood + num_params * np.log(num_obs),
        }

    return pd.DataFrame(rows).T


def t_copula_tail_dependence(correlation: pd.DataFrame, nu: float) -> pd.DataFrame:
    """
    Lower tail dependence coefficient for every pair of variables under a
    fitted t copula: the (limiting) probability that one variable is in
    its extreme lower tail given that another is too. Unlike a Gaussian
    copula (whose tail dependence is always 0), a t copula can capture the
    tendency for extreme moves to happen together -- important for risk
    management, where joint crashes matter more than the joint-normal
    dependence structure alone suggests.

    Expected input
    ---------------
    correlation : pd.DataFrame
        The t copula's correlation matrix (e.g. `fit_t_copula(...)["correlation"]`).
    nu : float
        The t copula's degrees of freedom (e.g. `fit_t_copula(...)["nu"]`).

    Output
    ------
    pd.DataFrame
        One row per unordered pair of variables, columns ["I", "J", "Rho", "Lambda"]:
        `I`/`J` are 1-based positions of the pair (matching `correlation`'s
        column order), `Rho` their correlation, `Lambda` their lower tail
        dependence coefficient.
    """
    labels = list(correlation.columns)
    corr_values = correlation.values
    n = len(labels)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rho = corr_values[i, j]
            quantile_arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            lower_tail_dependence = 2 * t_dist.cdf(quantile_arg, df=nu + 1)
            rows.append((i + 1, j + 1, rho, lower_tail_dependence))

    return pd.DataFrame(rows, columns=["I", "J", "Rho", "Lambda"])


def simulate_t_copula_portfolio_var_es(
    portfolio: pd.DataFrame,
    returns: pd.DataFrame,
    alpha: float = 0.05,
    num_simulations: int = 100_000,
    seed: int = None,
) -> pd.DataFrame:
    """
    Simulate joint portfolio P&L via a Student's t copula (each asset
    keeping its own fitted marginal return distribution) and compute
    VaR/ES per holding and for the total portfolio.

    This is the t-copula counterpart to `portfolio_risk.calculate_portfolio_var_es`
    (which uses a Gaussian copula and lets each asset's marginal be
    Normal or T, chosen per-asset). Here every asset's marginal is fit as
    a ("generalized") Student's t automatically, and the copula itself
    also has its own fitted degrees of freedom -- capturing tail
    dependence between assets that a Gaussian copula cannot.

    Expected input
    ---------------
    portfolio : pd.DataFrame
        One row per holding, with columns "Stock" (matching a column name
        in `returns`) and "currentValue" (the holding's current dollar value).
    returns : pd.DataFrame
        Historical returns, one column per asset (named to match
        `portfolio["Stock"]`), one row per historical period.
    alpha : float, default 0.05
        Tail probability for VaR/ES (e.g. 0.05 for a 95% VaR/ES).
    num_simulations : int, default 100000
        Number of joint scenarios to simulate.
    seed : int, optional
        Seed for reproducible simulation draws.

    Output
    ------
    pd.DataFrame
        Same shape as `portfolio_risk.calculate_portfolio_var_es`'s output:
        one row per holding plus "Total", columns
        [f"VaR{conf}", f"ES{conf}", f"VaR{conf}_Pct", f"ES{conf}_Pct"].
    """
    confidence = round((1 - alpha) * 100)
    var_col = f"VaR{confidence}"
    es_col = f"ES{confidence}"

    stocks = portfolio["Stock"].tolist()
    n_assets = len(stocks)

    marginals, uniform_scores = _fit_t_margins(returns[stocks])
    t_copula = fit_t_copula(returns[stocks])
    correlation = t_copula["correlation"]
    nu_copula = t_copula["nu"]

    rng = np.random.default_rng(seed)
    simulated_uniform = _simulate_copula_uniforms(
        correlation.values, "t", nu_copula, num_simulations, rng
    )

    starting_values = {}
    simulated_pnl = {}
    for i, stock in enumerate(stocks):
        fitted = marginals[stock]
        simulated_returns = t_dist.ppf(
            simulated_uniform[i], df=fitted["nu"], loc=fitted["mu"], scale=fitted["sigma"]
        )
        holding_value = float(portfolio.loc[portfolio["Stock"] == stock, "currentValue"].iloc[0])
        starting_values[stock] = holding_value
        simulated_pnl[stock] = holding_value * simulated_returns

    total_pnl = sum(simulated_pnl.values())
    total_value = sum(starting_values.values())

    rows = []
    for stock in stocks:
        pnl = simulated_pnl[stock]
        cutoff = np.quantile(pnl, alpha)
        var = -cutoff
        es = -pnl[pnl <= cutoff].mean()
        value = starting_values[stock]
        rows.append((stock, var, es, var / value, es / value))

    cutoff = np.quantile(total_pnl, alpha)
    var_total = -cutoff
    es_total = -total_pnl[total_pnl <= cutoff].mean()
    rows.append(("Total", var_total, es_total, var_total / total_value, es_total / total_value))

    return pd.DataFrame(
        rows, columns=["Stock", var_col, es_col, f"{var_col}_Pct", f"{es_col}_Pct"]
    )


_COPULAS = ("gaussian", "t")
_MARGINALS = ("normal", "t")
_CORRELATIONS = ("normal_scores", "spearman", "kendall")


def simulate_copula(
    returns: pd.DataFrame,
    copula: str = "gaussian",
    marginals="t",
    correlation: str = None,
    nu: float = None,
    num_simulations: int = 100_000,
    seed: int = None,
    output: str = "simulations",
):
    """
    Simulate joint returns from a Gaussian or Student's t copula: each
    column keeps its own fitted marginal distribution, while the copula
    supplies how the columns move together.

    Steps
    -----
    1. Fit each column's marginal distribution (Normal or t).
    2. Map the historical returns to uniform scores U = F(x) with each
       column's own fitted CDF.
    3. Estimate the copula's correlation matrix (and, for a t copula, its
       degrees of freedom nu by profile likelihood on U).
    4. Simulate correlated uniforms from the copula, and map each column
       back to returns with its own inverse CDF: x = F^-1(U).

    Expected input
    ---------------
    returns : pd.DataFrame
        Historical returns, one column per asset, no NaN values.
    copula : str, default "gaussian"
          - "gaussian": no tail dependence (joint extremes are rare).
          - "t": tail dependence controlled by the copula's own nu
            (smaller nu = extreme moves happen together more often).
    marginals : str or dict, default "t"
        The marginal distribution for every column ("normal" or "t"), or a
        dict {column: "normal"/"t"} to choose per column. Columns missing
        from the dict default to "t".
    correlation : str, optional
        How the copula's correlation matrix is estimated:
          - "normal_scores": Pearson correlation of the normal scores
            Phi^-1(U) (Gaussian copula only — the approach used by
            `portfolio_risk.calculate_portfolio_var_es`).
          - "spearman": Spearman rank correlation of the returns.
          - "kendall": sin(pi/2 * Kendall's tau), PSD-repaired (the
            approach used by `fit_t_copula`).
        Default: "normal_scores" for a Gaussian copula, "kendall" for a t
        copula. A rank-based matrix that isn't PSD is repaired with
        `psd_fix.fix_psd_matrix(method="near_psd")`.
    nu : float, optional
        t copula only: use this degrees of freedom instead of fitting it.
    num_simulations : int, default 100000
        Number of joint scenarios to simulate.
    seed : int, optional
        Seed for reproducible simulation draws.
    output : str, default "simulations"
          - "simulations": just the simulated returns DataFrame.
          - "both": the tuple (simulations, fit), where `fit` is a dict
            with "marginals" ({column: frozen scipy distribution}),
            "correlation" (pd.DataFrame), and "nu" (float, or None for a
            Gaussian copula).

    Output
    ------
    pd.DataFrame
        `num_simulations` rows, same columns as `returns`, holding the
        simulated returns (see `output` for the extra fit details).

    Raises
    ------
    ValueError
        If an option string is not supported, `correlation="normal_scores"`
        is combined with a t copula, or `nu` is given for a Gaussian copula.
    """
    if copula not in _COPULAS:
        raise ValueError(f"Unsupported copula '{copula}'. Expected one of: 'gaussian', 't'.")
    if output not in ("simulations", "both"):
        raise ValueError(f"Unsupported output '{output}'. Expected one of: 'simulations', 'both'.")
    if correlation is None:
        correlation = "normal_scores" if copula == "gaussian" else "kendall"
    if correlation not in _CORRELATIONS:
        raise ValueError(
            f"Unsupported correlation '{correlation}'. "
            "Expected one of: 'normal_scores', 'spearman', 'kendall'."
        )
    if copula == "t" and correlation == "normal_scores":
        raise ValueError("correlation='normal_scores' is only for copula='gaussian'.")
    if copula == "gaussian" and nu is not None:
        raise ValueError("nu is only used with copula='t'.")

    labels = returns.columns
    values = returns.values
    choices = {col: marginals for col in labels} if isinstance(marginals, str) else {
        col: marginals.get(col, "t") for col in labels
    }

    # 1-2. Fit each marginal and map the history to uniform scores.
    fitted_marginals = {}
    uniform_scores = np.zeros_like(values, dtype=float)
    for i, col in enumerate(labels):
        choice = choices[col].strip().lower()
        if choice not in _MARGINALS:
            raise ValueError(f"Unsupported marginal '{choice}' for '{col}'. Expected one of: 'normal', 't'.")
        fit = fit_normal_distribution if choice == "normal" else fit_t_distribution
        fitted_marginals[col] = fit(values[:, i], output="distribution")
        uniform_scores[:, i] = fitted_marginals[col].cdf(values[:, i])

    # 3. The copula's correlation matrix.
    if correlation == "normal_scores":
        corr = np.corrcoef(norm.ppf(uniform_scores), rowvar=False)
    elif correlation == "spearman":
        corr = returns.corr(method="spearman").values
    else:
        corr = kendall_correlation_matrix(returns).values
    if np.linalg.eigvalsh(corr)[0] < 0:
        corr = fix_psd_matrix(pd.DataFrame(corr, index=labels, columns=labels)).values

    if copula == "t" and nu is None:
        nu, _ = _fit_t_copula_nu(uniform_scores, corr)

    # 4. Simulate copula uniforms, then invert each column's own marginal.
    rng = np.random.default_rng(seed)
    simulated_uniform = _simulate_copula_uniforms(corr, copula, nu, num_simulations, rng)
    simulations = pd.DataFrame(
        {col: fitted_marginals[col].ppf(simulated_uniform[i]) for i, col in enumerate(labels)}
    )

    if output == "simulations":
        return simulations
    fit_details = {
        "marginals": fitted_marginals,
        "correlation": pd.DataFrame(corr, index=labels, columns=labels),
        "nu": nu,
    }
    return simulations, fit_details
