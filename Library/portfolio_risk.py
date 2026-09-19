"""
Portfolio-level Value at Risk (VaR) and Expected Shortfall (ES), estimated
by simulating each holding's returns jointly via a Gaussian copula.

A Gaussian copula lets each asset keep its own, individually-fitted return
distribution (e.g. one asset Normal, another Student's t) while still
capturing the dependence between them: each asset's historical returns are
mapped to a uniform score via its own fitted CDF, those uniforms are
mapped to standard-normal "copula scores" whose correlation is estimated
from history, correlated normal draws are simulated from that correlation
matrix, and mapped back through each asset's own inverse CDF to get
simulated returns in the original units.

Covers: FINTECH545 Test 9.1 (VaR/ES on a multi-asset portfolio with
per-asset Normal/T distributions, via copula simulation).
"""

import numpy as np
import pandas as pd
from scipy import stats

from distributions import fit_normal_distribution, fit_t_distribution
from cholesky import cholesky_psd

_SUPPORTED_DISTRIBUTIONS = ("normal", "t")


def _fit_marginal(returns: np.ndarray, distribution: str):
    """Fit one asset's marginal distribution, returning (mu, sigma, nu).
    `nu` is None for "normal" (no degrees-of-freedom parameter)."""
    if distribution == "normal":
        fitted = fit_normal_distribution(returns)
        return fitted["mu"], fitted["sigma"], None
    elif distribution == "t":
        fitted = fit_t_distribution(returns)
        return fitted["mu"], fitted["sigma"], fitted["nu"]
    else:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            f"Expected one of: {_SUPPORTED_DISTRIBUTIONS}."
        )


def _marginal_cdf(x, distribution, mu, sigma, nu):
    if distribution == "normal":
        return stats.norm.cdf(x, loc=mu, scale=sigma)
    return stats.t.cdf(x, df=nu, loc=mu, scale=sigma)


def _marginal_ppf(u, distribution, mu, sigma, nu):
    if distribution == "normal":
        return stats.norm.ppf(u, loc=mu, scale=sigma)
    return stats.t.ppf(u, df=nu, loc=mu, scale=sigma)


def calculate_portfolio_var_es(
    portfolio: pd.DataFrame,
    returns: pd.DataFrame,
    alpha: float = 0.05,
    num_simulations: int = 100_000,
    seed: int = None,
) -> pd.DataFrame:
    """
    Simulate joint portfolio P&L via a Gaussian copula and compute VaR/ES
    per holding and for the total portfolio.

    Expected input
    ---------------
    portfolio : pd.DataFrame
        One row per holding, with columns:
          - "Stock": the asset identifier, matching a column name in `returns`.
          - "Holding": number of shares/units held.
          - "Starting Price": current price per unit.
          - "Distribution": which marginal distribution to fit for this
            asset's returns -- "Normal" or "T" (case-insensitive).
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
        One row per holding plus a "Total" row, with columns
        [f"VaR{conf}", f"ES{conf}", f"VaR{conf}_Pct", f"ES{conf}_Pct"]
        where conf = round((1 - alpha) * 100) (e.g. "VaR95" for alpha=0.05).
        The dollar columns are in the same currency as `Starting Price`;
        the "_Pct" columns express the same VaR/ES as a fraction of that
        holding's (or the total portfolio's) starting value.
    """
    confidence = round((1 - alpha) * 100)
    var_col = f"VaR{confidence}"
    es_col = f"ES{confidence}"

    stocks = portfolio["Stock"].tolist()
    n_assets = len(stocks)

    # Fit each asset's own marginal distribution, then map its historical
    # returns to standard-normal "copula scores" via probability integral
    # transform (CDF) followed by the standard normal's inverse CDF.
    marginals = {}
    normal_scores = pd.DataFrame(index=returns.index)
    for _, row in portfolio.iterrows():
        stock = row["Stock"]
        distribution = row["Distribution"].strip().lower()
        asset_returns = returns[stock].values

        mu, sigma, nu = _fit_marginal(asset_returns, distribution)
        marginals[stock] = (distribution, mu, sigma, nu)

        uniform_scores = _marginal_cdf(asset_returns, distribution, mu, sigma, nu)
        normal_scores[stock] = stats.norm.ppf(uniform_scores)

    # The correlation of the copula scores captures the dependence
    # structure between assets, independent of each asset's own marginal.
    copula_correlation = normal_scores.corr()
    factor = cholesky_psd(copula_correlation).values

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_assets, num_simulations))
    correlated_z = factor @ z
    correlated_uniform = stats.norm.cdf(correlated_z)

    starting_values = {}
    simulated_pnl = {}
    for i, stock in enumerate(stocks):
        distribution, mu, sigma, nu = marginals[stock]
        simulated_returns = _marginal_ppf(correlated_uniform[i], distribution, mu, sigma, nu)

        holding_value = float(
            portfolio.loc[portfolio["Stock"] == stock, "Holding"].iloc[0]
            * portfolio.loc[portfolio["Stock"] == stock, "Starting Price"].iloc[0]
        )
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
