"""
Value at Risk (VaR) and Expected Shortfall (ES) for a single return series.

Covers: FINTECH545 Tests 8.1 - 8.6 (VaR/ES under a Normal or Student's t
distributional assumption, computed either analytically or by simulating
from the fitted distribution).
"""

import numpy as np
import pandas as pd
from scipy import stats

from distributions import fit_normal_distribution, fit_t_distribution


def _fit(data: pd.Series, distribution: str):
    """Fit the requested distribution and return (mu, sigma, nu). `nu` is
    None for the Normal distribution (it has no degrees-of-freedom param)."""
    if distribution == "normal":
        fitted = fit_normal_distribution(data)
        return fitted["mu"], fitted["sigma"], None
    elif distribution == "t":
        fitted = fit_t_distribution(data)
        return fitted["mu"], fitted["sigma"], fitted["nu"]
    else:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. Expected one of: 'normal', 't'."
        )


def calculate_var(
    data: pd.Series,
    distribution: str = "normal",
    method: str = "analytical",
    alpha: float = 0.05,
    num_simulations: int = 100_000,
    seed: int = None,
) -> pd.Series:
    """
    Compute Value at Risk (VaR) for a return series, expressed as a
    positive loss number (i.e. VaR = 0.05 means a 5% loss at the given
    confidence level).

    Expected input
    ---------------
    data : pd.Series (or array-like)
        A 1-D sample of returns.
    distribution : str, default "normal"
        Which distribution to assume for the returns:
          - "normal": fit a Normal distribution (`distributions.fit_normal_distribution`).
          - "t": fit a Student's t distribution (`distributions.fit_t_distribution`).
    method : str, default "analytical"
        How to compute the quantile once the distribution is fit:
          - "analytical": use the closed-form quantile of the fitted
            distribution.
          - "simulation": draw `num_simulations` samples from the fitted
            distribution and use the empirical quantile. Converges to the
            analytical answer as `num_simulations` grows, but carries
            Monte Carlo noise (particularly in the tail, where relatively
            few simulated points fall).
    alpha : float, default 0.05
        The tail probability (e.g. 0.05 for a 95% VaR).
    num_simulations : int, default 100000
        Only used when `method="simulation"`.
    seed : int, optional
        Only used when `method="simulation"` — seed for reproducible draws.

    Output
    ------
    pd.Series
        Index ["VaR Absolute", "VaR Diff from Mean"]:
          - "VaR Absolute": the loss threshold itself, i.e. -(fitted mean + quantile).
          - "VaR Diff from Mean": the same, but relative to the fitted
            mean rather than to zero (VaR Absolute - fitted mean's
            contribution) -- equivalently, VaR Absolute + fitted mean.

    Raises
    ------
    ValueError
        If `distribution` or `method` is not a supported option.
    """
    if method not in ("analytical", "simulation"):
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: 'analytical', 'simulation'."
        )

    mu, sigma, nu = _fit(data, distribution)

    if method == "analytical":
        if distribution == "normal":
            quantile = stats.norm.ppf(alpha, loc=0, scale=sigma)
        else:
            quantile = stats.t.ppf(alpha, df=nu, loc=0, scale=sigma)
        var_diff_from_mean = -quantile
    else:
        rng = np.random.default_rng(seed)
        if distribution == "normal":
            simulated = stats.norm.rvs(loc=0, scale=sigma, size=num_simulations, random_state=rng)
        else:
            simulated = stats.t.rvs(df=nu, loc=0, scale=sigma, size=num_simulations, random_state=rng)
        var_diff_from_mean = -np.quantile(simulated, alpha)

    var_absolute = var_diff_from_mean - mu
    return pd.Series({"VaR Absolute": var_absolute, "VaR Diff from Mean": var_diff_from_mean})


def calculate_es(
    data: pd.Series,
    distribution: str = "normal",
    method: str = "analytical",
    alpha: float = 0.05,
    num_simulations: int = 100_000,
    seed: int = None,
) -> pd.Series:
    """
    Compute Expected Shortfall (ES, a.k.a. Conditional VaR) for a return
    series: the expected loss given that the loss exceeds the VaR
    threshold, expressed as a positive loss number.

    Expected input / Output
    ------------------------
    Same shape as `calculate_var` above, but returns
    Index ["ES Absolute", "ES Diff from Mean"] instead. See `calculate_var`
    for the meaning of each parameter.

    Raises
    ------
    ValueError
        If `distribution` or `method` is not a supported option.
    """
    if method not in ("analytical", "simulation"):
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: 'analytical', 'simulation'."
        )

    mu, sigma, nu = _fit(data, distribution)

    if method == "analytical":
        if distribution == "normal":
            z_alpha = stats.norm.ppf(alpha)
            es_diff_from_mean = sigma * stats.norm.pdf(z_alpha) / alpha
        else:
            t_alpha = stats.t.ppf(alpha, df=nu)
            es_diff_from_mean = (
                sigma * (stats.t.pdf(t_alpha, df=nu) / alpha) * (nu + t_alpha**2) / (nu - 1)
            )
    else:
        rng = np.random.default_rng(seed)
        if distribution == "normal":
            simulated = stats.norm.rvs(loc=0, scale=sigma, size=num_simulations, random_state=rng)
        else:
            simulated = stats.t.rvs(df=nu, loc=0, scale=sigma, size=num_simulations, random_state=rng)
        cutoff = np.quantile(simulated, alpha)
        es_diff_from_mean = -simulated[simulated <= cutoff].mean()

    es_absolute = es_diff_from_mean - mu
    return pd.Series({"ES Absolute": es_absolute, "ES Diff from Mean": es_diff_from_mean})
