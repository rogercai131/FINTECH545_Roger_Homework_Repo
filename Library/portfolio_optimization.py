"""
Constructing portfolio weights: risk parity (equal or budgeted risk
contribution) and maximum Sharpe ratio.

Covers: FINTECH545 Tests 10.1 - 10.4 (risk parity with equal and custom
risk budgets, and max-Sharpe optimization with different weight bounds).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_risk_parity_weights(
    cov_matrix: pd.DataFrame, risk_budgets=None
) -> pd.Series:
    """
    Find portfolio weights so that each asset's contribution to total
    portfolio risk (volatility) matches a target risk budget.

    Expected input
    ---------------
    cov_matrix : pd.DataFrame
        A square covariance matrix of asset returns.
    risk_budgets : array-like, optional
        The target share of total portfolio risk each asset should
        contribute, one value per asset (they don't need to already sum to
        1 -- they're normalized internally). Defaults to an equal budget
        for every asset (plain equal-risk-contribution / "risk parity").
        E.g. to give the 5th asset half the risk budget of the others,
        pass `[1, 1, 1, 1, 0.5]`.

    Output
    ------
    pd.Series
        Portfolio weights, indexed by `cov_matrix`'s columns, summing to 1.
    """
    labels = cov_matrix.columns
    cov = cov_matrix.values
    n = cov.shape[0]

    budgets = np.ones(n) if risk_budgets is None else np.asarray(risk_budgets, dtype=float)
    budgets = budgets / budgets.sum()

    def objective(w):
        portfolio_vol = np.sqrt(w @ cov @ w)
        # Each asset's marginal contribution to portfolio volatility,
        # scaled by its weight, gives its absolute risk contribution;
        # dividing by total portfolio vol gives its *share* of risk.
        marginal_contribution = cov @ w / portfolio_vol
        risk_share = (w * marginal_contribution) / portfolio_vol
        return np.sum((risk_share - budgets) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(1e-8, 1.0)] * n
    initial_guess = np.ones(n) / n

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-16, "maxiter": 1000},
    )

    return pd.Series(result.x, index=labels, name="W")


def calculate_max_sharpe_weights(
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series,
    risk_free_rate: float = 0.0,
    lower_bound=0.0,
    upper_bound=1.0,
) -> pd.Series:
    """
    Find the portfolio weights that maximize the Sharpe ratio
    (expected excess return / volatility), subject to weights summing to
    1 and staying within per-asset bounds.

    Expected input
    ---------------
    cov_matrix : pd.DataFrame
        A square covariance matrix of asset returns.
    expected_returns : pd.Series (or array-like)
        Expected return for each asset, aligned with `cov_matrix`'s
        columns.
    risk_free_rate : float, default 0.0
        The risk-free rate used in the Sharpe ratio's excess-return
        numerator.
    lower_bound, upper_bound : float or array-like, default 0.0 / 1.0
        Per-asset weight bounds. Pass a single number to apply the same
        bound to every asset (e.g. `lower_bound=0` for long-only, or
        `lower_bound=0.1, upper_bound=0.5` to force every position between
        10% and 50%), or an array-like with one value per asset for
        asset-specific bounds.

    Output
    ------
    pd.Series
        Portfolio weights, indexed by `cov_matrix`'s columns, summing to 1.
    """
    labels = cov_matrix.columns
    cov = cov_matrix.values
    means = np.asarray(expected_returns, dtype=float)
    n = cov.shape[0]

    lower = np.broadcast_to(lower_bound, n)
    upper = np.broadcast_to(upper_bound, n)
    bounds = list(zip(lower, upper))

    def negative_sharpe_ratio(w):
        portfolio_return = w @ means
        portfolio_vol = np.sqrt(w @ cov @ w)
        return -(portfolio_return - risk_free_rate) / portfolio_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    initial_guess = np.ones(n) / n

    result = minimize(
        negative_sharpe_ratio,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-16, "maxiter": 2000},
    )

    return pd.Series(result.x, index=labels, name="W")
