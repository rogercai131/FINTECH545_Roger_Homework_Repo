"""
Ex-post return and risk (volatility) attribution for a portfolio: given a
history of realized returns and a starting set of weights, break the
portfolio's total realized return and volatility down into how much each
holding (or, optionally, each systematic factor plus a leftover "Alpha")
contributed.

Covers: FINTECH545 Tests 11.1 (attribution to holdings) and 11.2
(attribution to factors, with an Alpha residual for whatever the factors
don't explain).
"""

import numpy as np
import pandas as pd


def _drift_weights_and_portfolio_returns(returns: np.ndarray, weights: np.ndarray):
    """
    Simulate a buy-and-hold portfolio: starting from `weights`, each period
    the portfolio earns the weighted return of its holdings, and each
    holding's weight then drifts with its own realized return (winners
    become a larger share of the portfolio, losers a smaller share).

    Returns
    -------
    (weight_path, portfolio_returns) : tuple
        `weight_path[t]` is the weight vector *going into* period t (i.e.
        before that period's return is applied) -- this is what should be
        multiplied by period t's returns to get each holding's
        contribution in that period. `portfolio_returns[t]` is the
        resulting portfolio return in period t.
    """
    num_periods, num_holdings = returns.shape
    weight_path = np.zeros((num_periods, num_holdings))
    portfolio_returns = np.zeros(num_periods)

    w = weights.copy()
    for t in range(num_periods):
        weight_path[t] = w
        portfolio_returns[t] = w @ returns[t]
        w = w * (1 + returns[t]) / (1 + portfolio_returns[t])

    return weight_path, portfolio_returns


def _carino_attribution(contributions: np.ndarray, portfolio_returns: np.ndarray) -> np.ndarray:
    """
    Carino smoothing: turns a matrix of *single-period* return
    contributions (one column per holding/factor, one row per period,
    rows summing to that period's portfolio return) into contributions to
    the *total, compounded* portfolio return, such that they still sum
    exactly to the total compounded portfolio return.
    """
    total_return = np.prod(1 + portfolio_returns) - 1
    total_scale = np.log(1 + total_return) / total_return

    # Per-period scaling factor; a period with ~zero return needs no
    # log-compounding adjustment (the limit of log(1+x)/x as x -> 0 is 1).
    period_scale = np.where(
        np.abs(portfolio_returns) > 1e-12,
        np.log(1 + portfolio_returns) / portfolio_returns,
        1.0,
    )

    return np.sum(contributions * period_scale[:, None], axis=0) / total_scale


def _vol_attribution(contributions: np.ndarray, portfolio_returns: np.ndarray) -> np.ndarray:
    """
    Attribute portfolio return volatility to each holding/factor as its
    covariance with the total portfolio return series, scaled by
    portfolio volatility -- these shares sum exactly to portfolio
    volatility, since summing the per-holding/factor series recovers the
    portfolio return series itself.
    """
    portfolio_vol = portfolio_returns.std(ddof=1)
    num_series = contributions.shape[1]
    covariances = np.array(
        [np.cov(contributions[:, i], portfolio_returns, ddof=1)[0, 1] for i in range(num_series)]
    )
    return covariances / portfolio_vol


def return_attribution(
    returns: pd.DataFrame,
    weights,
    factor_returns: pd.DataFrame = None,
    betas: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Break a portfolio's total realized return and return-volatility down
    by contributor, over a set of historical periods.

    Expected input
    ---------------
    returns : pd.DataFrame
        Historical returns, one column per holding, one row per period
        (chronological order).
    weights : array-like
        Starting portfolio weights, one per column of `returns`.
    factor_returns : pd.DataFrame, optional
        If given (together with `betas`), attribution is instead done by
        systematic *factor* (one column per factor, one row per period,
        same periods as `returns`), with a residual "Alpha" contributor
        capturing whatever the factors don't explain. If omitted (the
        default), attribution is done directly by holding.
    betas : pd.DataFrame, optional
        Required when `factor_returns` is given: factor loadings, indexed
        by the same holding labels as `returns`'s columns, with one column
        per factor (matching `factor_returns`'s columns).

    Output
    ------
    pd.DataFrame
        Indexed by ["TotalReturn", "Return Attribution", "Vol Attribution"],
        with one column per contributor (holdings, or factors + "Alpha")
        plus a "Portfolio" column with the portfolio-level total:
          - "TotalReturn": each contributor's own compounded return in
            isolation (a holding's own buy-and-hold return, a factor's own
            compounded return, or -- for Alpha -- the compounded,
            drifting-weight residual series). These do *not* need to sum
            to the portfolio total; they describe each contributor on its
            own.
          - "Return Attribution": each contributor's share of the
            portfolio's total *compounded* return (via Carino smoothing),
            summing exactly to "Portfolio" TotalReturn.
          - "Vol Attribution": each contributor's share of the portfolio
            return series' volatility, summing exactly to the portfolio's
            return volatility.
    """
    holding_labels = list(returns.columns)
    returns_values = returns.values
    weights_array = np.asarray(weights, dtype=float)

    weight_path, portfolio_returns = _drift_weights_and_portfolio_returns(
        returns_values, weights_array
    )
    portfolio_total_return = np.prod(1 + portfolio_returns) - 1

    if factor_returns is None:
        labels = holding_labels
        # Each holding's per-period contribution to the portfolio return.
        contributions = weight_path * returns_values
        # Each holding's own standalone compounded return.
        total_returns = (1 + returns).prod(axis=0).values - 1
    else:
        factor_labels = list(factor_returns.columns)
        labels = factor_labels + ["Alpha"]

        beta_matrix = betas.loc[holding_labels, factor_labels].values
        factor_returns_values = factor_returns.values

        predicted = factor_returns_values @ beta_matrix.T
        residuals = returns_values - predicted

        # Portfolio's exposure to each factor each period, from the same
        # drifting holding weights used for the portfolio return itself.
        portfolio_beta_path = weight_path @ beta_matrix
        factor_contributions = portfolio_beta_path * factor_returns_values
        alpha_contribution = np.sum(weight_path * residuals, axis=1)

        contributions = np.column_stack([factor_contributions, alpha_contribution])
        total_returns = np.concatenate(
            [
                (1 + factor_returns).prod(axis=0).values - 1,
                [np.prod(1 + alpha_contribution) - 1],
            ]
        )

    return_attribution_values = _carino_attribution(contributions, portfolio_returns)
    vol_attribution_values = _vol_attribution(contributions, portfolio_returns)

    result = pd.DataFrame(
        {
            "TotalReturn": list(total_returns) + [portfolio_total_return],
            "Return Attribution": list(return_attribution_values) + [portfolio_total_return],
            "Vol Attribution": list(vol_attribution_values)
            + [portfolio_returns.std(ddof=1)],
        },
        index=labels + ["Portfolio"],
    ).T
    result.index.name = "Value"
    return result
