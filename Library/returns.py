"""
Converting between prices and returns under the three standard return
models:
  - Classical Brownian:  P_t = P_(t-1) + r_t          (r_t = P_t - P_(t-1))
  - Arithmetic:          P_t = P_(t-1) * (1 + r_t)    (r_t = P_t / P_(t-1) - 1)
  - Geometric Brownian:  P_t = P_(t-1) * exp(r_t)     (r_t = ln(P_t / P_(t-1)))

Covers: FINTECH545 Tests 6.1 - 6.2 (arithmetic returns and log returns),
plus the reverse direction (prices from returns) and the expected mean /
standard deviation of the next price under each model when r_t ~ N(mu, sigma^2).
"""

import numpy as np
import pandas as pd


def return_calculate(
    prices: pd.DataFrame,
    method: str = "arithmetic",
    date_column: str = "Date",
) -> pd.DataFrame:
    """
    Convert a table of prices into a table of period-over-period returns.

    Expected input
    ---------------
    prices : pd.DataFrame
        Rows are time periods in chronological order (oldest first), one
        column per asset holding its price level. May optionally include a
        non-numeric date/label column (see `date_column`).
    method : str, default "arithmetic"
        Which type of return to compute:
          - "classical": price change, P_t - P_(t-1) (Classical Brownian
            motion model; in price units, not a percentage).
          - "arithmetic": simple return, P_t / P_(t-1) - 1.
          - "log": log return, ln(P_t / P_(t-1)) (Geometric Brownian
            motion model).
          - "geometric": same as "log".
    date_column : str, default "Date"
        Name of a column to treat as a row label rather than a price
        series (excluded from the return calculation, then re-attached to
        the output). Pass `None` if `prices` has no such column and every
        column should be treated as a price series.

    Output
    ------
    pd.DataFrame
        One row shorter than `prices` (the first period has no prior price
        to compare against), with the same price columns now holding
        returns. If `date_column` was given, it is included in the output,
        aligned to the returns (i.e. the date of the *later* price in each
        pair).

    Raises
    ------
    ValueError
        If `method` is not "classical"/"arithmetic"/"log"/"geometric".
    """
    if method not in ("classical", "arithmetic", "log", "geometric"):
        raise ValueError(
            f"Unsupported method '{method}'. "
            "Expected one of: 'classical', 'arithmetic', 'log', 'geometric'."
        )

    if date_column is not None and date_column in prices.columns:
        dates = prices[date_column]
        price_levels = prices.drop(columns=[date_column])
    else:
        dates = None
        price_levels = prices

    if method == "classical":
        returns = price_levels.diff()
    elif method == "arithmetic":
        returns = price_levels.pct_change()
    else:
        returns = np.log(price_levels / price_levels.shift(1))

    # First row is always NaN (no prior price to compare to) — drop it.
    returns = returns.iloc[1:].reset_index(drop=True)

    if dates is not None:
        date_series = dates.iloc[1:].reset_index(drop=True).rename(date_column)
        returns = pd.concat([date_series, returns], axis=1)

    return returns


_PRICE_METHODS = ("classical", "arithmetic", "geometric")


def _check_price_method(method: str) -> None:
    if method not in _PRICE_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. "
            "Expected one of: 'classical', 'arithmetic', 'geometric'."
        )


def price_from_returns(
    returns,
    start_price,
    method: str = "classical",
    horizon: str = "one_step",
):
    """
    Turn returns into prices under one of the three return models — the
    reverse of `return_calculate`.

    Expected input
    ---------------
    returns : pd.Series, pd.DataFrame, or array-like
        The returns r_t. A DataFrame is treated as one return series per
        column (e.g. one column per asset); non-numeric columns are ignored.
    start_price : float or pd.Series
        The starting price P_(t-1). For a DataFrame, either one number for
        every column or a Series indexed by column name.
    method : str, default "classical"
        Which return model links P_(t-1), r_t and P_t:
          - "classical": Classical Brownian motion, P_t = P_(t-1) + r_t.
            Returns are price changes; prices can go negative.
          - "arithmetic": P_t = P_(t-1) * (1 + r_t). Returns are simple
            percentage returns; price goes negative if r_t < -1.
          - "geometric": Geometric Brownian motion, P_t = P_(t-1) * exp(r_t).
            Returns are log returns; prices always stay positive.
    horizon : str, default "one_step"
        How the returns are applied:
          - "one_step": every return is a separate scenario for the *same*
            next period, so each output price is P_(t-1) transformed by its
            own r_t (use this to get the distribution of P_t from simulated
            returns).
          - "path": the returns are consecutive periods, compounded in
            order, so each output price is the price after that many
            periods (use this to rebuild a price path from a return history).

    Output
    ------
    pd.Series or pd.DataFrame
        Prices, same shape as the (numeric part of the) returns: a Series
        for single-series input, a DataFrame (same columns) for DataFrame
        input. The starting price itself is not included.

    Raises
    ------
    ValueError
        If `method` or `horizon` is not a supported option.
    """
    _check_price_method(method)
    if horizon not in ("one_step", "path"):
        raise ValueError(
            f"Unsupported horizon '{horizon}'. Expected one of: 'one_step', 'path'."
        )

    if isinstance(returns, pd.DataFrame):
        r = returns.select_dtypes(include="number")
        p0 = start_price.reindex(r.columns) if isinstance(start_price, pd.Series) else start_price
    else:
        r = returns if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns, dtype=float).ravel())
        p0 = start_price

    if horizon == "one_step":
        if method == "classical":
            return r + p0
        if method == "arithmetic":
            return (1 + r) * p0
        return np.exp(r) * p0

    # "path": compound period by period.
    if method == "classical":
        return r.cumsum() + p0                  # P_n = P_0 + sum(r)
    if method == "arithmetic":
        return (1 + r).cumprod() * p0           # P_n = P_0 * prod(1 + r)
    return np.exp(r.cumsum()) * p0              # P_n = P_0 * exp(sum(r))


def expected_price_moments(
    start_price: float,
    sigma: float,
    mu: float = 0.0,
    method: str = "classical",
) -> pd.Series:
    """
    Theoretical mean and standard deviation of the next price P_t, given
    P_(t-1) and a normally distributed return r_t ~ N(mu, sigma^2), under
    one of the three return models. Handy for checking a simulation done
    with `price_from_returns(..., horizon="one_step")`.

    Formulas
    --------
      - "classical":  P_t = P_(t-1) + r_t
            E[P_t]  = P_(t-1) + mu
            SD[P_t] = sigma
      - "arithmetic": P_t = P_(t-1) * (1 + r_t)
            E[P_t]  = P_(t-1) * (1 + mu)
            SD[P_t] = P_(t-1) * sigma
      - "geometric":  P_t = P_(t-1) * exp(r_t)   (P_t is lognormal)
            E[P_t]  = P_(t-1) * exp(mu + sigma^2 / 2)
            SD[P_t] = P_(t-1) * sqrt((exp(sigma^2) - 1) * exp(2*mu + sigma^2))

    Note the geometric mean is *above* P_(t-1) even when mu = 0 — the
    convexity of exp() pushes the average up.

    Expected input
    ---------------
    start_price : float
        P_(t-1).
    sigma : float
        Standard deviation of the return r_t (must be >= 0).
    mu : float, default 0.0
        Mean of the return r_t.
    method : str, default "classical"
        "classical", "arithmetic", or "geometric" (see Formulas).

    Output
    ------
    pd.Series
        Indexed by ["Mean", "Std"].

    Raises
    ------
    ValueError
        If `method` is not supported or `sigma` is negative.
    """
    _check_price_method(method)
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")

    if method == "classical":
        mean, std = start_price + mu, sigma
    elif method == "arithmetic":
        mean, std = start_price * (1 + mu), start_price * sigma
    else:
        mean = start_price * np.exp(mu + sigma ** 2 / 2)
        std = start_price * np.sqrt((np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2))

    return pd.Series([mean, std], index=["Mean", "Std"])
