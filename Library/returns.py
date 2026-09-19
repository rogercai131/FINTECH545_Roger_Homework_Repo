"""
Converting a price series into a return series.

Covers: FINTECH545 Tests 6.1 - 6.2 (arithmetic returns and log returns).
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
          - "arithmetic": simple return, P_t / P_(t-1) - 1.
          - "log": log return, ln(P_t / P_(t-1)).
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
        If `method` is not "arithmetic"/"log".
    """
    if method not in ("arithmetic", "log"):
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: 'arithmetic', 'log'."
        )

    if date_column is not None and date_column in prices.columns:
        dates = prices[date_column]
        price_levels = prices.drop(columns=[date_column])
    else:
        dates = None
        price_levels = prices

    if method == "arithmetic":
        returns = price_levels.pct_change()
    else:
        returns = np.log(price_levels / price_levels.shift(1))

    # First row is always NaN (no prior price to compare to) — drop it.
    returns = returns.iloc[1:].reset_index(drop=True)

    if dates is not None:
        date_series = dates.iloc[1:].reset_index(drop=True).rename(date_column)
        returns = pd.concat([date_series, returns], axis=1)

    return returns
