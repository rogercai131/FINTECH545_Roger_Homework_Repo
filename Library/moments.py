"""
Descriptive statistics: the first four moments of a data set.

Covers: mean, variance, skewness, and kurtosis of one or more data series
(e.g. FINTECH545 Assignment 1, Problem 1).
"""

import numpy as np
import pandas as pd

from utils import handle_missing_data


def first_four_moments(
    data,
    method: str = "unbiased",
    kurtosis_type: str = "excess",
    missing_strategy: str = "drop_row",
):
    """
    Calculate the first four moments of a data set: mean, variance,
    skewness, and kurtosis.

    Definitions (with n observations x_i, sample mean m, and central
    moments m_k = (1/n) * sum((x_i - m)^k)):
      1. Mean      = (1/n) * sum(x_i)
      2. Variance  = sigma^2
      3. Skewness  = m_3 / m_2^(3/2)        (asymmetry; 0 for a normal)
      4. Kurtosis  = m_4 / m_2^2            (tail heaviness; 3 for a normal,
                                             or 0 when reported as "excess")

    Expected input
    ---------------
    data : pd.Series, pd.DataFrame, or array-like
        The observations. A Series / 1-D array is treated as a single data
        set; a DataFrame is treated as one data set per column, and
        non-numeric columns (e.g. a "Date" column) are ignored.
    method : str, default "unbiased"
        Which estimator to use for variance, skewness, and kurtosis (the
        mean is the same either way):
          - "unbiased": sample estimators with small-sample corrections —
            variance divides by (n - 1), skewness is the adjusted
            Fisher-Pearson G1, and kurtosis is the adjusted G2. Matches
            pandas' `.var()`, `.skew()`, `.kurt()` and Excel's VAR.S /
            SKEW / KURT. Needs n >= 4.
          - "biased": plain population formulas from the definitions
            above (variance divides by n, no correction). Matches
            `scipy.stats.skew/kurtosis(bias=True)` and `np.var`.
    kurtosis_type : str, default "excess"
        How to report kurtosis:
          - "excess": kurtosis minus 3, so a normal distribution scores 0
            (the convention used by pandas, scipy, and Excel).
          - "raw": the plain fourth standardized moment, so a normal
            distribution scores 3.
    missing_strategy : str, default "drop_row"
        How to handle NaNs when `data` is a DataFrame:
          - "drop_row": drop any row with a missing value in any column, so
            every column's moments use the same set of rows.
          - "pairwise": each column independently drops only its own
            missing values (keeps as much data as possible per column).
        For a single series the two options are equivalent.

    Output
    ------
    pd.Series or pd.DataFrame
        For a single series: a Series indexed by
        ["Mean", "Variance", "Skewness", "Kurtosis"].
        For a DataFrame: a DataFrame with those four rows and one column
        per input column.

    Raises
    ------
    ValueError
        If `method`, `kurtosis_type`, or `missing_strategy` is not a
        supported option, or a series has too few observations.
    """
    if method not in ("unbiased", "biased"):
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: 'unbiased', 'biased'."
        )
    if kurtosis_type not in ("excess", "raw"):
        raise ValueError(
            f"Unsupported kurtosis_type '{kurtosis_type}'. Expected one of: 'excess', 'raw'."
        )

    # Single series in -> single Series out.
    if not isinstance(data, pd.DataFrame):
        series = pd.Series(np.asarray(data, dtype=float).ravel())
        return _moments_of_series(series.dropna().to_numpy(), method, kurtosis_type)

    numeric = data.select_dtypes(include="number")
    numeric = handle_missing_data(numeric, missing_strategy)

    # Compute each column on its own non-missing values (after "drop_row"
    # there are none left, so this only matters for "pairwise").
    results = {
        col: _moments_of_series(numeric[col].dropna().to_numpy(), method, kurtosis_type)
        for col in numeric.columns
    }
    return pd.DataFrame(results)


def _moments_of_series(x: np.ndarray, method: str, kurtosis_type: str) -> pd.Series:
    """Compute the four moments of one 1-D numpy array (NaNs already removed)."""
    n = len(x)
    min_n = 4 if method == "unbiased" else 2
    if n < min_n:
        raise ValueError(
            f"Need at least {min_n} observations for method='{method}', got {n}."
        )

    mean = x.mean()
    deviations = x - mean

    # Population (biased) central moments m_2, m_3, m_4.
    m2 = np.mean(deviations ** 2)
    m3 = np.mean(deviations ** 3)
    m4 = np.mean(deviations ** 4)

    # Biased standardized moments, straight from the definitions.
    skew = m3 / m2 ** 1.5
    excess_kurt = m4 / m2 ** 2 - 3.0

    if method == "biased":
        variance = m2
    else:
        # Bessel's correction for variance.
        variance = m2 * n / (n - 1)
        # Adjusted Fisher-Pearson skewness (G1).
        skew = skew * np.sqrt(n * (n - 1)) / (n - 2)
        # Adjusted excess kurtosis (G2).
        excess_kurt = ((n + 1) * excess_kurt + 6.0) * (n - 1) / ((n - 2) * (n - 3))

    kurtosis = excess_kurt if kurtosis_type == "excess" else excess_kurt + 3.0

    return pd.Series(
        [mean, variance, skew, kurtosis],
        index=["Mean", "Variance", "Skewness", "Kurtosis"],
    )
