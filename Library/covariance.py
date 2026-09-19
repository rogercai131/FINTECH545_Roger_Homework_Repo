"""
Covariance and correlation calculations.

Covers:
  - FINTECH545 Tests 1.1 - 1.4 (missing-data covariance/correlation,
    both drop-row and pairwise strategies).
  - FINTECH545 Tests 2.1 - 2.3 (exponentially-weighted covariance/
    correlation, including mixing a different decay factor for variance
    vs. correlation).
  - FINTECH545 Test 13.1 (correlation from Kendall's tau).
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from utils import handle_missing_data
from psd_fix import fix_psd_matrix


def covariance_matrix(
    data: pd.DataFrame,
    method: str = "covariance",
    missing_strategy: str = "drop_row",
) -> pd.DataFrame:
    """
    Compute a covariance or correlation matrix from a data set, with control
    over how missing values are handled.

    Expected input
    ---------------
    data : pd.DataFrame
        Rows are observations (e.g. dates), columns are variables (e.g.
        assets/returns). May contain NaN values.
    method : str, default "covariance"
        Which matrix to compute:
          - "covariance": the covariance matrix.
          - "correlation": the correlation matrix.
    missing_strategy : str, default "drop_row"
        How to handle missing values (see `utils.handle_missing_data`):
          - "drop_row": drop any row with at least one NaN before computing
            (every column pair uses the same, fully-complete rows).
          - "pairwise": each pair of columns is computed using all rows
            where *that pair* is complete, independent of NaNs in other
            columns (pandas' native pairwise-complete-observations behavior).

    Output
    ------
    pd.DataFrame
        A square DataFrame (columns/index = `data`'s columns) holding the
        covariance or correlation matrix.

    Raises
    ------
    ValueError
        If `method` is not "covariance"/"correlation", or if
        `missing_strategy` is not a supported option.
    """
    if method not in ("covariance", "correlation"):
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: 'covariance', 'correlation'."
        )

    # For "drop_row" this removes incomplete rows up front. For "pairwise"
    # this is a no-op here, since pandas' .cov()/.corr() already compute
    # each column pair over its own complete-case rows.
    cleaned = handle_missing_data(data, missing_strategy=missing_strategy)

    if method == "covariance":
        return cleaned.cov()
    else:
        return cleaned.corr()


def _ew_weights(num_obs: int, lambda_: float) -> np.ndarray:
    """
    Build normalized exponentially-decaying weights for `num_obs`
    observations, where row 0 is the oldest observation and row
    `num_obs - 1` is the most recent (matches how the CSV test files are
    ordered: oldest observation first).

    weight[i] is proportional to lambda_ ** (num_obs - i - 1), so the most
    recent observation (i = num_obs - 1) gets the largest weight. Weights
    are normalized to sum to 1.
    """
    powers = np.arange(num_obs - 1, -1, -1)
    weights = (1 - lambda_) * (lambda_**powers)
    return weights / weights.sum()


def _ew_raw_covariance(x: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Exponentially-weighted covariance matrix of a raw numpy array `x`
    (rows = observations, columns = variables), using decay factor
    `lambda_`.

    Uses the *weighted* mean (not the simple mean) as the center point,
    consistent with a proper exponentially-weighted estimator.
    """
    num_obs = x.shape[0]
    weights = _ew_weights(num_obs, lambda_)
    weighted_mean = weights @ x
    centered = x - weighted_mean
    # Scaling each row by sqrt(weight) before the matrix product is
    # equivalent to summing weight_i * outer(centered_i, centered_i).
    scaled = centered * np.sqrt(weights)[:, None]
    return scaled.T @ scaled


def ew_covariance_matrix(
    data: pd.DataFrame,
    lambda_: float = 0.97,
    method: str = "covariance",
    lambda_corr: float = None,
) -> pd.DataFrame:
    """
    Compute an exponentially-weighted (EW) covariance or correlation matrix,
    with the option to use a *different* decay factor for the
    variance/volatility part than for the correlation part.

    Expected input
    ---------------
    data : pd.DataFrame
        Rows are observations ordered oldest-to-newest, columns are
        variables (e.g. asset returns). Should not contain NaN values.
    lambda_ : float, default 0.97
        Decay factor (between 0 and 1, exclusive) used to compute the EW
        variances. Closer to 1 means slower decay (longer memory). Also
        used for the correlation part unless `lambda_corr` is given.
    method : str, default "covariance"
        Which matrix to return:
          - "covariance": the EW covariance matrix.
          - "correlation": the EW correlation matrix.
    lambda_corr : float, optional
        If provided, the correlation structure is computed using this decay
        factor instead of `lambda_`, while the variances (diagonal /
        scaling) still use `lambda_`. This supports the common real-world
        scenario of pairing a slower/faster-decaying volatility estimate
        with a differently-decaying correlation estimate. If omitted,
        `lambda_` is used for both.

    Output
    ------
    pd.DataFrame
        A square DataFrame (columns/index = `data`'s columns) holding the
        EW covariance or correlation matrix.

    Raises
    ------
    ValueError
        If `method` is not "covariance"/"correlation".
    """
    if method not in ("covariance", "correlation"):
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: 'covariance', 'correlation'."
        )

    columns = data.columns
    x = data.values

    cov_for_variance = _ew_raw_covariance(x, lambda_)

    # Correlation is derived from its own covariance estimate, which may
    # use a different decay factor (lambda_corr) than the variances do.
    cov_for_correlation = (
        cov_for_variance if lambda_corr is None else _ew_raw_covariance(x, lambda_corr)
    )
    corr_std = np.sqrt(np.diag(cov_for_correlation))
    correlation = cov_for_correlation / np.outer(corr_std, corr_std)

    if method == "correlation":
        result = correlation
    else:
        var_std = np.sqrt(np.diag(cov_for_variance))
        result = correlation * np.outer(var_std, var_std)

    return pd.DataFrame(result, index=columns, columns=columns)


def kendall_correlation_matrix(data: pd.DataFrame, psd_fix_method: str = "near_psd") -> pd.DataFrame:
    """
    Estimate a correlation matrix from Kendall's tau rank correlation
    rather than Pearson correlation, then repair it to be a valid
    (positive semi-definite) correlation matrix.

    Kendall's tau is a rank-based measure of association, more robust to
    outliers and nonlinear-but-monotonic relationships than Pearson
    correlation. Converting it to a "linear correlation"-like number via
    rho = sin(pi * tau / 2) recovers the Pearson correlation implied by an
    elliptical (e.g. Gaussian or t) copula with that tau -- but the
    resulting matrix isn't guaranteed to be PSD, since each pairwise tau
    is estimated independently, so it's repaired before being returned.

    Expected input
    ---------------
    data : pd.DataFrame
        Rows are observations, columns are variables. Should not contain
        NaN values.
    psd_fix_method : str, default "near_psd"
        Which `psd_fix.fix_psd_matrix` method to repair the matrix with
        ("near_psd" or "higham").

    Output
    ------
    pd.DataFrame
        A square, PSD correlation matrix (columns/index = `data`'s
        columns).
    """
    columns = data.columns
    n = len(columns)
    values = data.values

    tau = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            tau_ij, _ = kendalltau(values[:, i], values[:, j])
            tau[i, j] = tau[j, i] = tau_ij

    correlation = np.sin(np.pi * tau / 2)
    np.fill_diagonal(correlation, 1.0)
    correlation_df = pd.DataFrame(correlation, index=columns, columns=columns)

    return fix_psd_matrix(correlation_df, method=psd_fix_method)
