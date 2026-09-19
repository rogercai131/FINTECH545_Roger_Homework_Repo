"""
Fitting a multivariate Student's t distribution to a data set.

Covers: FINTECH545 Tests 13.2 - 13.4 (the fitted mean vector, scale
matrix, and degrees of freedom / log-likelihood of a multivariate t fit).

Fitting approach: the mean vector is simply the sample mean of each
column. The scale matrix is built from each variable's own sample
standard deviation combined with the (PSD-repaired) Kendall's-tau-based
correlation matrix from `covariance.kendall_correlation_matrix` -- i.e.
the same correlation estimate used elsewhere in this library for
robustness to outliers/nonlinearity -- scaled by (nu - 2) / nu so that the
*covariance* implied by the fitted t distribution matches the sample
covariance. The only free parameter is then the degrees of freedom `nu`,
found by a 1-D profile likelihood search (this keeps the fit well-defined
and fast, rather than a full joint numerical MLE over the mean vector,
every entry of the scale matrix, and nu at once, which is both slower and
prone to landing in different local optima depending on the optimizer).
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

from covariance import kendall_correlation_matrix


def _multivariate_t_log_likelihood(data: np.ndarray, mu: np.ndarray, scale: np.ndarray, nu: float) -> float:
    """Log-likelihood of a multivariate t distribution (location `mu`,
    scale matrix `scale`, degrees of freedom `nu`) over each row of `data`."""
    num_obs, num_vars = data.shape
    scale_inv = np.linalg.inv(scale)
    _, log_det_scale = np.linalg.slogdet(scale)

    diffs = data - mu
    mahalanobis_sq = np.einsum("ij,jk,ik->i", diffs, scale_inv, diffs)

    log_norm_const = (
        gammaln((nu + num_vars) / 2)
        - gammaln(nu / 2)
        - (num_vars / 2) * np.log(nu * np.pi)
        - 0.5 * log_det_scale
    )
    return np.sum(log_norm_const - ((nu + num_vars) / 2) * np.log(1 + mahalanobis_sq / nu))


def fit_multivariate_t(data: pd.DataFrame) -> dict:
    """
    Fit a multivariate Student's t distribution to a data set.

    Expected input
    ---------------
    data : pd.DataFrame
        Rows are observations, columns are variables. Should not contain
        NaN values.

    Output
    ------
    dict with keys:
      - "mu": pd.Series, the fitted mean vector (indexed by `data`'s columns).
      - "scale": pd.DataFrame, the fitted scale matrix S. Note this is
        *not* the covariance matrix of the fitted distribution -- for a
        multivariate t with `nu` degrees of freedom, the implied
        covariance is `nu / (nu - 2) * S` (only defined for nu > 2).
      - "nu": float, the fitted degrees of freedom.
      - "log_likelihood": float, the log-likelihood at the fitted parameters.
    """
    labels = data.columns
    values = data.values
    mu = values.mean(axis=0)

    correlation = kendall_correlation_matrix(data).values
    std = values.std(axis=0, ddof=1)
    implied_covariance = np.outer(std, std) * correlation

    def scale_for(nu):
        return implied_covariance * (nu - 2) / nu

    def negative_profile_log_likelihood(nu):
        return -_multivariate_t_log_likelihood(values, mu, scale_for(nu), nu)

    result = minimize_scalar(
        negative_profile_log_likelihood, bounds=(2.0001, 1000), method="bounded",
        options={"xatol": 1e-10},
    )
    nu = result.x
    scale = scale_for(nu)
    log_likelihood = -result.fun

    return {
        "mu": pd.Series(mu, index=labels),
        "scale": pd.DataFrame(scale, index=labels, columns=labels),
        "nu": nu,
        "log_likelihood": log_likelihood,
    }
