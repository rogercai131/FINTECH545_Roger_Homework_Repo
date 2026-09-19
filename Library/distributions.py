"""
Fitting probability distributions to data.

Covers: FINTECH545 Tests 7.1 - 7.6 (fitting a Normal distribution, fitting
a Student's t distribution by MLE, a linear regression with t-distributed
errors, AICc of a fitted t distribution, and fitting a Normal Inverse
Gaussian (NIG) distribution by both the method of moments and MLE).
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize

from utils import aicc as _aicc


def fit_normal_distribution(data: pd.Series) -> pd.Series:
    """
    Fit a Normal distribution to a 1-D sample by its sample mean and
    (unbiased, ddof=1) sample standard deviation.

    Expected input
    ---------------
    data : pd.Series (or array-like)
        A 1-D sample of observations.

    Output
    ------
    pd.Series
        Index ["mu", "sigma"] holding the fitted mean and standard
        deviation.
    """
    values = np.asarray(data)
    mu = values.mean()
    sigma = values.std(ddof=1)
    return pd.Series({"mu": mu, "sigma": sigma})


def fit_t_distribution(data: pd.Series) -> pd.Series:
    """
    Fit a (location-scale) Student's t distribution to a 1-D sample by
    maximum likelihood.

    Expected input
    ---------------
    data : pd.Series (or array-like)
        A 1-D sample of observations.

    Output
    ------
    pd.Series
        Index ["mu", "sigma", "nu"] holding the fitted location, scale,
        and degrees of freedom.
    """
    values = np.asarray(data)
    nu, mu, sigma = stats.t.fit(values)
    return pd.Series({"mu": mu, "sigma": sigma, "nu": nu})


def fit_t_regression(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    """
    Fit a linear regression y = Alpha + B1*x1 + B2*x2 + ... + error, where
    the error term is assumed to follow a (zero-centered) Student's t
    distribution, via maximum likelihood.

    This differs from ordinary least squares in that it estimates the
    error distribution's degrees of freedom `nu` alongside the
    regression coefficients, so it can down-weight outliers relative to
    OLS when the errors are heavy-tailed.

    Expected input
    ---------------
    y : pd.Series (or array-like)
        The response variable, length n.
    x : pd.DataFrame (or array-like)
        The predictor variables, shape (n, k). Column names/order don't
        matter — the output always names coefficients B1..Bk in column
        order.

    Output
    ------
    pd.Series
        Index ["mu", "sigma", "nu", "Alpha", "B1", ..., "Bk"]. `mu` is
        always 0.0 (the error term is centered; any nonzero mean is
        absorbed into `Alpha`).
    """
    y_values = np.asarray(y)
    x_values = np.asarray(x)
    if x_values.ndim == 1:
        x_values = x_values.reshape(-1, 1)
    n, k = x_values.shape

    # Ordinary least squares gives a good starting point for the MLE search.
    design = np.column_stack([np.ones(n), x_values])
    ols_beta, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    ols_resid = y_values - design @ ols_beta
    sigma_init = ols_resid.std(ddof=1)

    def neg_log_likelihood(params):
        alpha = params[0]
        betas = params[1 : 1 + k]
        # sigma and nu are optimized in log-space so the optimizer can't
        # push them to invalid (<=0, or <=2 for nu) values.
        sigma = np.exp(params[1 + k])
        nu = 2.0 + np.exp(params[2 + k])
        fitted = alpha + x_values @ betas
        residuals = y_values - fitted
        return -np.sum(stats.t.logpdf(residuals, df=nu, loc=0.0, scale=sigma))

    initial_guess = np.concatenate(
        [ols_beta, [np.log(sigma_init), np.log(3.0)]]
    )
    result = optimize.minimize(
        neg_log_likelihood,
        initial_guess,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000},
    )

    alpha = result.x[0]
    betas = result.x[1 : 1 + k]
    sigma = np.exp(result.x[1 + k])
    nu = 2.0 + np.exp(result.x[2 + k])

    output = {"mu": 0.0, "sigma": sigma, "nu": nu, "Alpha": alpha}
    for i, beta in enumerate(betas, start=1):
        output[f"B{i}"] = beta
    return pd.Series(output)


def t_distribution_aicc(data: pd.Series, fitted: pd.Series = None) -> float:
    """
    Corrected Akaike Information Criterion (AICc) of a Student's t
    distribution fit to a 1-D sample.

    Expected input
    ---------------
    data : pd.Series (or array-like)
        The 1-D sample the distribution was (or will be) fit to.
    fitted : pd.Series, optional
        A previous result of `fit_t_distribution(data)` (with "mu",
        "sigma", "nu"). If omitted, the distribution is fit here.

    Output
    ------
    float
        The AICc value (lower is a better fit, penalized for the 3 fitted
        parameters and small sample size).
    """
    values = np.asarray(data)
    if fitted is None:
        fitted = fit_t_distribution(values)

    log_likelihood = np.sum(
        stats.t.logpdf(values, df=fitted["nu"], loc=fitted["mu"], scale=fitted["sigma"])
    )
    return _aicc(log_likelihood, num_params=3, num_obs=len(values))


def fit_nig_distribution(data: pd.Series, method: str = "moments") -> pd.Series:
    """
    Fit a Normal Inverse Gaussian (NIG) distribution to a 1-D sample.

    Expected input
    ---------------
    data : pd.Series (or array-like)
        A 1-D sample of observations.
    method : str, default "moments"
        Which fitting approach to use:
          - "moments": method of moments — solves for the 4 NIG
            parameters directly from the sample mean, variance, skewness,
            and (excess) kurtosis. Fast, closed-form, but less efficient
            than MLE.
          - "mle": maximum likelihood, via `scipy.stats.norminvgauss.fit`.
            More accurate but slower (numerical optimization).

    Output
    ------
    pd.Series
        Index ["mu", "alpha", "beta", "delta"] holding the fitted NIG
        location, tail-heaviness, asymmetry, and scale parameters.

    Raises
    ------
    ValueError
        If `method` is not "moments"/"mle".
    """
    if method not in ("moments", "mle"):
        raise ValueError(f"Unsupported method '{method}'. Expected one of: 'moments', 'mle'.")

    values = np.asarray(data)

    if method == "mle":
        # scipy's norminvgauss(a, b, loc, scale) is a *scaled* parameterization:
        # alpha = a / scale, beta = b / scale, mu = loc, delta = scale.
        a, b, loc, scale = stats.norminvgauss.fit(values)
        return pd.Series(
            {"mu": loc, "alpha": a / scale, "beta": b / scale, "delta": scale}
        )

    # Method of moments: invert the NIG mean/variance/skewness/kurtosis
    # formulas (Barndorff-Nielsen) to recover mu, alpha, beta, delta.
    mean = values.mean()
    variance = values.var(ddof=1)
    skewness = stats.skew(values)
    excess_kurtosis = stats.kurtosis(values)

    # rho = beta / alpha (asymmetry ratio, in (-1, 1)) solves from the
    # ratio of squared skewness to kurtosis.
    ratio = skewness**2 / excess_kurtosis
    rho_squared = ratio / (3 - 4 * ratio)
    rho = np.sign(skewness) * np.sqrt(rho_squared)

    # delta * gamma (gamma = sqrt(alpha^2 - beta^2)) solves from kurtosis.
    delta_gamma = 3 * (1 + 4 * rho_squared) / excess_kurtosis

    alpha = np.sqrt(delta_gamma / (variance * (1 - rho_squared) ** 2))
    beta = rho * alpha
    gamma = alpha * np.sqrt(1 - rho_squared)
    delta = delta_gamma / gamma
    mu = mean - delta * beta / gamma

    return pd.Series({"mu": mu, "alpha": alpha, "beta": beta, "delta": delta})
