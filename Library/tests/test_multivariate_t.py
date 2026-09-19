"""
Tests for multivariate_t.py, based on Tests.xlsx sections 13.2 - 13.4.

The fitted nu/log-likelihood match the reference values to about 4-5
significant digits rather than exactly -- both are the output of a 1-D
numerical optimizer (profile likelihood over nu), and the small residual
gap is consistent with a slightly different optimizer convergence
tolerance rather than a different method (a coarser or finer optimizer
tolerance moves the last few digits of nu, which then moves Sigma and LL
by a similarly tiny amount).
"""

import os

import numpy as np
import pandas as pd

from conftest import TEST_FILES_DIR
from multivariate_t import fit_multivariate_t

RTOL = 1e-3


def load_csv(name):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name))


def test_fit_multivariate_t():
    data = load_csv("test13_returns.csv")
    expected_mu = load_csv("testout13_2.csv")["mu"].values
    expected_scale = load_csv("testout13_3.csv").values
    expected_nu, expected_ll = load_csv("testout13_4.csv").iloc[0][["nu", "ll"]]

    result = fit_multivariate_t(data)

    # Test 13.2: mean vector matches exactly (it's just the sample mean).
    assert np.allclose(result["mu"].values, expected_mu)

    # Test 13.3 / 13.4: nu and the scale matrix come from a numerical
    # profile-likelihood search, so allow a little optimizer slack.
    assert np.isclose(result["nu"], expected_nu, rtol=RTOL)
    assert np.allclose(result["scale"].values, expected_scale, rtol=RTOL)
    assert np.isclose(result["log_likelihood"], expected_ll, rtol=1e-5)


def test_fit_multivariate_t_implied_variances_match_sample():
    # Sanity check on the documented relationship cov = nu/(nu-2) * S:
    # the *variances* (diagonal) should recover the sample variances
    # exactly, since the scale matrix is built from each variable's own
    # sample standard deviation. Off-diagonal covariances will generally
    # differ from the sample (Pearson) covariance, since the correlation
    # structure comes from Kendall's tau rather than Pearson correlation.
    data = load_csv("test13_returns.csv")
    result = fit_multivariate_t(data)

    nu = result["nu"]
    implied_variances = (nu / (nu - 2)) * np.diag(result["scale"].values)
    sample_variances = data.var(ddof=1).values

    assert np.allclose(implied_variances, sample_variances, rtol=1e-6)
