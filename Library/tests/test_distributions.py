"""
Tests for distributions.py, based on Tests.xlsx sections 7.1 - 7.6.

Note: the fitted parameters below (especially for `fit_t_regression` and
the NIG fits) come from numerical optimization, so they can differ from
the reference values in the last few digits depending on the optimizer's
convergence path. A relative tolerance of 1e-4 comfortably separates
"converged to the same optimum" from an actual bug, while still being far
tighter than the fitted values themselves.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from distributions import (
    fit_normal_distribution,
    fit_t_distribution,
    fit_t_regression,
    t_distribution_aicc,
    fit_nig_distribution,
)

RTOL = 1e-4


def load_csv(name):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name))


def test_fit_normal_distribution():
    # Test 7.1
    data = load_csv("test7_1.csv")["x1"]
    expected = load_csv("testout7_1.csv").iloc[0]

    result = fit_normal_distribution(data)

    assert np.isclose(result["mu"], expected["mu"], rtol=RTOL)
    assert np.isclose(result["sigma"], expected["sigma"], rtol=RTOL)


def test_fit_t_distribution():
    # Test 7.2
    data = load_csv("test7_2.csv")["x1"]
    expected = load_csv("testout7_2.csv").iloc[0]

    result = fit_t_distribution(data)

    assert np.isclose(result["mu"], expected["mu"], rtol=RTOL)
    assert np.isclose(result["sigma"], expected["sigma"], rtol=RTOL)
    assert np.isclose(result["nu"], expected["nu"], rtol=RTOL)


def test_fit_t_regression():
    # Test 7.3
    data = load_csv("test7_3.csv")
    y = data["y"]
    x = data[["x1", "x2", "x3"]]
    expected = load_csv("testout7_3.csv").iloc[0]

    result = fit_t_regression(y, x)

    for key in ["mu", "sigma", "nu", "Alpha", "B1", "B2", "B3"]:
        assert np.isclose(result[key], expected[key], rtol=RTOL, atol=1e-8), key


def test_t_distribution_aicc():
    # Test 7.4
    data = load_csv("test7_2.csv")["x1"]
    expected = load_csv("testout7_4.csv").iloc[0]["AICC"]

    result = t_distribution_aicc(data)

    assert np.isclose(result, expected, rtol=RTOL)


def test_fit_nig_distribution_moments():
    # Test 7.5
    data = load_csv("test7_5.csv")["x1"]
    expected = load_csv("testout7_5.csv").iloc[0]

    result = fit_nig_distribution(data, method="moments")

    for key in ["mu", "alpha", "beta", "delta"]:
        assert np.isclose(result[key], expected[key], rtol=RTOL), key


def test_fit_nig_distribution_mle():
    # Test 7.6
    data = load_csv("test7_5.csv")["x1"]
    expected = load_csv("testout7_6.csv").iloc[0]

    result = fit_nig_distribution(data, method="mle")

    for key in ["mu", "alpha", "beta", "delta"]:
        assert np.isclose(result[key], expected[key], rtol=RTOL), key


def test_fit_nig_distribution_invalid_method():
    data = load_csv("test7_5.csv")["x1"]
    with pytest.raises(ValueError):
        fit_nig_distribution(data, method="not_a_real_method")


# --- output="distribution"/"both": the frozen scipy distribution must match
# --- the fitted parameters (checked against each distribution's own formulas).


def test_fit_normal_distribution_frozen_output():
    data = load_csv("test7_1.csv")["x1"]
    params, dist = fit_normal_distribution(data, output="both")

    assert dist.dist.name == "norm"
    assert np.isclose(dist.mean(), params["mu"])
    assert np.isclose(dist.std(), params["sigma"])
    assert np.isclose(fit_normal_distribution(data, output="distribution").std(), params["sigma"])


def test_fit_t_distribution_frozen_output():
    data = load_csv("test7_2.csv")["x1"]
    params, dist = fit_t_distribution(data, output="both")

    assert dist.dist.name == "t"
    assert np.isclose(dist.median(), params["mu"])
    # sigma is the scale, so the std is sigma * sqrt(nu / (nu - 2)).
    nu = params["nu"]
    assert np.isclose(dist.std(), params["sigma"] * np.sqrt(nu / (nu - 2)))


def test_fit_t_regression_frozen_output_is_error_distribution():
    data = load_csv("test7_3.csv")
    params, dist = fit_t_regression(data["y"], data.drop(columns=["y"]), output="both")

    assert dist.dist.name == "t"
    assert np.isclose(dist.median(), 0.0)
    assert np.isclose(dist.kwds["scale"], params["sigma"])
    assert np.isclose(dist.kwds["df"], params["nu"])


@pytest.mark.parametrize("method", ["moments", "mle"])
def test_fit_nig_distribution_frozen_output(method):
    data = load_csv("test7_5.csv")["x1"]
    params, dist = fit_nig_distribution(data, method=method, output="both")

    assert dist.dist.name == "norminvgauss"
    # NIG mean = mu + delta * beta / gamma, gamma = sqrt(alpha^2 - beta^2):
    # confirms the conversion to scipy's (a, b, loc, scale) is right.
    gamma = np.sqrt(params["alpha"] ** 2 - params["beta"] ** 2)
    assert np.isclose(dist.mean(), params["mu"] + params["delta"] * params["beta"] / gamma)
    if method == "moments":
        # Method of moments matches the sample's mean and variance exactly.
        assert np.isclose(dist.mean(), data.mean())
        assert np.isclose(dist.var(), data.var(ddof=1))


def test_fit_invalid_output():
    data = load_csv("test7_1.csv")["x1"]
    for fit in (fit_normal_distribution, fit_t_distribution, fit_nig_distribution):
        with pytest.raises(ValueError):
            fit(data, output="frozen")
