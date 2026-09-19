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
