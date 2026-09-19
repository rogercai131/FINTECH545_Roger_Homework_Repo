"""
Tests for portfolio_optimization.py, based on Tests.xlsx sections 10.1 - 10.4.

These optimizations are deterministic (no randomness), but the reference
values were themselves produced by a numerical optimizer, so a couple of
weights that are "supposed to be" exactly 0 show up as tiny +/- 1e-8
noise in the reference CSVs. An atol of 1e-5 absorbs that noise while
still being far tighter than the actual portfolio weights.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from portfolio_optimization import calculate_risk_parity_weights, calculate_max_sharpe_weights

ATOL = 1e-5


def load_csv(name):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name))


def test_risk_parity_equal_budget():
    # Test 10.1
    cov = load_csv("test5_2.csv")
    expected = load_csv("testout10_1.csv")["W"].values

    result = calculate_risk_parity_weights(cov)

    assert np.allclose(result.values, expected, atol=ATOL)
    assert np.isclose(result.sum(), 1.0)


def test_risk_parity_custom_budget():
    # Test 10.2: half risk budget on x5
    cov = load_csv("test5_2.csv")
    expected = load_csv("testout10_2.csv")["W"].values

    result = calculate_risk_parity_weights(cov, risk_budgets=[1, 1, 1, 1, 0.5])

    assert np.allclose(result.values, expected, atol=ATOL)
    assert np.isclose(result.sum(), 1.0)


def test_max_sharpe_long_only():
    # Test 10.3
    cov = load_csv("test5_2.csv")
    means = load_csv("test10_3_means.csv")["Mean"]
    expected = load_csv("testout10_3.csv")["W"].values

    result = calculate_max_sharpe_weights(cov, means, risk_free_rate=0.04, lower_bound=0.0, upper_bound=1.0)

    assert np.allclose(result.values, expected, atol=ATOL)
    assert np.isclose(result.sum(), 1.0)
    assert (result.values >= -ATOL).all()


def test_max_sharpe_bounded():
    # Test 10.4: 0.1 <= w <= 0.5
    cov = load_csv("test5_2.csv")
    means = load_csv("test10_3_means.csv")["Mean"]
    expected = load_csv("testout10_4.csv")["W"].values

    result = calculate_max_sharpe_weights(cov, means, risk_free_rate=0.04, lower_bound=0.1, upper_bound=0.5)

    assert np.allclose(result.values, expected, atol=ATOL)
    assert np.isclose(result.sum(), 1.0)
    assert (result.values >= 0.1 - ATOL).all()
    assert (result.values <= 0.5 + ATOL).all()
