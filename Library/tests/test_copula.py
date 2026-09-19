"""
Tests for copula.py, based on Tests.xlsx sections 13.5 - 13.9.

As in test_multivariate_t.py, the fitted copula nu/log-likelihood values
involve a 1-D numerical optimizer, so tests allow a small amount of
optimizer-convergence slack rather than requiring a bit-exact match.
Test 13.9 is a Monte Carlo simulation (like test_simulation.py /
test_portfolio_risk.py), so it's checked for closeness rather than an
exact match to its reference CSV.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from copula import (
    gaussian_copula_log_likelihood,
    fit_t_copula,
    compare_copulas,
    t_copula_tail_dependence,
    simulate_t_copula_portfolio_var_es,
)

RTOL = 1e-3


def load_csv(name, **kwargs):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name), **kwargs)


def test_gaussian_copula_log_likelihood():
    # Test 13.5
    data = load_csv("test13_returns.csv")
    expected_ll = load_csv("testout13_5.csv").iloc[0]["ll"]

    result = gaussian_copula_log_likelihood(data)

    assert np.isclose(result, expected_ll, rtol=1e-4)


def test_fit_t_copula():
    # Test 13.6
    data = load_csv("test13_returns.csv")
    expected_nu, expected_ll = load_csv("testout13_6.csv").iloc[0][["nu", "ll"]]

    result = fit_t_copula(data)

    assert np.isclose(result["nu"], expected_nu, rtol=RTOL)
    assert np.isclose(result["log_likelihood"], expected_ll, rtol=1e-5)


def test_compare_copulas():
    # Test 13.7
    data = load_csv("test13_returns.csv")
    expected = load_csv("testout13_7.csv").set_index("Copula")

    result = compare_copulas(data)

    assert list(result.index) == ["Gaussian", "T"]
    for copula in ["Gaussian", "T"]:
        for col in ["LL", "K", "AICC", "BIC"]:
            assert np.isclose(result.loc[copula, col], expected.loc[copula, col], rtol=1e-3), (
                copula,
                col,
            )


def test_t_copula_tail_dependence():
    # Test 13.8
    data = load_csv("test13_returns.csv")
    t_fit = fit_t_copula(data)
    expected = load_csv("testout13_8.csv")

    result = t_copula_tail_dependence(t_fit["correlation"], t_fit["nu"])

    assert np.allclose(result[["I", "J", "Rho", "Lambda"]].values, expected.values, rtol=1e-3)


def test_simulate_t_copula_portfolio_var_es():
    # Test 13.9
    portfolio = load_csv("test13_portfolio.csv")
    returns = load_csv("test13_returns.csv")
    expected = load_csv("testout13_9.csv").set_index("Stock")

    result = simulate_t_copula_portfolio_var_es(portfolio, returns, seed=1).set_index("Stock")

    assert list(result.columns) == ["VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]
    for stock in list(expected.index):
        for col in expected.columns:
            assert np.isclose(result.loc[stock, col], expected.loc[stock, col], rtol=0.03), (
                stock,
                col,
            )


def test_gaussian_copula_zero_tail_dependence_reference():
    # Sanity check: a t copula with very large nu should behave like a
    # Gaussian copula, whose tail dependence is exactly 0.
    correlation = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["a", "b"], index=["a", "b"])
    result = t_copula_tail_dependence(correlation, nu=1e6)

    assert np.isclose(result.loc[0, "Lambda"], 0.0, atol=1e-3)
