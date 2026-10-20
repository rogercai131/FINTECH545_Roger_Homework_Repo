"""
Tests for copula.py, based on Tests.xlsx sections 13.5 - 13.9. simulate_copula
has no reference CSV; it's checked against the existing Gaussian (Test 9.1)
and t copula fits it should reproduce, plus simulated-moment checks.

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
    simulate_copula,
)
from portfolio_risk import calculate_portfolio_var_es

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


# --- simulate_copula ---


def test_simulate_copula_gaussian_matches_portfolio_risk():
    # Same marginals, correlation estimate, and seed as Test 9.1's Gaussian
    # copula => the exact same simulated returns, hence the same VaR/ES.
    portfolio = load_csv("test9_1_portfolio.csv", encoding="utf-8-sig")
    returns = load_csv("test9_1_returns.csv", encoding="utf-8-sig")
    marginals = dict(zip(portfolio["Stock"], portfolio["Distribution"]))

    sims = simulate_copula(returns, copula="gaussian", marginals=marginals, seed=1)

    expected = calculate_portfolio_var_es(portfolio, returns, seed=1).set_index("Stock")
    values = (portfolio["Holding"] * portfolio["Starting Price"]).values
    pnl = sims[portfolio["Stock"]].values * values
    total = pnl.sum(axis=1)
    assert np.isclose(-np.quantile(pnl[:, 0], 0.05), expected.loc["A", "VaR95"])
    assert np.isclose(-np.quantile(total, 0.05), expected.loc["Total", "VaR95"])


def test_simulate_copula_t_fit_matches_fit_t_copula():
    returns = load_csv("test13_returns.csv")
    sims, fit = simulate_copula(returns, copula="t", num_simulations=1000, seed=1, output="both")
    reference = fit_t_copula(returns)

    assert list(sims.columns) == list(returns.columns)
    assert sims.shape == (1000, returns.shape[1])
    assert np.isclose(fit["nu"], reference["nu"], rtol=RTOL)
    assert np.allclose(fit["correlation"].values, reference["correlation"].values)
    assert all(d.dist.name == "t" for d in fit["marginals"].values())


def test_simulate_copula_reproduces_marginals_and_correlation():
    returns = load_csv("test13_returns.csv")
    sims, fit = simulate_copula(returns, marginals="normal", output="both", seed=2)

    # With Normal marginals, each simulated column's mean/std should match
    # its fitted Normal, and the Pearson correlation should match the
    # copula's (sampling-noise tolerance at 100k draws).
    for col in returns.columns:
        assert sims[col].mean() == pytest.approx(fit["marginals"][col].mean(), abs=3e-2 * fit["marginals"][col].std())
        assert sims[col].std() == pytest.approx(fit["marginals"][col].std(), rel=2e-2)
    assert np.allclose(sims.corr().values, fit["correlation"].values, atol=2e-2)


def test_t_copula_has_more_joint_extremes_than_gaussian():
    returns = load_csv("test13_returns.csv").iloc[:, :2]

    def joint_tail_count(sims):
        lows = sims < sims.quantile(0.01)
        return lows.all(axis=1).sum()

    gaussian = simulate_copula(returns, copula="gaussian", correlation="kendall", seed=3)
    t_low_nu = simulate_copula(returns, copula="t", nu=3.0, seed=3)

    assert joint_tail_count(t_low_nu) > joint_tail_count(gaussian)


def test_simulate_copula_is_reproducible_with_seed():
    returns = load_csv("test13_returns.csv")
    a = simulate_copula(returns, num_simulations=500, seed=7)
    b = simulate_copula(returns, num_simulations=500, seed=7)
    assert a.equals(b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"copula": "clayton"},
        {"marginals": "lognormal"},
        {"correlation": "pearson"},
        {"copula": "t", "correlation": "normal_scores"},
        {"copula": "gaussian", "nu": 5.0},
        {"output": "fit"},
    ],
)
def test_simulate_copula_invalid_options(kwargs):
    returns = load_csv("test13_returns.csv")
    with pytest.raises(ValueError):
        simulate_copula(returns, num_simulations=10, **kwargs)
