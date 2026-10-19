"""
Tests for returns.py: return_calculate is based on Tests.xlsx sections
6.1 - 6.2; the classical method, price_from_returns, and
expected_price_moments are checked against their closed-form definitions.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from returns import return_calculate, price_from_returns, expected_price_moments


def load_csv(name):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name))


def test_return_calculate_arithmetic():
    # Test 6.1
    prices = load_csv("test6.csv")
    expected = load_csv("testout6_1.csv")

    result = return_calculate(prices, method="arithmetic")

    assert list(result["Date"]) == list(expected["Date"])
    assert np.allclose(
        result.drop(columns=["Date"]).values, expected.drop(columns=["Date"]).values
    )


def test_return_calculate_log():
    # Test 6.2
    prices = load_csv("test6.csv")
    expected = load_csv("testout6_2.csv")

    result = return_calculate(prices, method="log")

    assert list(result["Date"]) == list(expected["Date"])
    assert np.allclose(
        result.drop(columns=["Date"]).values, expected.drop(columns=["Date"]).values
    )


def test_return_calculate_without_date_column():
    prices = load_csv("test6.csv").drop(columns=["Date"])
    expected = load_csv("testout6_1.csv").drop(columns=["Date"])

    result = return_calculate(prices, method="arithmetic", date_column=None)

    assert "Date" not in result.columns
    assert np.allclose(result.values, expected.values)


def test_return_calculate_invalid_method():
    prices = load_csv("test6.csv")
    with pytest.raises(ValueError):
        return_calculate(prices, method="not_a_real_method")


def test_return_calculate_classical_and_geometric():
    prices = load_csv("test6.csv")
    levels = prices.drop(columns=["Date"])

    classical = return_calculate(prices, method="classical").drop(columns=["Date"])
    assert np.allclose(classical.values, levels.diff().iloc[1:].values)

    geometric = return_calculate(prices, method="geometric")
    assert geometric.equals(return_calculate(prices, method="log"))


@pytest.mark.parametrize(
    "method, expected",
    [
        ("classical", [100.5, 99.0]),                         # 100 + r
        ("arithmetic", [150.0, 0.0]),                         # 100 * (1 + r)
        ("geometric", [100 * np.exp(0.5), 100 * np.exp(-1.0)]),  # 100 * e^r
    ],
)
def test_price_from_returns_one_step(method, expected):
    result = price_from_returns([0.5, -1.0], start_price=100, method=method)
    assert np.allclose(result.values, expected)


@pytest.mark.parametrize("method", ["classical", "arithmetic", "geometric"])
def test_price_from_returns_path_round_trips(method):
    # prices -> returns -> prices should reproduce the original path.
    prices = load_csv("test6.csv").drop(columns=["Date"])
    returns = return_calculate(prices, method=method, date_column=None)

    rebuilt = price_from_returns(returns, prices.iloc[0], method=method, horizon="path")

    assert list(rebuilt.columns) == list(prices.columns)
    assert np.allclose(rebuilt.values, prices.iloc[1:].values)


@pytest.mark.parametrize("method", ["classical", "arithmetic", "geometric"])
def test_expected_price_moments_match_simulation(method):
    # Monte Carlo check: simulated next-price mean/std should land close to
    # the closed-form values (tolerance sized to sampling noise at 1e6 draws).
    rng = np.random.default_rng(545)
    sigma, start = 0.2, 100.0
    simulated = price_from_returns(rng.normal(0.0, sigma, 1_000_000), start, method=method)

    expected = expected_price_moments(start, sigma, method=method)

    assert simulated.mean() == pytest.approx(expected["Mean"], rel=2e-3)
    assert simulated.std() == pytest.approx(expected["Std"], rel=5e-3)


def test_price_functions_invalid_options():
    with pytest.raises(ValueError):
        price_from_returns([0.1], 100, method="log")  # use "geometric" here
    with pytest.raises(ValueError):
        price_from_returns([0.1], 100, horizon="forever")
    with pytest.raises(ValueError):
        expected_price_moments(100, sigma=-0.1)
