"""
Tests for portfolio_risk.py, based on Tests.xlsx section 9.1.

Like the other Monte Carlo tests in this suite (test_simulation.py,
test_risk_metrics.py's "simulation" method tests), the copula simulation
here can't reproduce `testout9_1.csv` exactly -- it's one particular
100,000-draw random sample. Instead we simulate our own sample with a
fixed seed and check it lands close to the reference values, within the
tolerance expected from sampling noise at this size (empirically, under
~1% relative error here).
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from portfolio_risk import calculate_portfolio_var_es

RTOL = 0.03


def load_csv(name, **kwargs):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name), **kwargs)


def test_calculate_portfolio_var_es():
    # Test 9.1
    portfolio = load_csv("test9_1_portfolio.csv", encoding="utf-8-sig")
    returns = load_csv("test9_1_returns.csv")
    expected = load_csv("testout9_1.csv").set_index("Stock")

    result = calculate_portfolio_var_es(portfolio, returns, alpha=0.05, seed=4).set_index("Stock")

    assert list(result.columns) == ["VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]
    for stock in ["A", "B", "Total"]:
        for col in ["VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]:
            assert np.isclose(result.loc[stock, col], expected.loc[stock, col], rtol=RTOL), (
                stock,
                col,
            )


def test_calculate_portfolio_var_es_pct_consistent_with_dollar_var():
    # Sanity check that the "_Pct" columns are just the dollar columns
    # divided by each holding's (or the total's) starting value.
    portfolio = load_csv("test9_1_portfolio.csv", encoding="utf-8-sig")
    returns = load_csv("test9_1_returns.csv")

    result = calculate_portfolio_var_es(portfolio, returns, seed=2).set_index("Stock")

    values = {"A": 100 * 20, "B": 100 * 30, "Total": 100 * 20 + 100 * 30}
    for stock, value in values.items():
        assert np.isclose(result.loc[stock, "VaR95_Pct"], result.loc[stock, "VaR95"] / value)
        assert np.isclose(result.loc[stock, "ES95_Pct"], result.loc[stock, "ES95"] / value)


def test_calculate_portfolio_var_es_invalid_distribution():
    portfolio = load_csv("test9_1_portfolio.csv", encoding="utf-8-sig")
    portfolio.loc[0, "Distribution"] = "NotARealDistribution"
    returns = load_csv("test9_1_returns.csv")

    with pytest.raises(ValueError):
        calculate_portfolio_var_es(portfolio, returns)
