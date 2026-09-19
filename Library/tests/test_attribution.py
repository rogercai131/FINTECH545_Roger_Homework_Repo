"""
Tests for attribution.py, based on Tests.xlsx sections 11.1 - 11.2.
"""

import os

import numpy as np
import pandas as pd

from conftest import TEST_FILES_DIR
from attribution import return_attribution


def load_csv(name, **kwargs):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name), **kwargs)


def test_return_attribution_by_holding():
    # Test 11.1
    returns = load_csv("test11_1_returns.csv")
    weights = load_csv("test11_1_weights.csv")["W"].values
    expected = load_csv("testout11_1.csv").set_index("Value")

    result = return_attribution(returns, weights)

    assert list(result.columns) == ["x1", "x2", "x3", "Portfolio"]
    assert np.allclose(result.values, expected.values)


def test_return_attribution_by_factor():
    # Test 11.2
    factor_returns = load_csv("test11_2_factor_returns.csv")
    stock_returns = load_csv("test11_2_stock_returns.csv")
    betas = load_csv("test11_2_beta.csv").set_index("Stock")
    weights = load_csv("test11_2_weights.csv")["W"].values
    expected = load_csv("testout11_2.csv").set_index("Value")

    result = return_attribution(
        stock_returns, weights, factor_returns=factor_returns, betas=betas
    )

    assert list(result.columns) == ["F1", "F2", "F3", "Alpha", "Portfolio"]
    assert np.allclose(result.values, expected.values)


def test_return_attribution_sums_to_portfolio():
    # Sanity check on the defining property of Carino/vol attribution,
    # independent of the reference CSV: contributions must sum exactly to
    # the portfolio-level total for both the return and vol rows.
    returns = load_csv("test11_1_returns.csv")
    weights = load_csv("test11_1_weights.csv")["W"].values

    result = return_attribution(returns, weights)
    contributor_cols = [c for c in result.columns if c != "Portfolio"]

    assert np.isclose(
        result.loc["Return Attribution", contributor_cols].sum(),
        result.loc["Return Attribution", "Portfolio"],
    )
    assert np.isclose(
        result.loc["Vol Attribution", contributor_cols].sum(),
        result.loc["Vol Attribution", "Portfolio"],
    )
