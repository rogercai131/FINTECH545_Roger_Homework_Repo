"""
Tests for returns.py, based on Tests.xlsx sections 6.1 - 6.2.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from returns import return_calculate


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
