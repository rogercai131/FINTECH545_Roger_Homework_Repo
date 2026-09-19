"""
Tests for covariance.py, based on Tests.xlsx sections 1.1 - 1.4.

Each test loads the shared `test1.csv` input and checks `covariance_matrix`
against the matching `testout_1.x.csv` expected output, for every
combination of method ("covariance"/"correlation") and missing_strategy
("drop_row"/"pairwise").
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from covariance import covariance_matrix, ew_covariance_matrix, kendall_correlation_matrix


def load_csv(name):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name))


@pytest.mark.parametrize(
    "method, missing_strategy, expected_file",
    [
        ("covariance", "drop_row", "testout_1.1.csv"),   # Test 1.1
        ("correlation", "drop_row", "testout_1.2.csv"),  # Test 1.2
        ("covariance", "pairwise", "testout_1.3.csv"),   # Test 1.3
        ("correlation", "pairwise", "testout_1.4.csv"),  # Test 1.4
    ],
)
def test_covariance_matrix_missing_data(method, missing_strategy, expected_file):
    data = load_csv("test1.csv")
    expected = load_csv(expected_file)

    result = covariance_matrix(data, method=method, missing_strategy=missing_strategy)

    assert np.allclose(result.values, expected.values)


def test_covariance_matrix_invalid_method():
    data = load_csv("test1.csv")
    with pytest.raises(ValueError):
        covariance_matrix(data, method="not_a_real_method")


def test_covariance_matrix_invalid_missing_strategy():
    data = load_csv("test1.csv")
    with pytest.raises(ValueError):
        covariance_matrix(data, missing_strategy="not_a_real_strategy")


def test_ew_covariance_lambda_097():
    # Test 2.1
    data = load_csv("test2.csv")
    expected = load_csv("testout_2.1.csv")

    result = ew_covariance_matrix(data, lambda_=0.97, method="covariance")

    assert np.allclose(result.values, expected.values)


def test_ew_correlation_lambda_094():
    # Test 2.2
    data = load_csv("test2.csv")
    expected = load_csv("testout_2.2.csv")

    result = ew_covariance_matrix(data, lambda_=0.94, method="correlation")

    assert np.allclose(result.values, expected.values)


def test_ew_covariance_mixed_lambda():
    # Test 2.3: EW variance (lambda=0.97) combined with EW correlation (lambda=0.94)
    data = load_csv("test2.csv")
    expected = load_csv("testout_2.3.csv")

    result = ew_covariance_matrix(
        data, lambda_=0.97, method="covariance", lambda_corr=0.94
    )

    assert np.allclose(result.values, expected.values)


def test_ew_covariance_matrix_invalid_method():
    data = load_csv("test2.csv")
    with pytest.raises(ValueError):
        ew_covariance_matrix(data, method="not_a_real_method")


def test_kendall_correlation_matrix():
    # Test 13.1
    data = load_csv("test13_returns.csv")
    expected = load_csv("testout13_1.csv")

    result = kendall_correlation_matrix(data)

    assert np.allclose(result.values, expected.values)
