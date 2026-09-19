"""
Tests for psd_fix.py, based on Tests.xlsx sections 3.1 - 3.4.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from psd_fix import fix_psd_matrix


def load_csv(name):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name))


@pytest.mark.parametrize(
    "method, input_file, expected_file",
    [
        ("near_psd", "testout_1.3.csv", "testout_3.1.csv"),  # Test 3.1: near_psd covariance
        ("near_psd", "testout_1.4.csv", "testout_3.2.csv"),  # Test 3.2: near_psd correlation
        ("higham", "testout_1.3.csv", "testout_3.3.csv"),    # Test 3.3: Higham covariance
        ("higham", "testout_1.4.csv", "testout_3.4.csv"),    # Test 3.4: Higham correlation
    ],
)
def test_fix_psd_matrix(method, input_file, expected_file):
    matrix = load_csv(input_file)
    expected = load_csv(expected_file)

    result = fix_psd_matrix(matrix, method=method)

    assert np.allclose(result.values, expected.values, atol=1e-6)


def test_fix_psd_matrix_invalid_method():
    matrix = load_csv("testout_1.3.csv")
    with pytest.raises(ValueError):
        fix_psd_matrix(matrix, method="not_a_real_method")
