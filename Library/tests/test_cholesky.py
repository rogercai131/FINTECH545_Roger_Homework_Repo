"""
Tests for cholesky.py, based on Tests.xlsx section 4.1.
"""

import os

import numpy as np
import pandas as pd

from conftest import TEST_FILES_DIR
from cholesky import cholesky_psd


def load_csv(name):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name))


def test_cholesky_psd():
    # Test 4.1: chol_psd of the near_psd covariance matrix from Test 3.1
    matrix = load_csv("testout_3.1.csv")
    expected = load_csv("testout_4.1.csv")

    result = cholesky_psd(matrix)

    assert np.allclose(result.values, expected.values, atol=1e-6)
    # Sanity check the defining property: L @ L.T reproduces the input.
    assert np.allclose(result.values @ result.values.T, matrix.values, atol=1e-6)
