"""
Tests for psd_fix.py: fix_psd_matrix is based on Tests.xlsx sections 3.1 - 3.4;
classify_psd is checked against small hand-built matrices.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from psd_fix import fix_psd_matrix, classify_psd


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


# --- classify_psd (no reference CSV; checked against hand-built matrices) ---



def test_classify_pd():
    r = classify_psd([[2.0, 1.0], [1.0, 2.0]])
    assert r["classification"] == "PD"
    assert np.allclose(r["eigenvalues"], [1.0, 3.0])
    assert r["min_eigenvalue"] == pytest.approx(1.0)
    assert r["max_eigenvalue"] == pytest.approx(3.0)


def test_classify_psd_singular():
    r = classify_psd([[1.0, 1.0], [1.0, 1.0]])
    assert r["classification"] == "PSD"
    assert r["negative_indices"].size == 0


def test_classify_psd_rounding_error_treated_as_zero():
    v = np.array([1.0, 2.0, 3.0])
    matrix = np.outer(v, v) - 1e-12 * np.eye(3)
    original = matrix.copy()

    r = classify_psd(matrix)

    assert r["classification"] == "PSD"
    assert r["small_negative_indices"].size == 2
    assert np.all(r["small_negative_values"] < 0)
    assert np.all(r["adjusted_eigenvalues"][r["small_negative_indices"]] == 0)
    assert "2 small negative" in r["summary"]
    assert np.array_equal(matrix, original)  # input never modified


def test_classify_non_psd_reports_negatives():
    r = classify_psd([[1.0, 2.0], [2.0, 1.0]])
    assert r["classification"] == "non-PSD"
    assert np.allclose(r["negative_values"], [-1.0])
    assert r["adjusted_eigenvalues"][0] == pytest.approx(-1.0)  # not repaired


def test_tolerance_is_scale_aware():
    # Same relative rounding error, at a 1e6 scale: still PSD.
    v = np.array([1.0, 2.0, 3.0])
    r = classify_psd(1e6 * np.outer(v, v) - 1e-6 * np.eye(3))
    assert r["classification"] == "PSD"
    # Tightening the tolerance flips the verdict — it's a numerical call.
    assert classify_psd(np.outer(v, v) - 1e-12 * np.eye(3), atol=0, rtol=0)["classification"] == "non-PSD"


def test_non_symmetric_report_and_symmetrize():
    matrix = [[1.0, 2.0], [0.0, 1.0]]

    reported = classify_psd(matrix)
    assert reported["classification"] == "non-symmetric"
    assert reported["eigenvalues"] is None
    assert reported["max_asymmetry"] == pytest.approx(2.0)

    symmetrized = classify_psd(matrix, asymmetry_strategy="symmetrize")
    assert symmetrized["symmetrized"] is True
    assert symmetrized["classification"] == "PSD"  # (A + A^T)/2 = [[1,1],[1,1]]
    assert "NOT symmetric" in symmetrized["summary"]


@pytest.mark.parametrize(
    "bad_input, error",
    [
        ([], ValueError),
        ([1.0, 2.0], ValueError),                       # not 2-D
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ValueError),  # not square
        ([[1.0, np.nan], [np.nan, 1.0]], ValueError),   # not finite
        ([[1 + 1j, 0], [0, 1]], TypeError),             # complex
        ([["a", "b"], ["c", "d"]], TypeError),          # not numeric
    ],
)
def test_classify_psd_invalid_input(bad_input, error):
    with pytest.raises(error):
        classify_psd(bad_input)


def test_classify_psd_invalid_options():
    with pytest.raises(ValueError):
        classify_psd(np.eye(2), asymmetry_strategy="ignore")
    with pytest.raises(ValueError):
        classify_psd(np.eye(2), atol=-1.0)
