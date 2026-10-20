"""
Tests for risk_metrics.py, based on Tests.xlsx sections 8.1 - 8.6.

The "simulation" method tests (8.3, 8.6) can't match their reference CSV
exactly -- they involve drawing a fresh random sample -- so, like the
Monte Carlo tests in test_simulation.py, they instead check the result is
close to the corresponding analytical value (8.2 for 8.3, 8.5 for 8.6)
within the tolerance expected from sampling noise at this sample size, per
the test description ("compare to 8.2/8.5 values"). The "historical" method
has no reference CSV, so it's checked by hand and against numpy directly.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_FILES_DIR
from risk_metrics import calculate_var, calculate_es

RTOL = 1e-4
SIMULATION_ATOL = 0.01


def load_csv(name):
    return pd.read_csv(os.path.join(TEST_FILES_DIR, name))


def test_calculate_var_normal():
    # Test 8.1
    data = load_csv("test7_1.csv")["x1"]
    expected = load_csv("testout8_1.csv").iloc[0]

    result = calculate_var(data, distribution="normal")

    assert np.isclose(result["VaR Absolute"], expected["VaR Absolute"], rtol=RTOL)
    assert np.isclose(result["VaR Diff from Mean"], expected["VaR Diff from Mean"], rtol=RTOL)


def test_calculate_var_t():
    # Test 8.2
    data = load_csv("test7_2.csv")["x1"]
    expected = load_csv("testout8_2.csv").iloc[0]

    result = calculate_var(data, distribution="t")

    assert np.isclose(result["VaR Absolute"], expected["VaR Absolute"], rtol=RTOL)
    assert np.isclose(result["VaR Diff from Mean"], expected["VaR Diff from Mean"], rtol=RTOL)


def test_calculate_var_simulation():
    # Test 8.3: compare against the analytical (8.2) value
    data = load_csv("test7_2.csv")["x1"]
    analytical = calculate_var(data, distribution="t")

    result = calculate_var(data, distribution="t", method="simulation", seed=1)

    assert np.isclose(result["VaR Absolute"], analytical["VaR Absolute"], atol=SIMULATION_ATOL)
    assert np.isclose(
        result["VaR Diff from Mean"], analytical["VaR Diff from Mean"], atol=SIMULATION_ATOL
    )


def test_calculate_es_normal():
    # Test 8.4
    data = load_csv("test7_1.csv")["x1"]
    expected = load_csv("testout8_4.csv").iloc[0]

    result = calculate_es(data, distribution="normal")

    assert np.isclose(result["ES Absolute"], expected["ES Absolute"], rtol=RTOL)
    assert np.isclose(result["ES Diff from Mean"], expected["ES Diff from Mean"], rtol=RTOL)


def test_calculate_es_t():
    # Test 8.5
    data = load_csv("test7_2.csv")["x1"]
    expected = load_csv("testout8_5.csv").iloc[0]

    result = calculate_es(data, distribution="t")

    assert np.isclose(result["ES Absolute"], expected["ES Absolute"], rtol=RTOL)
    assert np.isclose(result["ES Diff from Mean"], expected["ES Diff from Mean"], rtol=RTOL)


def test_calculate_es_simulation():
    # Test 8.6: compare against the analytical (8.5) value
    data = load_csv("test7_2.csv")["x1"]
    analytical = calculate_es(data, distribution="t")

    result = calculate_es(data, distribution="t", method="simulation", seed=1)

    assert np.isclose(result["ES Absolute"], analytical["ES Absolute"], atol=SIMULATION_ATOL)
    assert np.isclose(
        result["ES Diff from Mean"], analytical["ES Diff from Mean"], atol=SIMULATION_ATOL
    )


def test_calculate_var_invalid_distribution():
    data = load_csv("test7_1.csv")["x1"]
    with pytest.raises(ValueError):
        calculate_var(data, distribution="not_a_real_distribution")


def test_calculate_var_invalid_method():
    data = load_csv("test7_1.csv")["x1"]
    with pytest.raises(ValueError):
        calculate_var(data, method="not_a_real_method")


def test_calculate_es_invalid_distribution():
    data = load_csv("test7_1.csv")["x1"]
    with pytest.raises(ValueError):
        calculate_es(data, distribution="not_a_real_distribution")


# --- method="historical": no reference CSV; checked by hand and against numpy ---


def test_historical_var_es_by_hand():
    # 20 returns: two big losses (-0.20, -0.19) and 18 gains. At alpha=0.10,
    # np.quantile interpolates between -0.19 and 0.01 (cutoff = -0.01), so
    # the tail at/below the cutoff is exactly the two losses.
    returns = pd.Series([-0.20, -0.19, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                         0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18])
    cutoff = np.quantile(returns, 0.10)

    var = calculate_var(returns, method="historical", alpha=0.10)
    es = calculate_es(returns, method="historical", alpha=0.10)

    assert var["VaR Absolute"] == pytest.approx(-cutoff)
    assert es["ES Absolute"] == pytest.approx(-returns[returns <= cutoff].mean())
    assert es["ES Absolute"] >= var["VaR Absolute"]
    # Diff from Mean = Absolute + sample mean
    assert var["VaR Diff from Mean"] == pytest.approx(-cutoff + returns.mean())
    assert es["ES Diff from Mean"] == pytest.approx(es["ES Absolute"] + returns.mean())


def test_historical_ignores_distribution_and_drops_nan():
    data = load_csv("test7_1.csv")["x1"]
    with_nan = pd.concat([data, pd.Series([np.nan, np.nan])], ignore_index=True)

    base = calculate_var(data, method="historical")
    assert base.equals(calculate_var(data, method="historical", distribution="t"))
    assert np.allclose(calculate_var(with_nan, method="historical"), base)
    assert np.allclose(calculate_es(with_nan, method="historical"), calculate_es(data, method="historical"))


def test_historical_close_to_fitted_normal_on_normal_data():
    # On a big truly-normal sample, historical VaR/ES should approach the
    # analytical normal answers (sampling-noise tolerance at 200k draws).
    sample = np.random.default_rng(8).normal(0.001, 0.02, 200_000)
    assert calculate_var(sample, method="historical")["VaR Absolute"] == pytest.approx(
        calculate_var(sample)["VaR Absolute"], rel=1e-2)
    assert calculate_es(sample, method="historical")["ES Absolute"] == pytest.approx(
        calculate_es(sample)["ES Absolute"], rel=1e-2)


def test_calculate_es_invalid_method():
    data = load_csv("test7_1.csv")["x1"]
    with pytest.raises(ValueError):
        calculate_es(data, method="bootstrap")
