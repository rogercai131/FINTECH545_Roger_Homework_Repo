"""
Tests for moments.py.

There is no reference CSV for this calculation in test_files/, so results are
checked against pandas / scipy, which implement the same estimators.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from moments import first_four_moments


@pytest.fixture
def sample():
    rng = np.random.default_rng(545)
    return pd.Series(rng.standard_t(df=5, size=500))


def test_unbiased_matches_pandas(sample):
    result = first_four_moments(sample)

    assert np.isclose(result["Mean"], sample.mean())
    assert np.isclose(result["Variance"], sample.var())
    assert np.isclose(result["Skewness"], sample.skew())
    assert np.isclose(result["Kurtosis"], sample.kurt())


def test_biased_matches_scipy(sample):
    result = first_four_moments(sample, method="biased")

    assert np.isclose(result["Variance"], np.var(sample))
    assert np.isclose(result["Skewness"], stats.skew(sample, bias=True))
    assert np.isclose(result["Kurtosis"], stats.kurtosis(sample, bias=True))


def test_raw_kurtosis_is_excess_plus_three(sample):
    excess = first_four_moments(sample)["Kurtosis"]
    raw = first_four_moments(sample, kurtosis_type="raw")["Kurtosis"]
    assert np.isclose(raw, excess + 3.0)


def test_dataframe_input_per_column_and_missing_strategies():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=8),
        "A": [1.0, 2.0, 4.0, 7.0, np.nan, 3.0, 5.0, 9.0],
        "B": [2.0, 1.0, 0.0, 5.0, 3.0, 8.0, 2.0, 4.0],
    })

    dropped = first_four_moments(df)
    assert list(dropped.columns) == ["A", "B"]  # "Date" ignored
    assert list(dropped.index) == ["Mean", "Variance", "Skewness", "Kurtosis"]
    # drop_row: B loses the row where A is missing.
    assert np.isclose(dropped.loc["Mean", "B"], df["B"].drop(index=4).mean())

    pairwise = first_four_moments(df, missing_strategy="pairwise")
    # pairwise: B keeps all 8 rows.
    assert np.isclose(pairwise.loc["Mean", "B"], df["B"].mean())
    assert np.isclose(pairwise.loc["Kurtosis", "A"], df["A"].kurt())


def test_invalid_options_raise(sample):
    with pytest.raises(ValueError):
        first_four_moments(sample, method="population")
    with pytest.raises(ValueError):
        first_four_moments(sample, kurtosis_type="fisher")
    with pytest.raises(ValueError):
        first_four_moments([1.0, 2.0, 3.0])  # too few points for unbiased
