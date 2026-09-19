"""
Shared helper functions used across the risk-calculation modules in this library.

Anything that is reused by more than one calculation file (input validation,
missing-data handling, option-string dispatch, etc.) should live here instead
of being duplicated in each calculation module.
"""

import pandas as pd


def handle_missing_data(data: pd.DataFrame, missing_strategy: str = "drop_row") -> pd.DataFrame:
    """
    Apply a missing-data handling strategy to a DataFrame before a calculation.

    This is an "option-style" dispatcher: new strategies can be added later
    (e.g. "fill_mean") without changing the signature of the functions that
    call this helper.

    Parameters
    ----------
    data : pd.DataFrame
        The raw input data, which may contain NaN values.
    missing_strategy : str, default "drop_row"
        How to handle missing values:
          - "drop_row": remove any row that has at least one missing value
            (listwise deletion). Every column ends up using the exact same
            set of rows.
          - "pairwise": keep all rows as-is. Each pair of columns will only
            drop the rows where *that specific pair* has missing data. This
            is handled downstream by pandas' own `.cov()` / `.corr()`, which
            already skip NaNs pairwise, so here we simply return the data
            unchanged and let the caller compute on it directly.

    Returns
    -------
    pd.DataFrame
        The data ready to be handed to the actual calculation.

    Raises
    ------
    ValueError
        If `missing_strategy` is not one of the supported options.
    """
    if missing_strategy == "drop_row":
        return data.dropna()
    elif missing_strategy == "pairwise":
        return data
    else:
        raise ValueError(
            f"Unsupported missing_strategy '{missing_strategy}'. "
            "Expected one of: 'drop_row', 'pairwise'."
        )


def aicc(log_likelihood: float, num_params: int, num_obs: int) -> float:
    """
    Corrected Akaike Information Criterion (AICc) for a fitted model.

    AICc adjusts the plain AIC with an extra penalty for small sample
    sizes relative to the number of parameters, which matters a lot for
    something like a 100-observation distribution fit with 3+ parameters.
    Lower is better (better fit, penalized for complexity).

    Parameters
    ----------
    log_likelihood : float
        The maximized log-likelihood of the fitted model.
    num_params : int
        Number of estimated parameters in the model (e.g. 3 for a fitted
        Student's t distribution: location, scale, degrees of freedom).
    num_obs : int
        Number of observations the model was fit on.

    Returns
    -------
    float
        The AICc value.
    """
    aic = -2 * log_likelihood + 2 * num_params
    correction = (2 * num_params * (num_params + 1)) / (num_obs - num_params - 1)
    return aic + correction
