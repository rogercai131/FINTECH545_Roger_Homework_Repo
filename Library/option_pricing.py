"""
Pricing a single option and computing its Greeks.

Covers: FINTECH545 Tests 12.1 - 12.3 (European options via the closed-form
Generalized Black-Scholes-Merton (GBSM) formula, American options via a
binomial (Cox-Ross-Rubinstein) tree with a continuous dividend yield, and
American options with discrete dollar dividends on specific days).

A note on the two Greek conventions used here: the closed-form GBSM Greeks
(`price_european_option`) use the standard finance convention -- e.g.
Theta = dV/d(calendar time) = -dV/d(time-to-maturity), which is negative
for a typical long option (it loses value as time passes). The binomial
Greeks (`price_american_option`) are computed by bump-and-reprice, which
here differ by using Theta = +dV/d(time-to-maturity) directly and
Rho = dV/dr + dV/dq (equivalent to the sensitivity to the interest rate
holding the "cost of carry" r - q fixed). These are simply two different,
each internally-consistent conventions -- the sign/definition differs, so
don't compare a GBSM Greek to a binomial one directly without adjusting
for that.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

_GREEK_COLUMNS = ["Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]


def _validate_option_type(option_type: str):
    if option_type not in ("Call", "Put"):
        raise ValueError(f"Unsupported option_type '{option_type}'. Expected 'Call' or 'Put'.")


def price_european_option(
    option_type: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> pd.Series:
    """
    Price a European option and its Greeks via the closed-form Generalized
    Black-Scholes-Merton (GBSM) formula (Black-Scholes with a continuous
    dividend yield).

    Expected input
    ---------------
    option_type : str
        "Call" or "Put".
    S : float
        Current price of the underlying.
    K : float
        Strike price.
    T : float
        Time to maturity, in years (e.g. DaysToMaturity / DayPerYear).
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    sigma : float
        Volatility of the underlying (annualized).
    q : float, default 0.0
        Continuous dividend yield (annualized).

    Output
    ------
    pd.Series
        Index ["Value", "Delta", "Gamma", "Vega", "Rho", "Theta"].
    """
    _validate_option_type(option_type)

    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = norm.pdf(d1)

    # Gamma and Vega have the same formula for calls and puts.
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * np.exp(-q * T) * pdf_d1 * sqrt_t

    if option_type == "Call":
        n_d1, n_d2 = norm.cdf(d1), norm.cdf(d2)
        value = S * np.exp(-q * T) * n_d1 - K * np.exp(-r * T) * n_d2
        delta = np.exp(-q * T) * n_d1
        rho = K * T * np.exp(-r * T) * n_d2
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * sqrt_t)
            - r * K * np.exp(-r * T) * n_d2
            + q * S * np.exp(-q * T) * n_d1
        )
    else:
        n_neg_d1, n_neg_d2 = norm.cdf(-d1), norm.cdf(-d2)
        value = K * np.exp(-r * T) * n_neg_d2 - S * np.exp(-q * T) * n_neg_d1
        delta = -np.exp(-q * T) * n_neg_d1
        rho = -K * T * np.exp(-r * T) * n_neg_d2
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * sqrt_t)
            + r * K * np.exp(-r * T) * n_neg_d2
            - q * S * np.exp(-q * T) * n_neg_d1
        )

    return pd.Series(
        [value, delta, gamma, vega, rho, theta], index=_GREEK_COLUMNS
    )


def _binomial_tree_value(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    num_steps: int,
    q: float = 0.0,
    dividend_steps=None,
    dividend_amounts=None,
) -> float:
    """
    Price an American option with a Cox-Ross-Rubinstein binomial tree.

    If `dividend_steps` is empty/None, a continuous dividend yield `q`
    drives the risk-neutral growth rate as usual. Otherwise, dividends are
    handled as discrete dollar amounts at specific steps (and `q` is
    ignored) via the standard recursive method: build the tree only up to
    the next dividend date, then at each node just before the ex-dividend
    date, the option's value is the greater of (a) exercising immediately
    at the pre-dividend price, or (b) continuing to hold, valued by
    recursively re-pricing an option on the post-dividend price for the
    remaining time and remaining dividends.
    """
    if dividend_steps:
        return _binomial_tree_value_with_discrete_dividends(
            S, K, T, r, sigma, option_type, num_steps, list(dividend_steps), list(dividend_amounts)
        )

    dt = T / num_steps
    up = np.exp(sigma * np.sqrt(dt))
    down = 1 / up
    prob_up = (np.exp((r - q) * dt) - down) / (up - down)
    discount = np.exp(-r * dt)

    ups = np.arange(num_steps, -1, -1)
    downs = np.arange(0, num_steps + 1)
    prices = S * up**ups * down**downs
    values = np.maximum(prices - K, 0) if option_type == "Call" else np.maximum(K - prices, 0)

    for step in range(num_steps - 1, -1, -1):
        prices = S * up ** np.arange(step, -1, -1) * down ** np.arange(0, step + 1)
        values = discount * (prob_up * values[:-1] + (1 - prob_up) * values[1:])
        exercise = np.maximum(prices - K, 0) if option_type == "Call" else np.maximum(K - prices, 0)
        values = np.maximum(values, exercise)

    return values[0]


def _binomial_tree_value_with_discrete_dividends(
    S, K, T, r, sigma, option_type, num_steps, dividend_steps, dividend_amounts
):
    if not dividend_steps or dividend_steps[0] > num_steps:
        return _binomial_tree_value(S, K, T, r, sigma, option_type, num_steps)

    steps_to_dividend = dividend_steps[0]
    dt = T / num_steps
    up = np.exp(sigma * np.sqrt(dt))
    down = 1 / up
    prob_up = (np.exp(r * dt) - down) / (up - down)
    discount = np.exp(-r * dt)

    # Prices right before the dividend is paid, at each node of that step.
    pre_dividend_prices = (
        S * up ** np.arange(steps_to_dividend, -1, -1) * down ** np.arange(0, steps_to_dividend + 1)
    )
    remaining_time = T - steps_to_dividend * dt
    remaining_steps = num_steps - steps_to_dividend
    next_dividend_steps = [s - steps_to_dividend for s in dividend_steps[1:]]

    immediate_exercise = (
        np.maximum(pre_dividend_prices - K, 0)
        if option_type == "Call"
        else np.maximum(K - pre_dividend_prices, 0)
    )
    continuation = np.array(
        [
            _binomial_tree_value_with_discrete_dividends(
                price - dividend_amounts[0],
                K,
                remaining_time,
                r,
                sigma,
                option_type,
                remaining_steps,
                next_dividend_steps,
                dividend_amounts[1:],
            )
            for price in pre_dividend_prices
        ]
    )
    values = np.maximum(immediate_exercise, continuation)

    # Back through the steps leading up to the dividend date as usual.
    for step in range(steps_to_dividend - 1, -1, -1):
        prices = S * up ** np.arange(step, -1, -1) * down ** np.arange(0, step + 1)
        values = discount * (prob_up * values[:-1] + (1 - prob_up) * values[1:])
        exercise = np.maximum(prices - K, 0) if option_type == "Call" else np.maximum(K - prices, 0)
        values = np.maximum(values, exercise)

    return values[0]


def price_american_option(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    num_steps: int = 500,
    dividend_days=None,
    dividend_amounts=None,
    day_per_year: float = 365,
    compute_greeks: bool = True,
) -> pd.Series:
    """
    Price an American option (which may be exercised at any time before
    maturity) via a binomial tree, optionally with its Greeks estimated by
    bump-and-reprice.

    Expected input
    ---------------
    option_type : str
        "Call" or "Put".
    S, K, T, r, sigma : float
        Same meaning as in `price_european_option`.
    q : float, default 0.0
        Continuous dividend yield. Ignored if `dividend_days` is given.
    num_steps : int, default 500
        Number of steps in the binomial tree. Higher is more accurate but
        slower; 500 matches the precision used in this library's reference
        test cases.
    dividend_days : array-like, optional
        Days from now (on the same day-count basis as `day_per_year`) on
        which discrete dollar dividends are paid, e.g. [75, 150]. If
        given, discrete dividends are used instead of the continuous
        yield `q`.
    dividend_amounts : array-like, optional
        Dollar amount of each dividend in `dividend_days`, same length and
        order.
    day_per_year : float, default 365
        Day-count basis `T` and `dividend_days` are expressed in.
    compute_greeks : bool, default True
        If False, skip the (more expensive) Greek estimation and return
        only "Value" -- useful when only the price is needed.

    Output
    ------
    pd.Series
        Index ["Value", "Delta", "Gamma", "Vega", "Rho", "Theta"] (or just
        ["Value"] if `compute_greeks=False`).

    Raises
    ------
    ValueError
        If `option_type` is not "Call"/"Put".
    """
    _validate_option_type(option_type)

    dividend_steps = None
    if dividend_days is not None:
        total_days = T * day_per_year
        dividend_steps = [round(day / total_days * num_steps) for day in dividend_days]

    def value_at(S_, K_, T_, r_, sigma_, q_):
        return _binomial_tree_value(
            S_, K_, T_, r_, sigma_, option_type, num_steps, q=q_,
            dividend_steps=dividend_steps, dividend_amounts=dividend_amounts,
        )

    value = value_at(S, K, T, r, sigma, q)

    if not compute_greeks:
        return pd.Series([value], index=["Value"])

    # Bump sizes below were chosen (and verified against this library's
    # reference test cases) to balance finite-difference truncation error
    # against the binomial tree's own discretization noise -- Gamma in
    # particular is a second derivative and much more sensitive to bump
    # size than the others, hence its larger, separately-tuned bump.
    delta_bump = 0.001 * S
    gamma_bump = 0.015 * S
    vol_bump = 1e-4
    rate_bump = 1e-4
    time_bump = 1e-4

    delta = (value_at(S + delta_bump, K, T, r, sigma, q) - value_at(S - delta_bump, K, T, r, sigma, q)) / (
        2 * delta_bump
    )
    gamma = (
        value_at(S + gamma_bump, K, T, r, sigma, q)
        - 2 * value
        + value_at(S - gamma_bump, K, T, r, sigma, q)
    ) / (gamma_bump**2)
    vega = (
        value_at(S, K, T, r, sigma + vol_bump, q) - value_at(S, K, T, r, sigma - vol_bump, q)
    ) / (2 * vol_bump)

    # Rho here is the sensitivity to the interest rate holding the cost of
    # carry (r - q) fixed, i.e. bumping r and q together -- see the module
    # docstring.
    d_dr = (value_at(S, K, T, r + rate_bump, sigma, q) - value_at(S, K, T, r - rate_bump, sigma, q)) / (
        2 * rate_bump
    )
    d_dq = (value_at(S, K, T, r, sigma, q + rate_bump) - value_at(S, K, T, r, sigma, q - rate_bump)) / (
        2 * rate_bump
    )
    rho = d_dr + d_dq

    # Theta here is +dV/d(time-to-maturity) -- see the module docstring
    # for why this is the opposite sign of the GBSM Theta convention.
    theta = (
        value_at(S, K, T + time_bump, r, sigma, q) - value_at(S, K, T - time_bump, r, sigma, q)
    ) / (2 * time_bump)

    return pd.Series([value, delta, gamma, vega, rho, theta], index=_GREEK_COLUMNS)
