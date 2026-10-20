"""
Aggregator module for the risk-calculation library.

Import everything you need from here for convenience:
    from Risk_Library import covariance_matrix

Or, equivalently, import directly from the module that defines it:
    from covariance import covariance_matrix

Both work as long as this `Library/` folder is on `sys.path` (or you are
running from inside it) — the two import styles just give two ways to reach
the same functions.

When adding a new function to the library, add it both to the imports
below and to `__all__` at the bottom of this file. `__all__` is the
explicit list of public functions: it's what `from Risk_Library import *`
brings in, and what editors like VS Code use for autocomplete suggestions.
"""

from covariance import covariance_matrix, ew_covariance_matrix, kendall_correlation_matrix
from psd_fix import fix_psd_matrix, classify_psd
from cholesky import cholesky_psd
from simulation import simulate_normal
from returns import return_calculate, price_from_returns, expected_price_moments
from moments import first_four_moments
from distributions import (
    fit_normal_distribution,
    fit_t_distribution,
    fit_t_regression,
    t_distribution_aicc,
    fit_nig_distribution,
)
from risk_metrics import calculate_var, calculate_es
from portfolio_risk import calculate_portfolio_var_es
from portfolio_optimization import calculate_risk_parity_weights, calculate_max_sharpe_weights
from attribution import return_attribution
from option_pricing import price_european_option, price_american_option
from multivariate_t import fit_multivariate_t
from copula import (
    gaussian_copula_log_likelihood,
    fit_t_copula,
    compare_copulas,
    t_copula_tail_dependence,
    simulate_t_copula_portfolio_var_es,
    simulate_copula,
)

__all__ = [
    "covariance_matrix",
    "ew_covariance_matrix",
    "kendall_correlation_matrix",
    "fix_psd_matrix",
    "classify_psd",
    "cholesky_psd",
    "simulate_normal",
    "return_calculate",
    "price_from_returns",
    "expected_price_moments",
    "first_four_moments",
    "fit_normal_distribution",
    "fit_t_distribution",
    "fit_t_regression",
    "t_distribution_aicc",
    "fit_nig_distribution",
    "calculate_var",
    "calculate_es",
    "calculate_portfolio_var_es",
    "calculate_risk_parity_weights",
    "calculate_max_sharpe_weights",
    "return_attribution",
    "price_european_option",
    "price_american_option",
    "fit_multivariate_t",
    "gaussian_copula_log_likelihood",
    "fit_t_copula",
    "compare_copulas",
    "t_copula_tail_dependence",
    "simulate_t_copula_portfolio_var_es",
    "simulate_copula",
]
