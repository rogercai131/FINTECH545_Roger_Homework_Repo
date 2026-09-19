"""
Monte Carlo simulation of multivariate normal data from a covariance
matrix.

Covers: FINTECH545 Tests 5.1 - 5.5 (direct simulation from PD/PSD/non-PSD
covariance inputs, with optional near_psd/Higham repair, plus PCA-based
simulation that keeps only the components needed to explain a target
fraction of the total variance).
"""

import numpy as np
import pandas as pd

from cholesky import cholesky_psd
from psd_fix import fix_psd_matrix


def simulate_normal(
    cov_matrix: pd.DataFrame,
    num_simulations: int = 10000,
    mean=None,
    method: str = "direct",
    psd_fix: str = None,
    explained_variance: float = 1.0,
    seed: int = None,
) -> pd.DataFrame:
    """
    Draw simulated multivariate-normal observations from a covariance
    matrix.

    Expected input
    ---------------
    cov_matrix : pd.DataFrame
        A square covariance matrix. Does not need to be PSD if `psd_fix`
        is used to repair it first.
    num_simulations : int, default 10000
        Number of simulated draws (rows) to generate.
    mean : array-like, optional
        Mean vector to shift the draws by, one entry per variable. Defaults
        to a zero vector (mean-zero simulation).
    method : str, default "direct"
        Simulation approach:
          - "direct": factor the (optionally repaired) covariance matrix
            with `cholesky.cholesky_psd` and apply it to standard-normal
            draws. Uses the full dimensionality of `cov_matrix`.
          - "pca": factor the covariance matrix via its eigen-decomposition
            and keep only the top eigenvectors needed to explain
            `explained_variance` of the total variance, simulating in that
            reduced space and mapping back to the original variables. This
            is faster / uses fewer random draws per simulation when only a
            few components matter, at the cost of some fidelity.
    psd_fix : str, optional
        If the covariance matrix is not PSD, repair it first using
        `psd_fix.fix_psd_matrix` with this method ("near_psd" or
        "higham"). If None (default), the matrix is used as-is (assumed
        already PD/PSD).
    explained_variance : float, default 1.0
        Only used by `method="pca"` — the minimum cumulative fraction of
        total variance (0 < explained_variance <= 1) the retained
        components must explain.
    seed : int, optional
        Seed for the random number generator, for reproducible draws.

    Output
    ------
    pd.DataFrame
        `num_simulations` rows by one column per variable in `cov_matrix`
        (same column labels/order), holding the simulated draws.

    Raises
    ------
    ValueError
        If `method` is not "direct"/"pca".
    """
    if method not in ("direct", "pca"):
        raise ValueError(f"Unsupported method '{method}'. Expected one of: 'direct', 'pca'.")

    labels = cov_matrix.columns
    n = cov_matrix.shape[0]
    mean_vector = np.zeros(n) if mean is None else np.asarray(mean)

    cov = cov_matrix if psd_fix is None else fix_psd_matrix(cov_matrix, method=psd_fix)

    rng = np.random.default_rng(seed)

    if method == "direct":
        factor = cholesky_psd(cov).values
        z = rng.standard_normal((n, num_simulations))
        draws = (factor @ z).T + mean_vector
    else:
        eigvals, eigvecs = np.linalg.eigh(cov.values)

        # eigh returns ascending order; we want largest-variance components first.
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        total_variance = eigvals.sum()
        cumulative_share = np.cumsum(eigvals) / total_variance
        num_components = int(np.searchsorted(cumulative_share, explained_variance) + 1)

        kept_vals = eigvals[:num_components]
        kept_vecs = eigvecs[:, :num_components]
        factor = kept_vecs @ np.diag(np.sqrt(kept_vals))

        z = rng.standard_normal((num_components, num_simulations))
        draws = (factor @ z).T + mean_vector

    return pd.DataFrame(draws, columns=labels)
