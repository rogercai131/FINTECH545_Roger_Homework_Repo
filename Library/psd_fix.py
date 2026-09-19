"""
Fixing a matrix that should be positive semi-definite (PSD) but isn't —
this can happen to covariance/correlation matrices built from pairwise-
complete data (see covariance.py), where each pairwise entry is
individually valid but the resulting matrix as a whole may not be PSD.

Covers: FINTECH545 Tests 3.1 - 3.4 (near_psd and Higham's nearest-PSD
algorithm, applied to both covariance and correlation matrices).
"""

import numpy as np
import pandas as pd


def _is_correlation_matrix(matrix: np.ndarray) -> bool:
    """A matrix is treated as a correlation matrix if its diagonal is all 1s."""
    return np.allclose(np.diag(matrix), 1.0)


def _to_correlation(matrix: np.ndarray):
    """
    Rescale a covariance matrix into a correlation matrix (unit diagonal).

    Returns
    -------
    (correlation, inv_std) : tuple
        `correlation` is the rescaled matrix. `inv_std` is 1/standard
        deviation for each variable (needed to rescale back to a
        covariance matrix later), or None if `matrix` was already a
        correlation matrix (no rescaling needed).
    """
    if _is_correlation_matrix(matrix):
        return matrix, None
    inv_std = 1.0 / np.sqrt(np.diag(matrix))
    correlation = np.outer(inv_std, inv_std) * matrix
    return correlation, inv_std


def _from_correlation(correlation: np.ndarray, inv_std) -> np.ndarray:
    """Undo `_to_correlation`: rescale a fixed correlation matrix back to
    the original covariance scale, if it was rescaled in the first place."""
    if inv_std is None:
        return correlation
    std = 1.0 / inv_std
    return np.outer(std, std) * correlation


def _near_psd(matrix: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    Rebonato & Jackel's "near_psd" algorithm: clip negative eigenvalues to
    `epsilon`, then rescale eigenvectors so the resulting matrix keeps a
    unit diagonal (i.e. stays a valid correlation matrix).
    """
    correlation, inv_std = _to_correlation(matrix)

    vals, vecs = np.linalg.eigh(correlation)
    vals = np.maximum(vals, epsilon)

    # Rescale so that diag(B @ B.T) == 1 (T undoes the norm change from
    # clipping the eigenvalues), then reconstruct the PSD matrix.
    t = 1.0 / ((vecs * vecs) @ vals)
    scale_t = np.diag(np.sqrt(t))
    scale_l = np.diag(np.sqrt(vals))
    b = scale_t @ vecs @ scale_l
    fixed_correlation = b @ b.T

    return _from_correlation(fixed_correlation, inv_std)


def _higham_near_psd(
    matrix: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-9
) -> np.ndarray:
    """
    Higham's alternating-projections algorithm for the nearest PSD matrix
    (in Frobenius norm): repeatedly project onto the set of PSD matrices,
    then onto the set of matrices with unit diagonal, until convergence.
    """
    correlation, inv_std = _to_correlation(matrix)

    y = correlation.copy()
    y0 = y.copy()
    delta_s = np.zeros_like(y)
    prev_gamma = np.inf

    for _ in range(max_iterations):
        r = y - delta_s

        # Project onto the nearest symmetric PSD matrix (clip negative eigenvalues).
        vals, vecs = np.linalg.eigh(r)
        vals = np.maximum(vals, 0.0)
        x = vecs @ np.diag(vals) @ vecs.T

        delta_s = x - r

        # Project onto the set of matrices with a unit diagonal.
        y = x.copy()
        np.fill_diagonal(y, 1.0)

        gamma = np.linalg.norm(y - y0, ord="fro")
        if abs(gamma - prev_gamma) < tolerance:
            break
        prev_gamma = gamma

    return _from_correlation(y, inv_std)


def fix_psd_matrix(
    matrix: pd.DataFrame,
    method: str = "near_psd",
    epsilon: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Repair a covariance or correlation matrix that is supposed to be
    positive semi-definite (PSD) but, due to numerical issues (e.g. it was
    built from pairwise-complete data), has small negative eigenvalues.

    Expected input
    ---------------
    matrix : pd.DataFrame
        A square, symmetric covariance or correlation matrix. Whether it is
        a covariance or correlation matrix is auto-detected from its
        diagonal (all 1s => correlation matrix); covariance matrices are
        internally rescaled to correlation, fixed, then rescaled back so
        the algorithm always operates on a proper correlation matrix.
    method : str, default "near_psd"
        Which repair algorithm to use:
          - "near_psd": Rebonato & Jackel's eigenvalue-clipping method.
            Fast, one-shot, but only an approximate fix (not the closest
            possible PSD matrix).
          - "higham": Higham's alternating-projections method. Iterative,
            slower, but converges to the true nearest PSD matrix in
            Frobenius norm.
    epsilon : float, default 0.0
        Only used by "near_psd" — the floor applied to eigenvalues
        (eigenvalues below this are clipped up to it).
    max_iterations : int, default 100
        Only used by "higham" — maximum number of alternating-projection
        iterations before giving up.
    tolerance : float, default 1e-9
        Only used by "higham" — convergence tolerance on successive
        Frobenius-norm changes.

    Output
    ------
    pd.DataFrame
        A PSD matrix of the same shape/labels as `matrix`.

    Raises
    ------
    ValueError
        If `method` is not one of the supported options.
    """
    if method not in ("near_psd", "higham"):
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: 'near_psd', 'higham'."
        )

    labels = matrix.columns
    values = matrix.values

    if method == "near_psd":
        fixed = _near_psd(values, epsilon=epsilon)
    else:
        fixed = _higham_near_psd(
            values, max_iterations=max_iterations, tolerance=tolerance
        )

    return pd.DataFrame(fixed, index=labels, columns=labels)
