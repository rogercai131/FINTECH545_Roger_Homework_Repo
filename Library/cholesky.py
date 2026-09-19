"""
Cholesky factorization for positive semi-definite (PSD) matrices.

Unlike a standard Cholesky routine (e.g. numpy.linalg.cholesky), this
implementation tolerates matrices that are only positive *semi*-definite
(i.e. have zero eigenvalues / a variable with zero variance) rather than
raising an error — any column that ends up with a zero pivot is simply
given an all-zero row in the factor.

Covers: FINTECH545 Test 4.1 (chol_psd).
"""

import numpy as np
import pandas as pd


def cholesky_psd(matrix: pd.DataFrame, tolerance: float = 1e-8) -> pd.DataFrame:
    """
    Compute the lower-triangular Cholesky factor `L` of a PSD matrix `a`,
    such that `L @ L.T == a`.

    Expected input
    ---------------
    matrix : pd.DataFrame
        A square, symmetric, positive semi-definite covariance or
        correlation matrix (e.g. the output of `psd_fix.fix_psd_matrix`).
    tolerance : float, default 1e-8
        Numerical tolerance used when a diagonal entry works out to a
        small negative number purely from floating-point error (values in
        `[-tolerance, 0]` are treated as exactly 0 rather than raising an
        error from taking the square root of a negative number).

    Output
    ------
    pd.DataFrame
        The lower-triangular Cholesky factor, same shape/labels as
        `matrix`. Rows/columns corresponding to a zero eigenvalue
        (a redundant/zero-variance variable) come out as all zeros.
    """
    labels = matrix.columns
    a = matrix.values
    n = a.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        # Subtract off what's already been "explained" by earlier columns
        # of this row before taking the pivot for column j.
        s = root[j, :j] @ root[j, :j] if j > 0 else 0.0
        pivot = a[j, j] - s

        if -tolerance <= pivot <= 0:
            pivot = 0.0
        root[j, j] = np.sqrt(pivot)

        # A zero pivot means this variable is fully explained by the
        # earlier ones (zero variance left over) — leave the rest of the
        # column as zeros and move on.
        if root[j, j] == 0.0:
            continue

        inv_pivot = 1.0 / root[j, j]
        for i in range(j + 1, n):
            s = root[i, :j] @ root[j, :j] if j > 0 else 0.0
            root[i, j] = (a[i, j] - s) * inv_pivot

    return pd.DataFrame(root, index=labels, columns=labels)
