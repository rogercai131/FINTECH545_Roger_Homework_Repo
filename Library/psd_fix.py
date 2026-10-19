"""
Checking and fixing a matrix that should be positive semi-definite (PSD) but
might not be — this can happen to covariance/correlation matrices built from
pairwise-complete data (see covariance.py), where each pairwise entry is
individually valid but the resulting matrix as a whole may not be PSD.

Covers: FINTECH545 Tests 3.1 - 3.4 (near_psd and Higham's nearest-PSD
algorithm, applied to both covariance and correlation matrices), plus
`classify_psd`, a diagnostic that reports whether a matrix is PD / PSD /
non-PSD (and why) without changing it.
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


def _validate_square_matrix(matrix) -> np.ndarray:
    """Convert `matrix` to a float numpy array, checking that it is a
    non-empty, square, finite, real-valued 2-D matrix."""
    values = matrix.values if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)

    # dtype "kind": b = bool, i/u = (unsigned) integer, f = float. Complex
    # ("c") and object/string dtypes are rejected as not real-valued.
    if values.dtype.kind not in "biuf":
        raise TypeError(f"Matrix must be real-valued numeric, got dtype {values.dtype}.")
    if values.ndim != 2:
        raise ValueError(f"Matrix must be 2-D, got {values.ndim}-D input.")
    if values.size == 0:
        raise ValueError("Matrix must be non-empty.")
    if values.shape[0] != values.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Matrix must contain only finite values (no NaN or inf).")

    return values.astype(float)


def classify_psd(
    matrix,
    atol: float = 1e-12,
    rtol: float = 1e-8,
    asymmetry_strategy: str = "report",
) -> dict:
    """
    Classify a real square matrix as positive definite ("PD"), positive
    semi-definite ("PSD"), or "non-PSD", and report the eigenvalue evidence
    behind that call. The input matrix is never modified or repaired — use
    `fix_psd_matrix` for that.

    IMPORTANT: this is a *numerical* classification. Eigenvalues computed in
    floating point carry rounding error, so anything within the tolerance
    band [-threshold, +threshold] is treated as zero. The result therefore
    depends on `atol`/`rtol`, and a "PSD" verdict means "PSD to within this
    tolerance" — it does not prove the matrix is mathematically PSD (a tiny
    true negative eigenvalue inside the band would be hidden, and a matrix
    that is exactly PD but nearly singular may be reported as only PSD).

    How it decides
    --------------
    1. Symmetry check: the matrix is symmetric if
       max|A - A^T| <= atol + rtol * max|A|.
    2. Eigenvalues come from `numpy.linalg.eigvalsh` (the symmetric routine,
       returned in ascending order).
    3. Scale-aware threshold: threshold = atol + rtol * max|eigenvalue|, so
       the same rtol works for a covariance matrix of daily returns (~1e-4)
       or a matrix of dollar exposures (~1e6).
    4. Classification:
         - "non-PSD": some eigenvalue < -threshold.
         - "PD":      every eigenvalue > +threshold.
         - "PSD":     no eigenvalue < -threshold, but at least one lies
                      within threshold of zero (singular, to tolerance).
       Eigenvalues in [-threshold, 0) are "small numerical negatives": they
       are treated as 0 for the classification only, and reported with
       both their original and adjusted (0) values.

    Expected input
    ---------------
    matrix : pd.DataFrame or 2-D array-like
        A non-empty, square, finite, real-valued matrix.
    atol : float, default 1e-12
        Absolute part of the tolerance (matters mostly for matrices whose
        eigenvalues are all near zero, e.g. the zero matrix).
    rtol : float, default 1e-8
        Relative part of the tolerance, scaled by the largest |eigenvalue|
        (or the largest |entry|, for the symmetry check).
    asymmetry_strategy : str, default "report"
        What to do if the matrix is not symmetric within tolerance:
          - "report": do not classify it. The result has classification
            "non-symmetric", eigenvalue fields set to None, and the
            summary states the size of the asymmetry.
          - "symmetrize": classify the symmetric part (A + A^T) / 2
            instead, and say so explicitly in the result
            (`symmetrized=True`) and in the summary.

    Output
    ------
    dict with keys:
      - "classification": "PD", "PSD", "non-PSD", or "non-symmetric".
      - "summary": a one-line readable explanation of the evidence.
      - "is_symmetric": bool, whether the input passed the symmetry check.
      - "max_asymmetry": float, max|A - A^T| of the input.
      - "symmetrized": bool, whether the symmetric part was classified.
      - "eigenvalues": np.ndarray, the original eigenvalues (ascending).
      - "adjusted_eigenvalues": np.ndarray, the same with small numerical
        negatives set to 0 (larger negatives are left unchanged).
      - "threshold": float, the effective tolerance atol + rtol * max|eig|.
      - "small_negative_indices" / "small_negative_values": positions (into
        the ascending "eigenvalues" array) and original values of the
        eigenvalues in [-threshold, 0) that were treated as zero.
      - "negative_indices" / "negative_values": positions and values of
        eigenvalues < -threshold (the evidence for "non-PSD").
      - "min_eigenvalue" / "max_eigenvalue": float.
      For a "non-symmetric" result, every eigenvalue-related field is None.

    Raises
    ------
    TypeError
        If the matrix is not real-valued numeric.
    ValueError
        If the matrix is empty, not 2-D, not square, or has NaN/inf; if
        `atol`/`rtol` are negative or not finite; or if
        `asymmetry_strategy` is not a supported option.

    Examples
    --------
    >>> classify_psd([[2.0, 1.0], [1.0, 2.0]])["summary"]          # PD
    'PD: all 2 eigenvalues are above +threshold (3e-08); min eigenvalue = 1.'
    >>> classify_psd([[1.0, 1.0], [1.0, 1.0]])["summary"]          # PSD, singular
    'PSD: all eigenvalues are above -threshold (2e-08); 1 eigenvalue(s) within threshold of zero; 0 small negative eigenvalue(s) were treated as zero.'
    >>> v = np.array([1.0, 2.0, 3.0])
    >>> r = classify_psd(np.outer(v, v) - 1e-12 * np.eye(3))       # rounding error
    >>> r["summary"]
    'PSD: all eigenvalues are above -threshold (1.4e-07); 2 eigenvalue(s) within threshold of zero; 2 small negative eigenvalue(s) were treated as zero.'
    >>> bool(np.all(r["adjusted_eigenvalues"][r["small_negative_indices"]] == 0))
    True
    >>> classify_psd([[1.0, 2.0], [2.0, 1.0]])["summary"]          # non-PSD
    'non-PSD: 1 eigenvalue(s) below -threshold (3e-08): [-1].'
    """
    if asymmetry_strategy not in ("report", "symmetrize"):
        raise ValueError(
            f"Unsupported asymmetry_strategy '{asymmetry_strategy}'. "
            "Expected one of: 'report', 'symmetrize'."
        )
    for name, tol in (("atol", atol), ("rtol", rtol)):
        if not np.isfinite(tol) or tol < 0:
            raise ValueError(f"{name} must be a finite, non-negative number, got {tol}.")

    values = _validate_square_matrix(matrix)

    # --- 1. Symmetry check (on the input exactly as given) ---
    max_asymmetry = float(np.max(np.abs(values - values.T)))
    symmetry_threshold = atol + rtol * float(np.max(np.abs(values)))
    is_symmetric = max_asymmetry <= symmetry_threshold

    result = {
        "classification": None,
        "summary": None,
        "is_symmetric": is_symmetric,
        "max_asymmetry": max_asymmetry,
        "symmetrized": False,
        "eigenvalues": None,
        "adjusted_eigenvalues": None,
        "threshold": None,
        "small_negative_indices": None,
        "small_negative_values": None,
        "negative_indices": None,
        "negative_values": None,
        "min_eigenvalue": None,
        "max_eigenvalue": None,
    }

    if not is_symmetric:
        if asymmetry_strategy == "report":
            # eigvalsh would silently read only one triangle of a
            # non-symmetric matrix, so refuse to classify instead.
            result["classification"] = "non-symmetric"
            result["summary"] = (
                f"non-symmetric: max|A - A^T| = {max_asymmetry:.3g} exceeds the "
                f"symmetry tolerance {symmetry_threshold:.3g}; not classified. "
                "Pass asymmetry_strategy='symmetrize' to classify (A + A^T)/2."
            )
            return result
        # Explicit, reported symmetrization (a copy — the input is untouched).
        values = (values + values.T) / 2.0
        result["symmetrized"] = True

    # --- 2. Eigenvalues (ascending) and scale-aware threshold ---
    eigenvalues = np.linalg.eigvalsh(values)
    threshold = atol + rtol * float(np.max(np.abs(eigenvalues)))

    # --- 3. Sort eigenvalues into bands relative to the threshold ---
    negative_idx = np.flatnonzero(eigenvalues < -threshold)
    small_negative_idx = np.flatnonzero((eigenvalues >= -threshold) & (eigenvalues < 0))
    near_zero_count = int(np.sum(np.abs(eigenvalues) <= threshold))

    # Small numerical negatives -> 0 for classification/reporting only.
    adjusted = eigenvalues.copy()
    adjusted[small_negative_idx] = 0.0

    # --- 4. Classify ---
    if negative_idx.size > 0:
        classification = "non-PSD"
        summary = (
            f"non-PSD: {negative_idx.size} eigenvalue(s) below -threshold "
            f"({threshold:.3g}): [{', '.join(f'{x:.4g}' for x in eigenvalues[negative_idx])}]."
        )
        if small_negative_idx.size > 0:
            summary += f" ({small_negative_idx.size} further small negative(s) within tolerance.)"
    elif np.all(eigenvalues > threshold):
        classification = "PD"
        summary = (
            f"PD: all {eigenvalues.size} eigenvalues are above +threshold "
            f"({threshold:.3g}); min eigenvalue = {eigenvalues[0]:.4g}."
        )
    else:
        classification = "PSD"
        summary = (
            f"PSD: all eigenvalues are above -threshold ({threshold:.3g}); "
            f"{near_zero_count} eigenvalue(s) within threshold of zero; "
            f"{small_negative_idx.size} small negative eigenvalue(s) were treated as zero."
        )

    if result["symmetrized"]:
        summary = (
            f"[input was NOT symmetric (max|A - A^T| = {max_asymmetry:.3g}); "
            f"classified its symmetric part (A + A^T)/2] " + summary
        )

    result.update(
        classification=classification,
        summary=summary,
        eigenvalues=eigenvalues,
        adjusted_eigenvalues=adjusted,
        threshold=threshold,
        small_negative_indices=small_negative_idx,
        small_negative_values=eigenvalues[small_negative_idx],
        negative_indices=negative_idx,
        negative_values=eigenvalues[negative_idx],
        min_eigenvalue=float(eigenvalues[0]),
        max_eigenvalue=float(eigenvalues[-1]),
    )
    return result
