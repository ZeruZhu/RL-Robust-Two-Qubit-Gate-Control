"""Full Walsh basis utilities.

Rows are generated on a dyadic grid and then sorted by sequency, i.e. by the
number of sign changes in the row. This avoids relying on fragile bit-order
conventions while making the intended ordering explicit and testable.
"""

from __future__ import annotations

import numpy as np


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _hadamard_order_matrix(n_bins: int) -> np.ndarray:
    if not is_power_of_two(n_bins):
        raise ValueError(f"n_bins must be a positive power of two, got {n_bins}")
    matrix = np.array([[1.0]])
    while matrix.shape[0] < n_bins:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def sequency(row: np.ndarray) -> int:
    signs = np.asarray(row)
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def sequency_order(n_bins: int) -> np.ndarray:
    """Return Hadamard-row indices sorted by Walsh sequency."""

    matrix = _hadamard_order_matrix(n_bins)
    return np.array(sorted(range(n_bins), key=lambda idx: (sequency(matrix[idx]), idx)))


def walsh_matrix(n_bins: int, order: str = "sequency") -> np.ndarray:
    """Return a square Walsh matrix with rows ordered as requested."""

    matrix = _hadamard_order_matrix(n_bins)
    if order == "hadamard":
        return matrix
    if order != "sequency":
        raise ValueError(f"unsupported Walsh order {order!r}")
    return matrix[sequency_order(n_bins)]


def walsh_basis(num_modes: int, n_bins: int, order: str = "sequency") -> np.ndarray:
    """Return the first ``num_modes`` Walsh rows on ``n_bins`` time bins."""

    if num_modes < 1:
        raise ValueError("num_modes must be positive")
    if num_modes > n_bins:
        raise ValueError("num_modes cannot exceed n_bins")
    return walsh_matrix(n_bins, order=order)[:num_modes]


def reconstruct_piecewise(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Combine coefficient rows with Walsh rows into time-bin values.

    Parameters
    ----------
    coefficients:
        Array with shape ``(num_modes, action_dim)``.
    basis:
        Walsh basis with shape ``(num_modes, n_bins)``.
    """

    coefficients = np.asarray(coefficients, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape (num_modes, action_dim)")
    if basis.ndim != 2:
        raise ValueError("basis must have shape (num_modes, n_bins)")
    if coefficients.shape[0] != basis.shape[0]:
        raise ValueError("coefficient and basis mode counts differ")
    return basis.T @ coefficients
