"""KAK decomposition and seven-segment exchange reference pulse."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .. import quantum_ops as qops
from ..controls import single_hamiltonian_to_polar_controls
C_MATRIX = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1]], dtype=float)
C_MATRIX_INV = np.linalg.inv(C_MATRIX)
B_MAGIC = 1 / np.sqrt(2) * np.array(
    [[1.0, 0, 0, 1j], [0, 1j, 1.0, 0], [0, 1j, -1.0, 0], [1.0, 0, 0, -1j]],
    dtype=np.complex128,
)
B_MAGIC_DAGGER = B_MAGIC.conj().T

@dataclass
class KAKDecomposition:
    phase_left: complex
    left_1: np.ndarray
    left_2: np.ndarray
    phase_right: complex
    right_1: np.ndarray
    right_2: np.ndarray
    c0: float
    c1: float
    c2: float


def _to_su(unitary: np.ndarray) -> np.ndarray:
    return unitary / np.linalg.det(unitary) ** (1 / unitary.shape[0])


def _decompose_one_qubit_product(unitary: np.ndarray) -> tuple[complex, np.ndarray, np.ndarray]:
    i, j = np.unravel_index(np.argmax(np.abs(unitary), axis=None), unitary.shape)

    def qubit_1_set(index: int) -> tuple[int, int]:
        return (1, 3) if index % 2 else (0, 2)

    def qubit_2_set(index: int) -> tuple[int, int]:
        return (0, 1) if index < 2 else (2, 3)

    u1 = _to_su(unitary[np.ix_(qubit_1_set(i), qubit_1_set(j))])
    u2 = _to_su(unitary[np.ix_(qubit_2_set(i), qubit_2_set(j))])
    phase = unitary[i, j] / (u1[i // 2, j // 2] * u2[i % 2, j % 2])
    return phase, u1, u2


def canonical_unitary(c0: float, c1: float, c2: float) -> np.ndarray:
    return qops.la.expm(0.5j * (c0 * qops.XX + c1 * qops.YY + c2 * qops.ZZ))


def canonical_decomposition(unitary: np.ndarray, rounding: int | None = 14) -> KAKDecomposition:
    unitary = _to_su(np.asarray(unitary, dtype=np.complex128))
    unitary_prime = B_MAGIC_DAGGER @ unitary @ B_MAGIC
    m_squared = unitary_prime.T @ unitary_prime
    if rounding is not None:
        m_squared = np.round(m_squared, rounding)

    eigenvalues, eigenvectors = np.linalg.eig(m_squared)
    if np.isclose(np.linalg.det(eigenvectors), -1):
        eigenvectors[:, 0] *= -1
    k2 = eigenvectors.conj().T
    a_diag = np.sqrt(eigenvalues)
    if np.isclose(np.prod(a_diag), -1):
        a_diag[0] *= -1
    a_matrix = np.diag(a_diag)
    k1 = unitary_prime @ k2.T @ a_matrix.conj().T

    left = B_MAGIC @ k1 @ B_MAGIC_DAGGER
    right = B_MAGIC @ k2 @ B_MAGIC_DAGGER
    phase_left, left_1, left_2 = _decompose_one_qubit_product(left)
    phase_right, right_1, right_2 = _decompose_one_qubit_product(right)

    theta_vec = np.angle(np.diag(a_matrix))[:3]
    a0, a1, a2 = C_MATRIX_INV @ theta_vec
    return KAKDecomposition(
        phase_left=phase_left,
        left_1=left_1,
        left_2=left_2,
        phase_right=phase_right,
        right_1=right_1,
        right_2=right_2,
        c0=float(2 * a1),
        c1=float(-2 * a0),
        c2=float(2 * a2),
    )


def reconstruct_kak(decomposition: KAKDecomposition) -> np.ndarray:
    left = decomposition.phase_left * np.kron(decomposition.left_1, decomposition.left_2)
    right = decomposition.phase_right * np.kron(decomposition.right_1, decomposition.right_2)
    canonical = canonical_unitary(decomposition.c0, decomposition.c1, decomposition.c2)
    return left @ canonical @ right


def exchange_coefficients_from_canonical(c0: float, c1: float, c2: float) -> np.ndarray:
    """Coefficients for the three exchange-like Cartan directions."""

    return np.array(
        [
            0.5 * (c0 + c1 - c2),
            0.5 * (-c0 + c1 + c2),
            0.5 * (c0 - c1 + c2),
        ],
        dtype=float,
    )


def _single_qubit_constant_controls(unitary: np.ndarray, segment_time: float) -> np.ndarray:
    return single_hamiltonian_to_polar_controls(
        qops.constant_hamiltonian_for_unitary(unitary, segment_time)
    )


def _write_kak_exchange_sq_block(
    coefficients: np.ndarray,
    offset: int,
    first_qubit_unitary: np.ndarray,
    second_qubit_unitary: np.ndarray,
    segment_time: float,
) -> None:
    first = _single_qubit_constant_controls(first_qubit_unitary, segment_time)
    second = _single_qubit_constant_controls(second_qubit_unitary, segment_time)
    coefficients[offset] = first[0]
    coefficients[offset + 1] = second[0]
    coefficients[offset + 2] = first[1]
    coefficients[offset + 3] = second[1]
    coefficients[offset + 4] = first[2]
    coefficients[offset + 5] = second[2]


def kak_exchange_coefficients(
    config: dict,
    target: np.ndarray | None = None,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Return 7-segment KAK-exchange physical coefficients with W0 populated."""

    segment_time = float(config.get("segment_time", 10.0e-9))
    num_modes = int(config.get("num_walsh_modes", 2))
    if target is None:
        target = qops.gate_by_name(config.get("target_gate", "CZ"))
    if initial is None:
        initial = qops.gate_by_name(config.get("initial_gate", "II"))
    residual = np.asarray(target, dtype=np.complex128) @ np.asarray(initial, dtype=np.complex128).conj().T
    decomposition = canonical_decomposition(residual)

    h_gate = qops.H_GATE
    convention = str(
        config.get("kak_exchange_basis_convention", "coworker_exchange_compiled")
    ).strip().lower().replace("-", "_")
    if convention in {"coworker", "coworker_exchange_compiled", "exchange_compiled"}:
        # Absorb the local basis changes into the four local segments.
        v_zx = qops.la.expm(-0.25j * np.pi * qops.X)
        middle_1 = h_gate.conj().T
        middle_2 = v_zx.conj().T @ h_gate
        left_frame = v_zx
    else:
        raise ValueError(
            "Only the exchange_compiled reference convention is supported"
        )
    b1, b2, b3 = exchange_coefficients_from_canonical(
        decomposition.c0,
        decomposition.c1,
        decomposition.c2,
    )

    if config.get("control_model", "kak_exchange_27") != "kak_exchange_27":
        raise ValueError("Only kak_exchange_27 is supported")
    coefficient_dim = 27
    sq_offsets = (0, 7, 14, 21)
    tq_indices = (6, 13, 20)

    row = np.zeros(coefficient_dim, dtype=float)
    _write_kak_exchange_sq_block(row, sq_offsets[0], decomposition.right_1, decomposition.right_2, segment_time)
    row[tq_indices[0]] = -b1 / (2.0 * segment_time)
    _write_kak_exchange_sq_block(row, sq_offsets[1], middle_1, middle_1, segment_time)
    row[tq_indices[1]] = -b2 / (2.0 * segment_time)
    _write_kak_exchange_sq_block(row, sq_offsets[2], middle_2, middle_2, segment_time)
    row[tq_indices[2]] = -b3 / (2.0 * segment_time)
    _write_kak_exchange_sq_block(
        row,
        sq_offsets[3],
        decomposition.left_1 @ left_frame,
        decomposition.left_2 @ left_frame,
        segment_time,
    )

    coefficients = np.zeros((num_modes, coefficient_dim), dtype=float)
    coefficients[0] = row
    return coefficients
