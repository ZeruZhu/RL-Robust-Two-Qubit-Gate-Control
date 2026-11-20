"""Quantum linear-algebra helpers used by the clean reproduction."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H_GATE = (X + Z) / np.sqrt(2)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
S_DAGGER = S_GATE.conj().T
SIGMA_MINUS = np.array([[0, 0], [1, 0]], dtype=np.complex128)

II = np.eye(4, dtype=np.complex128)
X1 = np.kron(X, I2)
Y1 = np.kron(Y, I2)
Z1 = np.kron(Z, I2)
X2 = np.kron(I2, X)
Y2 = np.kron(I2, Y)
Z2 = np.kron(I2, Z)
XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)
EXCHANGE_XX_YY = XX + YY
EXCHANGE_YY_ZZ = YY + ZZ
EXCHANGE_ZZ_XX = ZZ + XX


def cnot() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )


def cz() -> np.ndarray:
    return np.diag([1, 1, 1, -1]).astype(np.complex128)


def iswap() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    )


def gate_by_name(name: str) -> np.ndarray:
    key = name.lower()
    gates = {
        "i": I2,
        "id": I2,
        "identity": I2,
        "x": X,
        "y": Y,
        "z": Z,
        "h": H_GATE,
        "s": S_GATE,
        "ii": II,
        "cnot": cnot(),
        "cz": cz(),
        "iswap": iswap(),
    }
    if key not in gates:
        raise ValueError(f"unknown gate {name!r}")
    return gates[key].copy()


def haar_su(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    phases = np.diag(r) / np.abs(np.diag(r))
    q = q @ np.diag(phases)
    return q / np.linalg.det(q) ** (1 / n)


def encode_complex_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    return np.concatenate([matrix.real.reshape(-1), matrix.imag.reshape(-1)]).astype(np.float64)


def encode_complex_matrix_magnitude_phase(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    flat = matrix.reshape(-1)
    magnitudes = np.abs(flat)
    phases = (np.angle(flat) / (2.0 * np.pi) + 1.0) / 2.0
    return np.column_stack([magnitudes, phases]).reshape(-1).astype(np.float64)


def unitary_fidelity(target: np.ndarray, actual: np.ndarray) -> float:
    dim = target.shape[0]
    return float(abs(np.trace(target.conj().T @ actual)) / dim)


def process_trace_fidelity(target_super: np.ndarray, actual_super: np.ndarray) -> float:
    dim_super = target_super.shape[0]
    return float(abs(np.trace(target_super.conj().T @ actual_super)) / dim_super)


def average_gate_fidelity_from_process_trace(process_trace_fidelity_value: float, hilbert_dim: int) -> float:
    return float((hilbert_dim * process_trace_fidelity_value + 1.0) / (hilbert_dim + 1.0))


def average_gate_fidelity_from_unitary_fidelity(unitary_fidelity_value: float, hilbert_dim: int) -> float:
    return float((hilbert_dim * unitary_fidelity_value**2 + 1.0) / (hilbert_dim + 1.0))


def unitary_to_superoperator(unitary: np.ndarray) -> np.ndarray:
    return np.kron(unitary.conj(), unitary)


def liouvillian(hamiltonian: np.ndarray, jump_ops: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    dim = hamiltonian.shape[0]
    eye = np.eye(dim, dtype=np.complex128)
    generator = -1j * (np.kron(eye, hamiltonian) - np.kron(hamiltonian.T, eye))
    for jump in jump_ops:
        jump = np.asarray(jump, dtype=np.complex128)
        jump_dag_jump = jump.conj().T @ jump
        generator += np.kron(jump.conj(), jump)
        generator -= 0.5 * np.kron(eye, jump_dag_jump)
        generator -= 0.5 * np.kron(jump_dag_jump.T, eye)
    return generator


def propagate_unitary(hamiltonians: list[np.ndarray], total_time: float) -> np.ndarray:
    if not hamiltonians:
        raise ValueError("at least one Hamiltonian bin is required")
    dim = hamiltonians[0].shape[0]
    dt = total_time / len(hamiltonians)
    unitary = np.eye(dim, dtype=np.complex128)
    for hamiltonian in hamiltonians:
        unitary = la.expm(-1j * dt * hamiltonian) @ unitary
    return unitary


def propagate_superoperator(
    hamiltonians: list[np.ndarray],
    total_time: float,
    jump_ops: list[np.ndarray] | tuple[np.ndarray, ...],
) -> np.ndarray:
    if not hamiltonians:
        raise ValueError("at least one Hamiltonian bin is required")
    dim = hamiltonians[0].shape[0]
    dt = total_time / len(hamiltonians)
    superop = np.eye(dim * dim, dtype=np.complex128)
    for hamiltonian in hamiltonians:
        superop = la.expm(dt * liouvillian(hamiltonian, jump_ops)) @ superop
    return superop


def remove_global_phase(unitary: np.ndarray) -> np.ndarray:
    dim = unitary.shape[0]
    return unitary / np.linalg.det(unitary) ** (1 / dim)


def constant_hamiltonian_for_unitary(unitary: np.ndarray, total_time: float) -> np.ndarray:
    su_unitary = remove_global_phase(unitary)
    hamiltonian = 1j * la.logm(su_unitary) / total_time
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    identity_part = np.trace(hamiltonian) / hamiltonian.shape[0]
    return hamiltonian - identity_part * np.eye(hamiltonian.shape[0], dtype=np.complex128)
