"""Physical controls used by the seven-segment pulse."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import quantum_ops as qops

@dataclass(frozen=True)
class TwoQubitControlBounds:
    alpha_max: float
    gamma_max: float
    gamma_phase_max: float
    g_eff_max: float

    @classmethod
    def from_config(cls, config: dict, final_time: float) -> "TwoQubitControlBounds":
        pi_freq = np.pi / final_time
        return cls(
            alpha_max=float(config.get("alpha_max", pi_freq / 2)),
            gamma_max=float(config.get("gamma_max", pi_freq / 2)),
            gamma_phase_max=float(config.get("gamma_phase_max", np.pi)),
            g_eff_max=float(config.get("g_eff_max", pi_freq / 2)),
        )


def single_hamiltonian_to_polar_controls(hamiltonian: np.ndarray) -> np.ndarray:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    alpha = float(np.real(np.trace(hamiltonian @ qops.Z) / 2.0))
    x_component = float(np.real(np.trace(hamiltonian @ qops.X) / 2.0))
    y_component = float(np.real(np.trace(hamiltonian @ qops.Y) / 2.0))
    gamma = float(np.hypot(x_component, y_component))
    phi = float(np.arctan2(y_component, x_component)) if gamma > 0 else 0.0
    return np.array([alpha, gamma, phi], dtype=float)


def two_qubit_local_hamiltonian(
    alpha1: float,
    gamma1: float,
    phi1: float,
    alpha2: float,
    gamma2: float,
    phi2: float,
    detuning1: float = 0.0,
    detuning2: float = 0.0,
) -> np.ndarray:
    local_1 = (detuning1 + alpha1) * qops.Z1 + gamma1 * (np.cos(phi1) * qops.X1 + np.sin(phi1) * qops.Y1)
    local_2 = (detuning2 + alpha2) * qops.Z2 + gamma2 * (np.cos(phi2) * qops.X2 + np.sin(phi2) * qops.Y2)
    return local_1 + local_2
