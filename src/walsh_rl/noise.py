"""Quasi-static detuning and Markovian dissipative noise."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import quantum_ops as qops

@dataclass
class NoiseSample:
    detuning: np.ndarray
    relaxation_rates: np.ndarray
    dephasing_rates: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.detuning, self.relaxation_rates, self.dephasing_rates]).astype(float)


class MarkovianNoiseModel:
    def __init__(self, n_qubits: int, config: dict | None = None):
        self.n_qubits = n_qubits
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.detuning_std = float(self.config.get("detuning_std", 0.0))
        self.detuning_values = self.config.get("detuning_values")
        configured_detuning_scale = self.config.get("detuning_observation_scale")
        self.detuning_scale = (
            float(configured_detuning_scale)
            if configured_detuning_scale is not None
            else max(abs(self.detuning_std), 1.0)
        )
        if self.detuning_values is not None:
            values = np.asarray(self.detuning_values, dtype=float)
            if values.ndim == 1:
                self.detuning_scale = max(float(np.max(np.abs(values))), self.detuning_scale)
            elif values.ndim == 2 and values.shape[0] == n_qubits:
                self.detuning_scale = max(float(np.max(np.abs(values))), self.detuning_scale)
            else:
                raise ValueError("detuning_values must be a 1D list or a 2D list with one row per qubit")
        self.relaxation_rates = np.asarray(
            self.config.get("relaxation_rates", [0.0] * n_qubits), dtype=float
        )
        self.dephasing_rates = np.asarray(
            self.config.get("dephasing_rates", [0.0] * n_qubits), dtype=float
        )
        if self.relaxation_rates.shape != (n_qubits,):
            raise ValueError("relaxation_rates must contain one value per qubit")
        if self.dephasing_rates.shape != (n_qubits,):
            raise ValueError("dephasing_rates must contain one value per qubit")

    @classmethod
    def disabled(cls, n_qubits: int) -> "MarkovianNoiseModel":
        return cls(n_qubits, {"enabled": False})

    def sample(self, rng: np.random.Generator) -> NoiseSample:
        if not self.enabled:
            zeros = np.zeros(self.n_qubits, dtype=float)
            return NoiseSample(zeros.copy(), zeros.copy(), zeros.copy())
        if self.detuning_values is not None:
            values = np.asarray(self.detuning_values, dtype=float)
            if values.ndim == 1:
                detuning = rng.choice(values, size=self.n_qubits)
            else:
                detuning = np.array([rng.choice(values[idx]) for idx in range(self.n_qubits)], dtype=float)
        else:
            detuning = rng.normal(0.0, self.detuning_std, size=self.n_qubits)
        return NoiseSample(detuning, self.relaxation_rates.copy(), self.dephasing_rates.copy())

    def normalized_vector(self, sample: NoiseSample) -> np.ndarray:
        rate_scale = max(
            float(np.max(np.abs(self.relaxation_rates))) if self.relaxation_rates.size else 0.0,
            float(np.max(np.abs(self.dephasing_rates))) if self.dephasing_rates.size else 0.0,
            1.0,
        )
        return np.concatenate(
            [
                np.clip(sample.detuning / self.detuning_scale, -1.0, 1.0),
                np.clip(sample.relaxation_rates / rate_scale, -1.0, 1.0),
                np.clip(sample.dephasing_rates / rate_scale, -1.0, 1.0),
            ]
        ).astype(np.float64)

    def normalized_rate_vector(self, sample: NoiseSample) -> np.ndarray:
        return self.normalized_vector(sample)[self.n_qubits :]

    def normalized_std_vector(self, sample: NoiseSample) -> np.ndarray:
        rate_vector = self.normalized_rate_vector(sample)
        std_vector = np.full(
            self.n_qubits,
            np.clip(self.detuning_std / self.detuning_scale, -1.0, 1.0),
            dtype=np.float64,
        )
        return np.concatenate([std_vector, rate_vector]).astype(np.float64)


def two_qubit_jump_ops(sample: NoiseSample) -> list[np.ndarray]:
    sm1 = np.kron(qops.SIGMA_MINUS, qops.I2)
    sm2 = np.kron(qops.I2, qops.SIGMA_MINUS)
    return [
        np.sqrt(max(sample.relaxation_rates[0], 0.0)) * sm1,
        np.sqrt(max(sample.relaxation_rates[1], 0.0)) * sm2,
        np.sqrt(max(sample.dephasing_rates[0], 0.0)) * qops.Z1,
        np.sqrt(max(sample.dephasing_rates[1], 0.0)) * qops.Z2,
    ]
