"""Reference-frame projection kernels extracted from the research simulator."""
from __future__ import annotations
import numpy as np
from .. import quantum_ops as qops

LOCAL_PROJECTION_NULL_RTOL = 1.0e-12


class ProjectionKernels:
    def _residual_generator(self, operator: np.ndarray | None = None) -> np.ndarray:
        if operator is None and getattr(self, "_residual_generator_cache", None) is not None:
            return self._residual_generator_cache.copy()
        residual = self._residual_operator() if operator is None else self.target_operator @ operator.conj().T
        if self.use_superoperator:
            generator = qops.la.logm(residual) / max(self.final_time, np.finfo(float).eps)
        else:
            residual_unitary = np.asarray(residual, dtype=np.complex128)
            generator = 1j * qops.la.logm(residual_unitary) / max(self.final_time, np.finfo(float).eps)
            generator = 0.5 * (generator + generator.conj().T)
        if operator is None:
            self._residual_generator_cache = generator.copy()
        return generator


    def _projection_values_for_operators(
        self,
        operators: list[np.ndarray] | tuple[np.ndarray, ...],
        operator: np.ndarray | None = None,
    ) -> np.ndarray:
        generator = self._residual_generator(operator)
        if self.use_superoperator:
            basis_generators = [qops.liouvillian(basis_operator, []) for basis_operator in operators]
        else:
            basis_generators = list(operators)

        projections = []
        for basis_generator in basis_generators:
            denominator = np.vdot(basis_generator, basis_generator)
            if abs(denominator) <= 0.0:
                projections.append(0.0)
            else:
                projections.append(float(np.real(np.vdot(basis_generator, generator) / denominator)))
        return np.array(projections, dtype=np.float64)


    @classmethod
    def _local_pauli_component(cls, operator: np.ndarray) -> np.ndarray:
        result = np.zeros_like(operator, dtype=np.complex128)
        for _, pauli in cls._pauli_projection_basis()[:6]:
            denominator = np.vdot(pauli, pauli)
            coefficient = np.vdot(pauli, operator) / denominator
            result += coefficient * pauli
        result = 0.5 * (result + result.conj().T)
        # Purely nonlocal transported axes leave roundoff, not a local direction
        # that can safely be normalized by its own squared norm.
        source_norm_squared = float(np.vdot(operator, operator).real)
        local_norm_squared = float(np.vdot(result, result).real)
        if local_norm_squared <= LOCAL_PROJECTION_NULL_RTOL**2 * source_norm_squared:
            return np.zeros_like(result)
        return result


    def set_toggling_projection_reference_coefficients(
        self,
        coefficients: np.ndarray | None,
    ) -> None:
        self._toggling_projection_reference_coefficients_override = (
            None if coefficients is None else np.asarray(coefficients, dtype=float).copy()
        )


    def _toggling_z_basis_operators(self) -> list[np.ndarray]:
        segments = self._toggling_z_segments()
        if not segments:
            return []
        active_modes = self._active_iterative_basis_count()
        active_basis = self.basis[:active_modes]
        reference_coefficients = self._toggling_projection_reference_coefficients()
        active_coefficients = reference_coefficients[:active_modes]
        cache_key = (
            active_modes,
            segments,
            self._toggling_z_frame_point(),
            active_coefficients.tobytes(),
        )
        if not hasattr(self, "_toggling_z_basis_cache"):
            self._toggling_z_basis_cache = {}
        if cache_key in self._toggling_z_basis_cache:
            return self._toggling_z_basis_cache[cache_key]

        sequence = self._kak_exchange_segment_sequence()
        segment_unitaries = []
        midpoint_unitaries = []
        for segment in sequence:
            hamiltonians = self._build_kak_exchange_control_hamiltonian_bins_for_coefficients(
                segment,
                active_basis,
                active_coefficients,
            )
            segment_unitaries.append(qops.propagate_unitary(hamiltonians, self.segment_time))
            midpoint_unitaries.append(qops.propagate_unitary(hamiltonians, 0.5 * self.segment_time))

        after_segments = [qops.II.copy() for _ in sequence]
        suffix = qops.II.copy()
        for index in range(len(sequence) - 1, -1, -1):
            after_segments[index] = suffix
            suffix = suffix @ segment_unitaries[index]

        operators = []
        for index, segment in enumerate(sequence):
            if segment not in segments:
                continue
            frame = after_segments[index] @ midpoint_unitaries[index]
            for pauli in (qops.Z1, qops.Z2):
                operator = frame @ pauli @ frame.conj().T
                operators.append(0.5 * (operator + operator.conj().T))
        self._toggling_z_basis_cache[cache_key] = operators
        return operators


    def _local_action_toggling_basis_operators(self) -> list[np.ndarray]:
        if not self._include_local_action_toggling_projection_observation():
            return []
        active_modes = self._active_iterative_basis_count()
        active_basis = self.basis[:active_modes]
        reference_coefficients = self._toggling_projection_reference_coefficients()
        active_coefficients = reference_coefficients[:active_modes]
        role = self._local_action_toggling_role()
        cache_key = (
            active_modes,
            role,
            self._toggling_z_frame_point(),
            active_coefficients.tobytes(),
        )
        if not hasattr(self, "_local_action_toggling_basis_cache"):
            self._local_action_toggling_basis_cache = {}
        if cache_key in self._local_action_toggling_basis_cache:
            return [
                value.copy()
                for value in self._local_action_toggling_basis_cache[cache_key]
            ]

        sequence = self._kak_exchange_segment_sequence()
        segment_unitaries = []
        midpoint_unitaries = []
        for segment in sequence:
            hamiltonians = self._build_kak_exchange_control_hamiltonian_bins_for_coefficients(
                segment,
                active_basis,
                active_coefficients,
            )
            segment_unitaries.append(qops.propagate_unitary(hamiltonians, self.segment_time))
            midpoint_unitaries.append(
                qops.propagate_unitary(hamiltonians, 0.5 * self.segment_time)
            )

        after_segments = [qops.II.copy() for _ in sequence]
        suffix = qops.II.copy()
        for index in range(len(sequence) - 1, -1, -1):
            after_segments[index] = suffix
            suffix = suffix @ segment_unitaries[index]

        operators = []
        for index, segment in enumerate(sequence):
            if not segment.startswith("sq"):
                continue
            frame = after_segments[index] @ midpoint_unitaries[index]
            for qubit_index in range(2):
                if role == "x":
                    source = qops.X1 if qubit_index == 0 else qops.X2
                else:
                    source = qops.Y1 if qubit_index == 0 else qops.Y2
                transported = frame @ source @ frame.conj().T
                operators.append(
                    self._local_pauli_component(
                        0.5 * (transported + transported.conj().T)
                    )
                )
        self._local_action_toggling_basis_cache[cache_key] = [
            value.copy() for value in operators
        ]
        return operators
