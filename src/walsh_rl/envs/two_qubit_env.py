"""Seven-segment simulator for the released calibration method.

This backend evaluates complete physical pulses. The actor observation and
accept/reject protocol belong to FidelityOnlyEightAgentEnv, not this backend.
"""
from __future__ import annotations

import numpy as np

from .. import quantum_ops as qops
from ..controls import TwoQubitControlBounds, two_qubit_local_hamiltonian
from ..noise import MarkovianNoiseModel, two_qubit_jump_ops
from ..rewards import log_infidelity_reward
from ..walsh_basis import walsh_basis
from .projection_kernels import LOCAL_PROJECTION_NULL_RTOL, ProjectionKernels


class TwoQubitWalshEnv(ProjectionKernels):
    """Two local controls per qubit and exchange in seven fixed segments."""

    def __init__(self, config: dict):
        self.config = config
        self.control_model = config.get("control_model", "kak_exchange_27")
        if self.control_model != "kak_exchange_27" or config.get("action_mode", "residual") != "residual":
            raise ValueError("Only the seven-segment kak_exchange_27 residual model is supported")
        if config.get("control_error", {}).get("enabled", False):
            raise ValueError("Additional control-error models are outside this artifact")
        self.action_mode = "residual"
        self.segment_time = float(config.get("segment_time", 30e-9))
        if not np.isfinite(self.segment_time) or self.segment_time <= 0:
            raise ValueError("segment_time must be positive and finite")
        self.final_time = 7 * self.segment_time
        self.num_walsh_modes = int(config.get("num_walsh_modes", 2))
        self.num_time_bins = int(config.get("num_time_bins", 2))
        self.coefficient_dim = 27
        self.basis = walsh_basis(self.num_walsh_modes, self.num_time_bins, config.get("walsh_order", "sequency"))
        self.bounds = TwoQubitControlBounds.from_config(config.get("control_bounds", {}), self.segment_time)
        self.target_config = config.get("target_gate", "CNOT")
        self.target_sampling = config.get("target_sampling")
        if isinstance(self.target_config, str) and self.target_config.lower() == "random_su4":
            self.target_sampling = "haar_su4"
        if self.target_sampling not in (None, "haar_su4"):
            raise ValueError("Only a fixed target or haar_su4 sampling is supported")
        self.fixed_target = None if self.target_sampling else self._load_target(self.target_config)
        self.initial = self._load_target(config.get("initial_gate", "II"))
        self.noise_observability = config.get("noise_observability", "hidden")
        if self.noise_observability != "hidden":
            raise ValueError("The released actor uses hidden noise configurations")
        self.noise_model = MarkovianNoiseModel(2, config.get("noise", {}))
        self.use_superoperator = self.noise_model.enabled
        segments = config.get("noise", {}).get("kak_exchange_detuning_segments", "all")
        if segments != "all":
            raise ValueError("The released model applies detuning in all seven segments")
        self.kak_exchange_detuning_segments = frozenset(self._kak_exchange_segment_sequence())
        self.iterative_constant_config = dict(config.get("iterative_constant", {}))
        self.reward_eps = float(config.get("reward_eps", 1e-8))
        self.rng = np.random.default_rng(config.get("seed"))
        self.reset()

    @staticmethod
    def _load_target(value):
        matrix = qops.gate_by_name(value) if isinstance(value, str) else np.asarray(value, dtype=complex)
        if matrix.shape != (4, 4) or not np.allclose(matrix.conj().T @ matrix, qops.II):
            raise ValueError("The initial and target gates must be 4 by 4 unitary matrices")
        return matrix.copy()

    def reset(self, *, seed=None, options=None):
        from ..baselines.two_qubit_kak import kak_exchange_coefficients

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.target = qops.haar_su(4, self.rng) if self.target_sampling else self.fixed_target.copy()
        override = (options or {}).get("reference_coefficients")
        self.base_physical_coefficients = (
            kak_exchange_coefficients(self.config, self.target, self.initial)
            if override is None else np.asarray(override, dtype=float).copy()
        )
        if self.base_physical_coefficients.shape != (self.num_walsh_modes, 27):
            raise ValueError("Invalid reference coefficient shape")
        self.noise_sample = self.noise_model.sample(self.rng)
        convert = qops.unitary_to_superoperator if self.use_superoperator else np.copy
        self.current_operator = convert(self.initial)
        self.target_operator = convert(self.target)
        self.physical_coefficients = np.zeros_like(self.base_physical_coefficients)
        self.control_error_coefficients = np.zeros_like(self.physical_coefficients)
        self.current_step = 0
        self._residual_generator_cache = None
        self._toggling_z_basis_cache = {}
        self._local_action_toggling_basis_cache = {}
        self._toggling_projection_reference_coefficients_override = None
        self.fidelity = self._compute_fidelity()
        self.reward = log_infidelity_reward(self.fidelity, self.reward_eps)
        return np.empty(0), self._get_info()

    @staticmethod
    def _kak_exchange_segment_sequence():
        return ("sq1", "tq1", "sq2", "tq2", "sq3", "tq3", "sq4")

    @staticmethod
    def _kak_exchange_sq_offset(segment):
        return {"sq1": 0, "sq2": 7, "sq3": 14, "sq4": 21}[segment]

    @staticmethod
    def _kak_exchange_tq_index(segment):
        return {"tq1": 6, "tq2": 13, "tq3": 20}[segment]

    def _kak_exchange_local_hamiltonian(self, coeff, segment):
        offset = self._kak_exchange_sq_offset(segment)
        return two_qubit_local_hamiltonian(
            coeff[offset], coeff[offset+2], coeff[offset+4],
            coeff[offset+1], coeff[offset+3], coeff[offset+5])

    def _build_kak_exchange_control_hamiltonian_bins_for_coefficients(self, segment, active_basis, active_coefficients):
        result = []
        for index in range(self.num_time_bins):
            hamiltonian = np.zeros((4, 4), dtype=complex)
            for coeff, basis in zip(active_coefficients, active_basis):
                term = (self._kak_exchange_local_hamiltonian(coeff, segment)
                        if segment.startswith("sq") else coeff[self._kak_exchange_tq_index(segment)] * qops.EXCHANGE_XX_YY)
                hamiltonian += basis[index] * term
            result.append(hamiltonian)
        return result

    def _build_kak_exchange_hamiltonian_bins(self, segment):
        result = self._build_kak_exchange_control_hamiltonian_bins_for_coefficients(
            segment, self.basis[:self.current_step], self.physical_coefficients[:self.current_step])
        if self.noise_model.enabled:
            for hamiltonian in result:
                hamiltonian += self.noise_sample.detuning[0] * qops.Z1
                hamiltonian += self.noise_sample.detuning[1] * qops.Z2
        return result

    def evaluate_physical_coefficients(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (self.num_walsh_modes, 27) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Expected finite physical coefficients with shape (num_walsh_modes, 27)")
        self.physical_coefficients = coefficients.copy()
        self.current_step = self.num_walsh_modes
        self._residual_generator_cache = None
        if self.use_superoperator:
            jumps = two_qubit_jump_ops(self.noise_sample)
            current = qops.unitary_to_superoperator(self.initial)
            for segment in self._kak_exchange_segment_sequence():
                evolution = qops.propagate_superoperator(self._build_kak_exchange_hamiltonian_bins(segment), self.segment_time, jumps)
                current = evolution @ current
        else:
            current = self.initial.copy()
            for segment in self._kak_exchange_segment_sequence():
                evolution = qops.propagate_unitary(self._build_kak_exchange_hamiltonian_bins(segment), self.segment_time)
                current = evolution @ current
        self.current_operator = current
        self.fidelity = self._compute_fidelity()
        self.reward = log_infidelity_reward(self.fidelity, self.reward_eps)
        return self._get_info()

    def _compute_fidelity(self):
        function = qops.process_trace_fidelity if self.use_superoperator else qops.unitary_fidelity
        return function(self.target_operator, self.current_operator)

    def _get_info(self):
        function = (qops.average_gate_fidelity_from_process_trace if self.use_superoperator
                    else qops.average_gate_fidelity_from_unitary_fidelity)
        return dict(fidelity=self.fidelity, average_gate_fidelity=function(self.fidelity, 4),
                    reward=self.reward, current_step=self.current_step,
                    noise_params=self.noise_sample.as_vector(),
                    physical_coefficients=self.physical_coefficients.copy(),
                    base_physical_coefficients=self.base_physical_coefficients.copy(),
                    initial_unitary=self.initial.copy(), target_unitary=self.target.copy())

    def pulse_hamiltonian_bins(self):
        return [h for segment in self._kak_exchange_segment_sequence()
                for h in self._build_kak_exchange_hamiltonian_bins(segment)]

    def _residual_operator(self):
        return self.target_operator @ self.current_operator.conj().T

    @staticmethod
    def _pauli_projection_basis():
        return list(zip(("XI", "YI", "ZI", "IX", "IY", "IZ"),
                        (qops.X1, qops.Y1, qops.Z1, qops.X2, qops.Y2, qops.Z2)))

    def _toggling_z_segments(self):
        return self._kak_exchange_segment_sequence()

    def _active_iterative_basis_count(self):
        # Segment directions use only the constant, uncalibrated reference.
        return 1

    def _toggling_z_frame_point(self):
        return "midpoint"

    def _toggling_projection_reference_coefficients(self):
        value = self._toggling_projection_reference_coefficients_override
        return self.base_physical_coefficients if value is None else value

    def _include_local_action_toggling_projection_observation(self):
        return bool(self.iterative_constant_config.get("include_local_action_toggling_projection_observation", False))

    def _local_action_toggling_role(self):
        role = self.iterative_constant_config.get("local_action_toggling_role", "x")
        if role not in ("x", "y"):
            raise ValueError("Transverse segment directions must be x or y")
        return role

    def _toggling_z_observation_scale(self):
        scale = float(self.iterative_constant_config.get("toggling_z_projection_scale", self.config.get("noise", {}).get("detuning_std", 1)))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("Projection normalization must be positive and finite")
        return scale

    def _scaled_projection(self, operators):
        return self._projection_values_for_operators(operators) * self.final_time / self.segment_time / self._toggling_z_observation_scale()

    def _toggling_z_projection_observation(self):
        if not self.iterative_constant_config.get("include_toggling_z_projection_observation", False):
            return np.empty(0)
        return self._scaled_projection([self._local_pauli_component(op) for op in self._toggling_z_basis_operators()])

    def _local_action_toggling_projection_observation(self):
        if not self._include_local_action_toggling_projection_observation():
            return np.empty(0)
        return self._scaled_projection(self._local_action_toggling_basis_operators())

    def close(self):
        pass
