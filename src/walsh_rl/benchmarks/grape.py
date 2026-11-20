"""Exact-gradient GRAPE for the seven-segment pulse."""
from __future__ import annotations
import time
import numpy as np
from numpy.polynomial.hermite import hermgauss
import scipy.linalg as la
from .. import quantum_ops as qops
from ..controls import TwoQubitControlBounds
from ..noise import NoiseSample, two_qubit_jump_ops
from ..fidelity_only_controls import KAK_EXCHANGE_ALPHA_INDICES as ALPHA, KAK_EXCHANGE_EXCHANGE_INDICES as EXCHANGE, LOCAL_GAMMA_INDICES as GAMMA, LOCAL_PHASE_INDICES as PHASE
SEGMENTS = ("sq1", "tq1", "sq2", "tq2", "sq3", "tq3", "sq4")
CONTROL_NAMES = ("alpha_1", "x_1", "y_1", "alpha_2", "x_2", "y_2", "g_xx_yy")
H_CONTROLS = np.array([
    qops.Z1, qops.X1, qops.Y1, qops.Z2, qops.X2, qops.Y2, qops.EXCHANGE_XX_YY,
])
D_L = np.array([qops.liouvillian(h, []) for h in H_CONTROLS])
TRACE_FIELDS = (
    "iteration", "average_gate_fidelity", "process_trace_fidelity", "infidelity",
    "objective_average_gate_fidelity", "objective_noise_std", "accepted",
    "step_size", "gradient_norm", "objective_evaluations", "gradient_evaluations",
    "optimization_noise_propagations", "evaluation_noise_propagations", "wall_seconds",
)

def gaussian_ensemble(std: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    if std < 0 or order < 1:
        raise ValueError("nonnegative noise std and positive quadrature order required")
    nodes, weights = hermgauss(order)
    nodes = np.sqrt(2.0) * std * nodes
    a, b = np.meshgrid(nodes, nodes, indexing="ij")
    return np.column_stack([a.ravel(), b.ravel()]), np.outer(weights, weights).ravel() / np.pi


class PulseMap:
    """Repeat the deployed two-coefficient feasible set over sub-bin pairs."""

    def __init__(self, base: np.ndarray, env_config: dict, bins: int):
        if bins not in (14, 140):
            raise ValueError("the benchmark supports 14 or 140 total bins")
        if np.asarray(base).shape != (2, 27):
            raise ValueError("expected cached KAK coefficients with shape (2, 27)")
        self.base = np.asarray(base, dtype=float)
        self.pairs = bins // 14
        self.bins = bins
        self.shape = (self.pairs, 2, 27)
        self.bins_per_segment = 2 * self.pairs
        settings = env_config["fidelity_only_eight_agent"]
        self.scales = np.array(
            [settings["alpha_bound"]] * 8 + [settings["g_bound"]] * 3
            + [settings["xy_bound"]] * 16, dtype=float,
        )
        self.base_x = self.base[:, GAMMA] * np.cos(self.base[:, PHASE])
        self.base_y = self.base[:, GAMMA] * np.sin(self.base[:, PHASE])
        bounds = TwoQubitControlBounds.from_config(
            env_config.get("control_bounds", {}), env_config["segment_time"]
        )
        self.gamma_limits = np.maximum(bounds.gamma_max, np.abs(self.base[:, GAMMA]))
        self.slot_indices = []
        for segment in range(7):
            for half in range(2):
                for pair in range(self.pairs):
                    self.slot_indices.append((segment, half, pair))
        self.segment_labels = np.repeat(np.array(SEGMENTS), self.bins_per_segment)

    def _coefficients(self, z: np.ndarray):
        z = np.asarray(z, dtype=float).reshape(self.shape)
        if np.any(~np.isfinite(z)) or np.max(np.abs(z)) > 1 + 1e-12:
            raise ValueError("normalized commands outside [-1,1]")
        command = z * self.scales
        alpha = self.base[None, :, ALPHA] + command[..., :8]
        exchange = self.base[None, :, EXCHANGE] + command[..., 8:11]
        raw_xy = np.stack([
            self.base_x[None] + command[..., 11:19],
            self.base_y[None] + command[..., 19:27],
        ], axis=-1)
        norm = la.norm(raw_xy, axis=-1)
        ratio = np.minimum(1.0, self.gamma_limits[None] / np.maximum(norm, 1e-300))
        xy = raw_xy * ratio[..., None]
        return alpha, exchange, xy, raw_xy, norm, ratio

    def physical(self, z: np.ndarray) -> np.ndarray:
        alpha, exchange, xy, _, _, _ = self._coefficients(z)
        pulse = np.zeros((self.bins, 7))
        for slot, (segment, half, pair) in enumerate(self.slot_indices):
            weights = np.array([1.0, 1.0 if half == 0 else -1.0])
            if segment % 2 == 0:
                local = segment // 2
                for qubit in range(2):
                    index = 2 * local + qubit
                    offset = 3 * qubit
                    pulse[slot, offset] = weights @ alpha[pair, :, index]
                    pulse[slot, offset + 1:offset + 3] = weights @ xy[pair, :, index]
            else:
                pulse[slot, 6] = weights @ exchange[pair, :, segment // 2]
        return pulse

    def pullback(self, z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        _, _, _, raw, norm, ratio = self._coefficients(z)
        result = np.zeros(self.shape)
        xy_gradient = np.zeros_like(raw)
        for slot, (segment, half, pair) in enumerate(self.slot_indices):
            weights = np.array([1.0, 1.0 if half == 0 else -1.0])
            if segment % 2 == 0:
                local = segment // 2
                for qubit in range(2):
                    index = 2 * local + qubit
                    offset = 3 * qubit
                    result[pair, :, index] += weights * gradient[slot, offset]
                    xy_gradient[pair, :, index] += weights[:, None] * gradient[slot, offset+1:offset+3]
            else:
                result[pair, :, 8 + segment // 2] += weights * gradient[slot, 6]
        direction = raw / np.maximum(norm[..., None], 1e-300)
        clipped_gradient = ratio[..., None] * (
            xy_gradient - direction * np.sum(direction * xy_gradient, axis=-1, keepdims=True)
        )
        # The radial map is nonsmooth exactly on its circle; use the interior
        # derivative there and the tangent derivative strictly outside.
        xy_gradient = np.where((ratio < 1.0)[..., None], clipped_gradient, xy_gradient)
        result[..., 11:19] = xy_gradient[..., 0]
        result[..., 19:27] = xy_gradient[..., 1]
        return result * self.scales


class Dynamics:
    def __init__(self, env_config: dict, target: np.ndarray, bins: int):
        if env_config["control_model"] != "kak_exchange_27":
            raise ValueError("current benchmark requires kak_exchange_27")
        if env_config["noise"].get("kak_exchange_detuning_segments") != "all":
            raise ValueError("detuning must act in all seven segments")
        if env_config["initial_gate"].lower() != "ii":
            raise ValueError("current benchmark starts from identity")
        self.total_time = 7 * float(env_config["segment_time"])
        self.dt = self.total_time / bins
        noise = env_config["noise"]
        sample = NoiseSample(np.zeros(2), np.array(noise["relaxation_rates"], dtype=float), np.array(noise["dephasing_rates"], dtype=float))
        self.dissipator = qops.liouvillian(np.zeros((4, 4)), two_qubit_jump_ops(sample))
        self.target_super = qops.unitary_to_superoperator(target)
        self.propagations = 0

    def channel(self, pulse: np.ndarray, detuning: np.ndarray):
        generators = self.dt * (
            self.dissipator[None] + np.einsum("nc,cab->nab", pulse, D_L)
            + detuning[0] * D_L[0] + detuning[1] * D_L[3]
        )
        propagators = [la.expm(a) for a in generators]
        prefixes = [np.eye(16, dtype=complex)]
        for p in propagators:
            prefixes.append(p @ prefixes[-1])
        self.propagations += 1
        return generators, propagators, prefixes

    def score(self, channel: np.ndarray) -> float:
        overlap = np.trace(self.target_super.conj().T @ channel)
        if abs(overlap.imag) > 1e-8 or overlap.real < -1e-8:
            raise FloatingPointError("nonphysical process overlap")
        score = float((overlap.real + 4.0) / 20.0)
        if not np.isfinite(score) or score < 0.2 - 1e-9 or score > 1 + 1e-9:
            raise FloatingPointError("invalid average gate fidelity")
        return score

    def objective(self, pulse: np.ndarray, detunings: np.ndarray, weights: np.ndarray, *, gradient: bool):
        values = np.empty(len(weights))
        total_gradient = np.zeros_like(pulse)
        for member, (detuning, weight) in enumerate(zip(detunings, weights)):
            generators, propagators, prefixes = self.channel(pulse, detuning)
            values[member] = self.score(prefixes[-1])
            if not gradient:
                continue
            suffix = np.eye(16, dtype=complex)
            for slot in range(len(pulse)-1, -1, -1):
                # Under the bilinear trace pairing, D exp_A is self-adjoint.
                # One Frechet derivative per slot yields all control gradients.
                m = prefixes[slot] @ self.target_super.conj().T @ suffix
                adjoint = la.expm_frechet(generators[slot], m, compute_expm=False)
                total_gradient[slot] += weight * self.dt / 20.0 * np.einsum(
                    "ab,cba->c", adjoint, D_L
                ).real
                suffix = suffix @ propagators[slot]
        mean = float(weights @ values)
        std = float(np.sqrt(weights @ (values - mean)**2))
        return mean, std, total_gradient


def optimize(
    mapping: PulseMap, dynamics: Dynamics, method: str, test_detuning: np.ndarray,
    settings: dict, *, initial: np.ndarray | None = None, callback=None,
):
    z = np.zeros(mapping.shape) if initial is None else np.array(initial).reshape(mapping.shape)
    if method == "robust":
        detunings, weights = gaussian_ensemble(settings["detuning_std"], settings["robust_quadrature_order"])
    else:
        raise ValueError("Only the distribution-averaged robust benchmark is included")
    moment, variance = np.zeros_like(z), np.zeros_like(z)
    beta1, beta2 = 0.9, 0.999
    learning_rate = float(settings.get("learning_rate", 0.05))
    iterations = int(settings["iterations"])
    objective_calls = gradient_calls = optimization_solves = evaluation_solves = 0
    start = time.perf_counter()
    rows = []

    def objective(candidate, gradient):
        nonlocal objective_calls, gradient_calls, optimization_solves
        objective_calls += 1
        gradient_calls += int(gradient)
        optimization_solves += len(weights)
        return dynamics.objective(mapping.physical(candidate), detunings, weights, gradient=gradient)

    def record(iteration, mean, std, accepted, step_size, gradient_norm):
        nonlocal evaluation_solves
        # The evaluator is downstream of optimization and never selects a pulse.
        evaluated, _, _ = dynamics.objective(
            mapping.physical(z), np.array([test_detuning]), np.ones(1), gradient=False
        )
        evaluation_solves += 1
        row = dict(zip(TRACE_FIELDS, [
            iteration, evaluated, (5*evaluated-1)/4, 1-evaluated, mean, std,
            int(accepted), step_size, gradient_norm, objective_calls, gradient_calls,
            optimization_solves, evaluation_solves, time.perf_counter()-start,
        ]))
        rows.append(row)
        if callback is not None:
            callback(row)

    mean, std, _ = objective(z, False)
    record(0, mean, std, False, 0.0, 0.0)
    for iteration in range(1, iterations + 1):
        mean, std, physical_gradient = objective(z, True)
        gradient = mapping.pullback(z, physical_gradient)
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("nonfinite GRAPE gradient")
        moment = beta1 * moment + (1-beta1) * gradient
        variance = beta2 * variance + (1-beta2) * gradient**2
        denominator = np.sqrt(variance / (1-beta2**iteration)) + 1e-12
        direction = (moment / (1-beta1**iteration)) / denominator
        direction /= max(1.0, float(np.max(np.abs(direction))))
        displacement = np.clip(z + learning_rate*direction, -1, 1) - z
        if float(np.sum(gradient * displacement)) <= 0:
            direction = gradient / denominator
            direction /= max(1.0, float(np.max(np.abs(direction))))
        accepted = False
        step_size = learning_rate
        for _ in range(int(settings.get("max_backtracks", 16))):
            candidate = np.clip(z + step_size * direction, -1, 1)
            gain = float(np.sum(gradient * (candidate-z)))
            if gain <= 0:
                break
            candidate_mean, candidate_std, _ = objective(candidate, False)
            if candidate_mean >= mean + 1e-4 * gain:
                z, mean, std = candidate, candidate_mean, candidate_std
                accepted = True
                break
            step_size *= 0.5
        record(iteration, mean, std, accepted, step_size if accepted else 0.0, float(la.norm(gradient)))
    return z, rows
