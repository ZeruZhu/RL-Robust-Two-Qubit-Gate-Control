"""Staged calibration with fidelity, global, or segment-projection feedback."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .. import quantum_ops as qops
from .._compat import EnvBase, spaces
from ..fidelity_only_controls import (
    KAK_EXCHANGE_ALPHA_INDICES,
    KAK_EXCHANGE_EXCHANGE_INDICES,
    LOCAL_GAMMA_INDICES,
    LOCAL_PHASE_INDICES,
    coefficients_from_fidelity_commands,
    copy_fidelity_commands,
    validate_fidelity_commands,
    zero_fidelity_commands,
    log10_infidelity,
)
from .two_qubit_env import TwoQubitWalshEnv
from ..noise import NoiseSample


KAK_EXCHANGE_SEGMENTS = ("sq1", "tq1", "sq2", "tq2", "sq3", "tq3", "sq4")


FIDELITY_ONLY_STAGE_ORDER = (
    "alpha_0",
    "alpha_1",
    "g_0",
    "g_1",
    "x_0",
    "x_1",
    "y_0",
    "y_1",
)


@dataclass(frozen=True)
class FidelityOnlyStageSpec:
    key: str
    family: str
    mode_index: int

    @property
    def action_dim(self) -> int:
        return 3 if self.family == "g" else 8

    @property
    def steps(self) -> int:
        return 16 if self.family == "alpha" else 4

    @property
    def observation_dim(self) -> int:
        if self.family == "alpha":
            return 120
        if self.family == "g":
            return 104
        return 132


def parse_fidelity_only_stage(stage_key: str) -> FidelityOnlyStageSpec:
    key = str(stage_key).strip().lower()
    if key not in FIDELITY_ONLY_STAGE_ORDER:
        raise ValueError(f"invalid fidelity-only stage {stage_key!r}")
    family, mode = key.rsplit("_", 1)
    return FidelityOnlyStageSpec(key=key, family=family, mode_index=int(mode))


def fidelity_only_projection_mode(config: dict) -> str:
    settings = dict(config.get("fidelity_only_eight_agent", {}))
    mode = str(settings.get("residual_projection", "none"))
    mode = mode.strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "disabled": "none",
        "global_xyz": "local_pauli",
        "local_xyz": "local_pauli",
        "single_qubit_pauli": "local_pauli",
        "local_truncated": "local_truncated_stage",
        "local_truncated_toggling": "local_truncated_stage",
        "stagexy": "local_truncated_stage",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "local_pauli", "local_truncated_stage"}:
        raise ValueError(
            "fidelity_only_eight_agent.residual_projection must be "
            "'none', 'local_pauli', or 'local_truncated_stage'"
        )
    return mode


def fidelity_only_projection_dim(config: dict, stage_key: str) -> int:
    mode = fidelity_only_projection_mode(config)
    if mode == "none":
        return 0
    if mode == "local_pauli":
        return 6
    family = parse_fidelity_only_stage(stage_key).family
    return 20 if family in {"alpha", "g"} else 14


def fidelity_only_observation_dim(config: dict, stage_key: str) -> int:
    return (
        parse_fidelity_only_stage(stage_key).observation_dim
        + fidelity_only_projection_dim(config, stage_key)
    )


class FidelityOnlyEightAgentEnv(EnvBase):
    """Run one heterogeneous policy block from a materialized pulse context."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict, *, active_stage: str, use_cached_noise: bool = False):
        super().__init__()
        self.use_cached_noise = bool(use_cached_noise)
        self.config = copy.deepcopy(config)
        settings = dict(self.config.get("fidelity_only_eight_agent", {}))
        self.spec = parse_fidelity_only_stage(active_stage)
        self.projection_mode = fidelity_only_projection_mode(self.config)
        self.projection_dim = fidelity_only_projection_dim(
            self.config, self.spec.key
        )
        self.observation_dim = fidelity_only_observation_dim(
            self.config, self.spec.key
        )
        self.projection_scale = float(
            settings.get(
                "residual_projection_observation_scale",
                self.config.get("noise", {}).get("detuning_std", 1.0),
            )
        )
        self.clip_projection_observation = bool(
            settings.get("clip_residual_projection_observation", False)
        )
        if self.projection_mode == "local_truncated_stage":
            iterative = self.config.setdefault("iterative_constant", {})
            iterative.update(
                {
                    "toggling_z_projection_local_only": True,
                    "toggling_z_projection_reduce": "segments",
                    "toggling_z_frame_point": "midpoint",
                    "toggling_projection_control_source": "base",
                    "toggling_z_projection_scale": self.projection_scale,
                    "clip_toggling_z_projection_observation": False,
                    "clip_local_action_toggling_projection_observation": False,
                    "include_toggling_z_projection_observation": self.spec.family
                    in {"alpha", "g"},
                    "include_local_action_toggling_projection_observation": self.spec.family
                    in {"x", "y"},
                    "local_action_toggling_role": self.spec.family
                    if self.spec.family in {"x", "y"}
                    else "x",
                }
            )
        self.reward_eps = float(self.config.get("reward_eps", 1.0e-8))
        self.score_scale = float(settings.get("score_observation_scale", 8.0))
        self.history_length = int(settings.get("history_length", 4))
        self.alpha_increment = float(settings["alpha_increment_scale"])
        self.alpha_bound = float(settings["alpha_bound"])
        self.alpha_update_mode = str(
            settings.get("alpha_update_mode", "accept_reject")
        ).strip().lower()
        self.g_bound = float(settings.get("g_bound", 1.6e7))
        self.xy_bound = float(settings.get("xy_bound", 8.0e6))
        self.g_radii = np.asarray(
            settings.get("g_proposal_radii", [2.0e6, 1.0e6, 5.0e5, 2.5e5]),
            dtype=float,
        )
        self.xy_radii = np.asarray(
            settings.get(
                "xy_proposal_radii", [1.0e6, 5.0e5, 2.5e5, 1.25e5]
            ),
            dtype=float,
        )
        self.acceptance_tolerance = float(settings.get("acceptance_tolerance", 0.0))
        configured_order = tuple(
            settings.get("stage_order", FIDELITY_ONLY_STAGE_ORDER)
        )
        if configured_order != FIDELITY_ONLY_STAGE_ORDER:
            raise ValueError("fidelity-only stage order does not match the eight-agent contract")
        self._validate_settings()

        self.inner_env = TwoQubitWalshEnv(self.config)
        self._validate_inner_env()
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(self.spec.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        self.context: dict | None = None
        self.commands = self.zero_commands()
        self.current_coefficients = np.zeros((2, 27), dtype=float)
        self.current_operator = np.eye(16, dtype=complex)
        self.current_average_gate_fidelity = 0.0
        self.incoming_average_gate_fidelity = 0.0
        self.analytic_average_gate_fidelity = 0.0
        self.step_index = 0
        self.history: list[np.ndarray] = []
        self.pulse_evaluations = 0
        self.accepted_proposals = 0
        self.monotonicity_violations = 0
        self.gamma_clip_events = 0
        self.gamma_values_seen = 0
        self.observation_out_of_range = 0
        self.observation_values_seen = 0
        self.attempted_actions: list[np.ndarray] = []
        self.score_changes: list[float] = []
        self.accepted_history: list[bool] = []

    def _validate_settings(self) -> None:
        if self.history_length != 4:
            raise ValueError("fidelity-only v1 requires four outcome-history records")
        positive = {
            "score scale": self.score_scale,
            "alpha increment": self.alpha_increment,
            "alpha bound": self.alpha_bound,
            "g bound": self.g_bound,
            "XY bound": self.xy_bound,
        }
        if self.projection_mode != "none":
            positive["residual projection observation scale"] = self.projection_scale
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isclose(self.alpha_bound, 16.0 * self.alpha_increment):
            raise ValueError("alpha bound must equal sixteen alpha increments")
        if self.alpha_update_mode != "accept_reject":
            raise ValueError(
                "The released method requires alpha_update_mode='accept_reject'"
            )
        for name, radii in (("g", self.g_radii), ("XY", self.xy_radii)):
            if radii.shape != (4,) or np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
                raise ValueError(f"{name} proposal radii must contain four positive values")
            if np.any(np.diff(radii) >= 0.0):
                raise ValueError(f"{name} proposal radii must decrease strictly")
        if self.acceptance_tolerance < 0.0:
            raise ValueError("acceptance_tolerance must be nonnegative")

    def _validate_inner_env(self) -> None:
        if self.inner_env.control_model != "kak_exchange_27":
            raise ValueError("fidelity-only control requires kak_exchange_27")
        if self.inner_env.num_walsh_modes != 2 or self.inner_env.num_time_bins != 2:
            raise ValueError("fidelity-only control requires K=2 and two time bins")
        if self.inner_env.noise_observability != "hidden":
            raise ValueError("fidelity-only control requires hidden noise")
        if not np.isclose(self.inner_env.final_time, 210.0e-9, atol=1.0e-18):
            raise ValueError("fidelity-only control requires a 210 ns pulse")
        if not np.isclose(self.inner_env.segment_time, 30.0e-9, atol=1.0e-18):
            raise ValueError("fidelity-only control requires seven 30 ns segments")
        if self.inner_env.kak_exchange_detuning_segments != set(KAK_EXCHANGE_SEGMENTS):
            raise ValueError("fidelity-only control requires detuning in all segments")

    @staticmethod
    def zero_commands() -> dict[str, np.ndarray]:
        return zero_fidelity_commands(2)

    @staticmethod
    def copy_commands(commands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return copy_fidelity_commands(commands)

    @staticmethod
    def _commands_from_context(context: dict) -> dict[str, np.ndarray]:
        commands = {
            "alpha": np.asarray(context["alpha_commands"], dtype=float),
            "g": np.asarray(context["g_commands"], dtype=float),
            "x": np.asarray(context["x_commands"], dtype=float),
            "y": np.asarray(context["y_commands"], dtype=float),
        }
        return validate_fidelity_commands(commands, 2)

    def coefficients_from_commands(
        self, commands: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        return coefficients_from_fidelity_commands(
            self.context["analytic_coefficients"],
            commands,
            np.full(8, self.inner_env.bounds.gamma_max, dtype=float),
        )

    def _base_xy(self) -> tuple[np.ndarray, np.ndarray]:
        base = np.asarray(self.context["analytic_coefficients"], dtype=float)
        gamma = base[:, LOCAL_GAMMA_INDICES]
        phase = base[:, LOCAL_PHASE_INDICES]
        return gamma * np.cos(phase), gamma * np.sin(phase)

    def _install_context(self, context: dict) -> None:
        self.context = context
        seed = int(context["seed"])
        self.inner_env.reset(seed=seed)
        if not np.allclose(self.inner_env.target, context["target"], atol=1.0e-12):
            raise RuntimeError("fidelity-only cached target does not replay")
        if self.use_cached_noise:
            cached = np.asarray(context["noise"], dtype=float)
            regenerated = self.inner_env.noise_sample.as_vector()
            if cached.shape != (6,) or not np.all(np.isfinite(cached)):
                raise ValueError("cached noise must contain six finite values")
            if not np.array_equal(cached[2:], regenerated[2:]):
                raise RuntimeError("fidelity-only cached noise rates do not replay")
            self.inner_env.noise_sample = NoiseSample(
                cached[:2].copy(), regenerated[2:4].copy(), regenerated[4:].copy()
            )
        elif not np.allclose(
            self.inner_env.noise_sample.as_vector(), context["noise"], atol=1.0e-12
        ):
            raise RuntimeError("fidelity-only cached noise does not replay")
        analytic = np.asarray(context["analytic_coefficients"], dtype=float)
        if analytic.shape != (2, 27):
            raise ValueError("analytic coefficients must have shape (2, 27)")
        self.inner_env.analytic_base_physical_coefficients = analytic.copy()
        self.inner_env.base_physical_coefficients = analytic.copy()
        self.commands = self._commands_from_context(context)
        coefficients, _ = self.coefficients_from_commands(self.commands)
        expected_coefficients = np.asarray(context["current_coefficients"], dtype=float)
        if not np.allclose(coefficients, expected_coefficients, rtol=0.0, atol=1.0e-8):
            raise RuntimeError("cached commands do not reconstruct physical coefficients")
        self.current_coefficients = coefficients
        self.current_operator = np.asarray(context["current_superoperator"]).copy()
        self.current_average_gate_fidelity = float(context["current_agf"])
        self.incoming_average_gate_fidelity = self.current_average_gate_fidelity
        self.analytic_average_gate_fidelity = float(context["analytic_agf"])
        self.inner_env.physical_coefficients = coefficients.copy()
        self.inner_env.current_operator = self.current_operator.copy()
        self.inner_env.current_step = self.inner_env.num_walsh_modes
        self.inner_env.fidelity = self.inner_env._compute_fidelity()
        replay_agf = (
            qops.average_gate_fidelity_from_process_trace(self.inner_env.fidelity, 4)
            if self.inner_env.use_superoperator
            else qops.average_gate_fidelity_from_unitary_fidelity(
                self.inner_env.fidelity, 4
            )
        )
        if not np.isclose(
            replay_agf, self.current_average_gate_fidelity, rtol=0.0, atol=3.0e-10
        ):
            raise RuntimeError("cached fidelity-only AGF does not replay")

    def _analytic_cartesian_context(self) -> np.ndarray:
        base = np.asarray(self.context["analytic_coefficients"], dtype=float)
        base_x, base_y = self._base_xy()
        alpha_scale = np.repeat(
            np.asarray([self.inner_env.bounds.alpha_max] * 4, dtype=float), 2
        )
        gamma_scale = np.repeat(
            np.asarray([self.inner_env.bounds.gamma_max] * 4, dtype=float), 2
        )
        g_scale = np.asarray([self.inner_env.bounds.g_eff_max] * 3, dtype=float)
        return np.concatenate(
            [
                base[0, KAK_EXCHANGE_ALPHA_INDICES] / alpha_scale,
                base[0, KAK_EXCHANGE_EXCHANGE_INDICES] / g_scale,
                base_x[0] / gamma_scale,
                base_y[0] / gamma_scale,
            ]
        )

    def _normalized_command_context(self) -> np.ndarray:
        frames = []
        for mode in range(2):
            frames.append(
                np.concatenate(
                    [
                        self.commands["alpha"][mode] / self.alpha_bound,
                        self.commands["g"][mode] / self.g_bound,
                        self.commands["x"][mode] / self.xy_bound,
                        self.commands["y"][mode] / self.xy_bound,
                    ]
                )
            )
        return np.concatenate(frames)

    def _radius_fraction(self) -> float:
        if self.spec.family == "alpha":
            return self.alpha_increment / self.alpha_bound
        radii = self.g_radii if self.spec.family == "g" else self.xy_radii
        bound = self.g_bound if self.spec.family == "g" else self.xy_bound
        return float(radii[self.step_index] / bound)

    def _gamma_headroom(self) -> np.ndarray:
        base_x, base_y = self._base_xy()
        x_values = base_x[self.spec.mode_index] + self.commands["x"][self.spec.mode_index]
        y_values = base_y[self.spec.mode_index] + self.commands["y"][self.spec.mode_index]
        gamma = np.hypot(x_values, y_values)
        nominal = np.full(8, self.inner_env.bounds.gamma_max, dtype=float)
        base_gamma = np.abs(
            np.asarray(self.context["analytic_coefficients"])[
                self.spec.mode_index, LOCAL_GAMMA_INDICES
            ]
        )
        limits = np.maximum(nominal, base_gamma)
        return (limits - gamma) / limits

    def _history_record_dim(self) -> int:
        if self.spec.family == "alpha":
            return 9
        return self.spec.action_dim + 2

    def _residual_projection_observation(self) -> np.ndarray:
        if self.projection_mode == "none":
            return np.empty(0, dtype=float)
        residual = self.inner_env._residual_operator()
        duration = max(self.inner_env.final_time, np.finfo(float).eps)
        paulis = (qops.X1, qops.Y1, qops.Z1, qops.X2, qops.Y2, qops.Z2)
        if self.inner_env.use_superoperator:
            generator = qops.la.logm(residual) / duration
            basis_generators = tuple(qops.liouvillian(pauli, []) for pauli in paulis)
        else:
            generator = 1j * qops.la.logm(
                np.asarray(residual, dtype=np.complex128)
            ) / duration
            generator = 0.5 * (generator + generator.conj().T)
            basis_generators = paulis
        projections = []
        for basis_generator in basis_generators:
            denominator = np.vdot(basis_generator, basis_generator)
            projections.append(
                0.0
                if abs(denominator) <= np.finfo(float).eps
                else float(
                    np.real(np.vdot(basis_generator, generator) / denominator)
                )
            )
        observation = np.asarray(projections, dtype=float) / self.projection_scale
        if self.clip_projection_observation:
            observation = np.clip(observation, -1.0, 1.0)
        if self.projection_mode == "local_truncated_stage":
            stage_projection = (
                self.inner_env._toggling_z_projection_observation()
                if self.spec.family in {"alpha", "g"}
                else self.inner_env._local_action_toggling_projection_observation()
            )
            observation = np.concatenate([observation, stage_projection])
        if observation.shape != (self.projection_dim,):
            raise RuntimeError(
                f"{self.spec.key} projection has shape {observation.shape}, "
                f"expected {(self.projection_dim,)}"
            )
        return observation

    def _observation(self) -> np.ndarray:
        denominator = max(1, self.spec.steps - 1)
        values = [
            self._analytic_cartesian_context(),
            self._normalized_command_context(),
            [log10_infidelity(self.current_average_gate_fidelity, self.reward_eps) / self.score_scale],
        ]
        if self.projection_dim:
            values.append(self._residual_projection_observation())
        values.extend(
            ([self.step_index / denominator], [self._radius_fraction()])
        )
        if self.spec.family in {"x", "y"}:
            values.append(self._gamma_headroom())
        history = list(self.history[: self.history_length])
        zero = np.zeros(self._history_record_dim(), dtype=float)
        while len(history) < self.history_length:
            history.append(zero.copy())
        values.extend(history)
        observation = np.concatenate(values).astype(np.float32)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(
                f"fidelity-only observation has shape {observation.shape}, "
                f"expected {(self.observation_dim,)}"
            )
        self.observation_out_of_range += int(np.count_nonzero(np.abs(observation) > 1.0))
        self.observation_values_seen += int(observation.size)
        return observation

    def reset_from_context(self, context: dict):
        self._install_context(context)
        self.step_index = 0
        self.history = []
        self.pulse_evaluations = 0
        self.accepted_proposals = 0
        self.monotonicity_violations = 0
        self.gamma_clip_events = 0
        self.gamma_values_seen = 0
        self.observation_out_of_range = 0
        self.observation_values_seen = 0
        self.attempted_actions = []
        self.score_changes = []
        self.accepted_history = []
        return self._observation(), self._info()

    def reset(self, *, seed=None, options=None):
        del seed
        options = options or {}
        if "context" not in options:
            raise ValueError("fidelity-only reset requires options['context']")
        return self.reset_from_context(options["context"])

    def _evaluate(self, commands: dict[str, np.ndarray]) -> tuple[dict, np.ndarray, np.ndarray]:
        coefficients, clipped = self.coefficients_from_commands(commands)
        info = self.inner_env.evaluate_physical_coefficients(coefficients)
        self.pulse_evaluations += 1
        return info, coefficients, clipped

    def _restore_retained_state(self) -> None:
        self.inner_env.physical_coefficients = self.current_coefficients.copy()
        self.inner_env.current_operator = self.current_operator.copy()
        self.inner_env.current_step = self.inner_env.num_walsh_modes
        self.inner_env.fidelity = self.inner_env._compute_fidelity()

    def step(self, action):
        if self.step_index >= self.spec.steps:
            raise RuntimeError("fidelity-only stage is complete; reset before stepping")
        action = np.asarray(action, dtype=float)
        if action.shape != (self.spec.action_dim,):
            raise ValueError(
                f"{self.spec.key} action must have shape {(self.spec.action_dim,)}"
            )
        action = np.clip(action, -1.0, 1.0)
        old_agf = self.current_average_gate_fidelity
        old_score = log10_infidelity(self.current_average_gate_fidelity, self.reward_eps)
        proposal_commands = self.copy_commands(self.commands)
        accepted = True

        if self.spec.family == "alpha":
            proposal_commands["alpha"][self.spec.mode_index] = np.clip(
                self.commands["alpha"][self.spec.mode_index]
                + self.alpha_increment * action,
                -self.alpha_bound,
                self.alpha_bound,
            )
        else:
            radii = self.g_radii if self.spec.family == "g" else self.xy_radii
            bound = self.g_bound if self.spec.family == "g" else self.xy_bound
            proposal_commands[self.spec.family][self.spec.mode_index] = np.clip(
                self.commands[self.spec.family][self.spec.mode_index]
                + radii[self.step_index] * action,
                -bound,
                bound,
            )

        proposal_info, proposal_coefficients, clipped = self._evaluate(
            proposal_commands
        )
        proposal_agf = float(proposal_info["average_gate_fidelity"])
        proposal_score = log10_infidelity(proposal_agf, self.reward_eps)
        score_change = float(proposal_score - old_score)

        accepted = proposal_agf > (
            self.current_average_gate_fidelity + self.acceptance_tolerance
        )
        if accepted:
            self.commands = proposal_commands
            self.current_coefficients = proposal_coefficients.copy()
            self.current_operator = self.inner_env.current_operator.copy()
            self.current_average_gate_fidelity = proposal_agf
            self.accepted_proposals += 1
        else:
            self._restore_retained_state()

        if self.current_average_gate_fidelity < old_agf - 1.0e-12:
            self.monotonicity_violations += 1

        self.gamma_clip_events += int(np.count_nonzero(clipped))
        self.gamma_values_seen += int(clipped.size)
        record_values = [action, [score_change]]
        if self.spec.family != "alpha":
            record_values.append([float(accepted)])
        self.history.insert(0, np.concatenate(record_values))
        del self.history[self.history_length :]
        self.attempted_actions.append(action.copy())
        self.score_changes.append(score_change)
        self.accepted_history.append(bool(accepted))
        self.step_index += 1
        terminated = self.step_index >= self.spec.steps
        reward = score_change
        observation = (
            np.zeros(self.observation_dim, dtype=np.float32)
            if terminated
            else self._observation()
        )
        return observation, float(reward), terminated, False, self._info()

    def _info(self) -> dict:
        time_bin_g = self.inner_env.basis.T @ self.current_coefficients[
            :, KAK_EXCHANGE_EXCHANGE_INDICES
        ]
        g_limits = np.full((1, 3), self.inner_env.bounds.g_eff_max, dtype=float)
        saturation_rates = {
            "alpha": float(np.mean(np.abs(self.commands["alpha"]) >= self.alpha_bound - 1.0e-9)),
            "g": float(np.mean(np.abs(self.commands["g"]) >= self.g_bound - 1.0e-9)),
            "x": float(np.mean(np.abs(self.commands["x"]) >= self.xy_bound - 1.0e-9)),
            "y": float(np.mean(np.abs(self.commands["y"]) >= self.xy_bound - 1.0e-9)),
        }
        return {
            "stage_key": self.spec.key,
            "alpha_update_mode": self.alpha_update_mode,
            "residual_projection": self.projection_mode,
            "residual_projection_dim": self.projection_dim,
            "step_index": int(self.step_index),
            "steps": int(self.spec.steps),
            "average_gate_fidelity": float(self.current_average_gate_fidelity),
            "incoming_average_gate_fidelity": float(self.incoming_average_gate_fidelity),
            "analytic_average_gate_fidelity": float(self.analytic_average_gate_fidelity),
            "commands": self.copy_commands(self.commands),
            "physical_coefficients": self.current_coefficients.copy(),
            "final_superoperator": self.current_operator.copy(),
            "pulse_evaluations": int(self.pulse_evaluations),
            "accepted_proposals": int(self.accepted_proposals),
            "acceptance_rate": float(
                self.accepted_proposals / max(1, self.step_index)
                if (
                    self.spec.family != "alpha"
                    or self.alpha_update_mode == "accept_reject"
                )
                else 1.0
            ),
            "monotonicity_violations": int(self.monotonicity_violations),
            "gamma_clip_rate": float(
                self.gamma_clip_events / max(1, self.gamma_values_seen)
            ),
            "g_nominal_exceedance_rate": float(
                np.mean(np.abs(time_bin_g) > g_limits + 1.0e-9)
            ),
            "command_saturation_rates": saturation_rates,
            "observation_out_of_range_rate": float(
                self.observation_out_of_range / max(1, self.observation_values_seen)
            ),
            "attempted_actions": np.asarray(self.attempted_actions, dtype=float),
            "score_changes": np.asarray(self.score_changes, dtype=float),
            "accepted_history": np.asarray(self.accepted_history, dtype=bool),
        }

    def export_context(self) -> dict:
        if self.context is None:
            raise RuntimeError("cannot export an uninitialized fidelity-only context")
        return {
            "seed": int(self.context["seed"]),
            "target": np.asarray(self.context["target"]).copy(),
            "noise": np.asarray(self.context["noise"], dtype=float).copy(),
            "analytic_coefficients": np.asarray(
                self.context["analytic_coefficients"], dtype=float
            ).copy(),
            "analytic_agf": float(self.analytic_average_gate_fidelity),
            "alpha_commands": self.commands["alpha"].copy(),
            "g_commands": self.commands["g"].copy(),
            "x_commands": self.commands["x"].copy(),
            "y_commands": self.commands["y"].copy(),
            "current_coefficients": self.current_coefficients.copy(),
            "current_superoperator": self.current_operator.copy(),
            "current_agf": float(self.current_average_gate_fidelity),
            "cumulative_pulse_evaluations": int(
                self.context.get("cumulative_pulse_evaluations", 1)
                + self.pulse_evaluations
            ),
            "stage_provenance": str(self.spec.key),
        }

    def close(self):
        close = getattr(self.inner_env, "close", None)
        if close is not None:
            close()


__all__ = [
    "FIDELITY_ONLY_STAGE_ORDER",
    "FidelityOnlyEightAgentEnv",
    "FidelityOnlyStageSpec",
    "fidelity_only_observation_dim",
    "fidelity_only_projection_dim",
    "fidelity_only_projection_mode",
    "parse_fidelity_only_stage",
]
