"""Shared physical mapping for fidelity-only alpha/g/X/Y commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KAK_EXCHANGE_LOCAL_OFFSETS = (0, 7, 14, 21)
KAK_EXCHANGE_ALPHA_INDICES = np.array(
    [offset + index for offset in KAK_EXCHANGE_LOCAL_OFFSETS for index in (0, 1)],
    dtype=int,
)
KAK_EXCHANGE_EXCHANGE_INDICES = np.array([6, 13, 20], dtype=int)
LOCAL_GAMMA_INDICES = np.array(
    [offset + index for offset in KAK_EXCHANGE_LOCAL_OFFSETS for index in (2, 3)],
    dtype=int,
)
LOCAL_PHASE_INDICES = np.array(
    [offset + index for offset in KAK_EXCHANGE_LOCAL_OFFSETS for index in (4, 5)],
    dtype=int,
)
FIDELITY_COMMAND_FAMILIES = ("alpha", "g", "x", "y")
FIDELITY_COMMAND_WIDTHS = {"alpha": 8, "g": 3, "x": 8, "y": 8}
FIDELITY_COMMAND_DIM_PER_MODE = sum(FIDELITY_COMMAND_WIDTHS.values())


def log10_infidelity(average_gate_fidelity: float, eps: float) -> float:
    """Return the bounded log-fidelity score used by calibration policies."""

    return float(-np.log10(max(float(eps), 1.0 - float(average_gate_fidelity))))


@dataclass(frozen=True)
class FidelityCommandBounds:
    alpha: float
    g: float
    xy: float

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("g", self.g), ("xy", self.xy)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} command bound must be positive and finite")


def zero_fidelity_commands(num_modes: int) -> dict[str, np.ndarray]:
    if int(num_modes) < 1:
        raise ValueError("num_modes must be positive")
    return {
        family: np.zeros((int(num_modes), width), dtype=float)
        for family, width in FIDELITY_COMMAND_WIDTHS.items()
    }


def copy_fidelity_commands(commands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        family: np.asarray(commands[family], dtype=float).copy()
        for family in FIDELITY_COMMAND_FAMILIES
    }


def validate_fidelity_commands(
    commands: dict[str, np.ndarray], num_modes: int
) -> dict[str, np.ndarray]:
    if set(commands) != set(FIDELITY_COMMAND_FAMILIES):
        raise ValueError(
            f"commands must contain exactly {FIDELITY_COMMAND_FAMILIES}"
        )
    copied = copy_fidelity_commands(commands)
    for family, width in FIDELITY_COMMAND_WIDTHS.items():
        expected = (int(num_modes), width)
        if copied[family].shape != expected:
            raise ValueError(
                f"{family} commands must have shape {expected}, got "
                f"{copied[family].shape}"
            )
        if np.any(~np.isfinite(copied[family])):
            raise ValueError(f"{family} commands must be finite")
    return copied


def fidelity_commands_from_normalized_actions(
    normalized_actions: np.ndarray,
    bounds: FidelityCommandBounds,
) -> dict[str, np.ndarray]:
    actions = np.asarray(normalized_actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != FIDELITY_COMMAND_DIM_PER_MODE:
        raise ValueError(
            "normalized fidelity-only actions must have shape "
            f"(K, {FIDELITY_COMMAND_DIM_PER_MODE}), got {actions.shape}"
        )
    if np.any(~np.isfinite(actions)) or np.any(np.abs(actions) > 1.0 + 1.0e-12):
        raise ValueError("normalized fidelity-only actions must lie in [-1, 1]")
    actions = np.clip(actions, -1.0, 1.0)
    splits = np.cumsum([FIDELITY_COMMAND_WIDTHS[key] for key in FIDELITY_COMMAND_FAMILIES])[:-1]
    alpha, g_values, x_values, y_values = np.split(actions, splits, axis=1)
    return {
        "alpha": bounds.alpha * alpha,
        "g": bounds.g * g_values,
        "x": bounds.xy * x_values,
        "y": bounds.xy * y_values,
    }


def normalized_actions_from_fidelity_commands(
    commands: dict[str, np.ndarray],
    bounds: FidelityCommandBounds,
) -> np.ndarray:
    num_modes = np.asarray(commands["alpha"]).shape[0]
    values = validate_fidelity_commands(commands, num_modes)
    actions = np.concatenate(
        [
            values["alpha"] / bounds.alpha,
            values["g"] / bounds.g,
            values["x"] / bounds.xy,
            values["y"] / bounds.xy,
        ],
        axis=1,
    )
    if np.any(np.abs(actions) > 1.0 + 1.0e-12):
        raise ValueError("physical fidelity-only commands exceed configured bounds")
    return np.clip(actions, -1.0, 1.0)


def coefficients_from_fidelity_commands(
    analytic_coefficients: np.ndarray,
    commands: dict[str, np.ndarray],
    nominal_gamma_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply commands using the deployed fidelity-only physical semantics."""

    base = np.asarray(analytic_coefficients, dtype=float)
    if base.ndim != 2 or base.shape[1] != 27:
        raise ValueError(f"analytic coefficients must have shape (K, 27), got {base.shape}")
    values = validate_fidelity_commands(commands, base.shape[0])
    limits = np.asarray(nominal_gamma_limits, dtype=float).reshape(-1)
    if limits.shape == (4,):
        limits = np.repeat(limits, 2)
    if limits.shape != (8,) or np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("nominal gamma limits must have shape (4,) or (8,)")

    coefficients = base.copy()
    coefficients[:, KAK_EXCHANGE_ALPHA_INDICES] += values["alpha"]
    coefficients[:, KAK_EXCHANGE_EXCHANGE_INDICES] += values["g"]

    base_gamma = base[:, LOCAL_GAMMA_INDICES]
    base_phase = base[:, LOCAL_PHASE_INDICES]
    x_values = base_gamma * np.cos(base_phase) + values["x"]
    y_values = base_gamma * np.sin(base_phase) + values["y"]
    gamma = np.hypot(x_values, y_values)
    effective_limits = np.maximum(limits[None, :], np.abs(base_gamma))
    clipped = gamma > effective_limits
    scale = np.ones_like(gamma)
    nonzero = gamma > 0.0
    scale[nonzero] = np.minimum(
        1.0, effective_limits[nonzero] / gamma[nonzero]
    )
    x_values *= scale
    y_values *= scale
    gamma = np.hypot(x_values, y_values)
    phase = (np.arctan2(y_values, x_values) + np.pi) % (2.0 * np.pi) - np.pi
    phase[gamma == 0.0] = 0.0
    transverse_active = (values["x"] != 0.0) | (values["y"] != 0.0)
    gamma_coefficients = coefficients[:, LOCAL_GAMMA_INDICES].copy()
    phase_coefficients = coefficients[:, LOCAL_PHASE_INDICES].copy()
    gamma_coefficients[transverse_active] = gamma[transverse_active]
    phase_coefficients[transverse_active] = phase[transverse_active]
    coefficients[:, LOCAL_GAMMA_INDICES] = gamma_coefficients
    coefficients[:, LOCAL_PHASE_INDICES] = phase_coefficients
    return coefficients, clipped & transverse_active


def embed_normalized_fidelity_actions(
    source_actions: np.ndarray, destination_modes: int
) -> np.ndarray:
    source = np.asarray(source_actions, dtype=float)
    if source.ndim != 2 or source.shape[1] != FIDELITY_COMMAND_DIM_PER_MODE:
        raise ValueError(
            f"source actions must have shape (K, {FIDELITY_COMMAND_DIM_PER_MODE})"
        )
    if int(destination_modes) < source.shape[0]:
        raise ValueError("destination cannot have fewer Walsh modes than source")
    destination = np.zeros(
        (int(destination_modes), FIDELITY_COMMAND_DIM_PER_MODE), dtype=float
    )
    destination[: source.shape[0]] = source
    return destination


__all__ = [
    "FIDELITY_COMMAND_DIM_PER_MODE",
    "FIDELITY_COMMAND_FAMILIES",
    "FIDELITY_COMMAND_WIDTHS",
    "FidelityCommandBounds",
    "KAK_EXCHANGE_ALPHA_INDICES",
    "KAK_EXCHANGE_EXCHANGE_INDICES",
    "LOCAL_GAMMA_INDICES",
    "LOCAL_PHASE_INDICES",
    "coefficients_from_fidelity_commands",
    "copy_fidelity_commands",
    "embed_normalized_fidelity_actions",
    "fidelity_commands_from_normalized_actions",
    "normalized_actions_from_fidelity_commands",
    "log10_infidelity",
    "validate_fidelity_commands",
    "zero_fidelity_commands",
]
