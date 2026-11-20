from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np

from walsh_rl import quantum_ops as qops
from walsh_rl.baselines.two_qubit_kak import (
    kak_exchange_coefficients,
)
from walsh_rl.envs.fidelity_only_multi_agent_env import (
    FIDELITY_ONLY_STAGE_ORDER,
    fidelity_only_observation_dim,
    fidelity_only_projection_dim,
    parse_fidelity_only_stage,
)
from walsh_rl.envs.two_qubit_env import TwoQubitWalshEnv
from walsh_rl.training.fidelity_only_cycles import (
    fidelity_only_evaluation_config,
)


def _config(convention: str) -> dict:
    return {
        "system": "two_qubit",
        "control_model": "kak_exchange_27",
        "action_mode": "residual",
        "base_mode": "analytic",
        "target_gate": "CNOT",
        "initial_gate": "II",
        "segment_time": 30.0e-9,
        "num_walsh_modes": 2,
        "num_time_bins": 2,
        "kak_exchange_basis_convention": convention,
        "noise": {"enabled": False},
    }


def test_released_kak_convention_is_exact() -> None:
    config = _config("coworker_exchange_compiled")
    coefficients = kak_exchange_coefficients(config)
    env = TwoQubitWalshEnv(copy.deepcopy(config))
    info = env.evaluate_physical_coefficients(coefficients)
    env.close()
    assert info["average_gate_fidelity"] > 1.0 - 1.0e-12


def test_reference_implements_random_su4_without_noise():
    config = _config("coworker_exchange_compiled")
    config["target_gate"] = "random_su4"
    env = TwoQubitWalshEnv(config)
    for seed in range(8):
        env.reset(seed=seed)
        result = env.evaluate_physical_coefficients(env.base_physical_coefficients)
        assert result["average_gate_fidelity"] > 1 - 1e-12


def test_fidelity_only_stage_contract() -> None:
    expected = {
        "alpha": (8, 16, 120),
        "g": (3, 4, 104),
        "x": (8, 4, 132),
        "y": (8, 4, 132),
    }
    for key in FIDELITY_ONLY_STAGE_ORDER:
        spec = parse_fidelity_only_stage(key)
        assert (spec.action_dim, spec.steps, spec.observation_dim) == expected[spec.family]


def test_stage_local_truncated_projection_dimensions() -> None:
    config = {
        "fidelity_only_eight_agent": {
            "residual_projection": "local_truncated_stage"
        }
    }
    expected_projection = {"alpha": 20, "g": 20, "x": 14, "y": 14}
    for key in FIDELITY_ONLY_STAGE_ORDER:
        spec = parse_fidelity_only_stage(key)
        projection_dim = expected_projection[spec.family]
        assert fidelity_only_projection_dim(config, key) == projection_dim
        assert fidelity_only_observation_dim(config, key) == (
            spec.observation_dim + projection_dim
        )


def _legacy_local_component(operator):
    result = np.zeros_like(operator, dtype=np.complex128)
    for _, pauli in TwoQubitWalshEnv._pauli_projection_basis()[:6]:
        result += np.vdot(pauli, operator) / np.vdot(pauli, pauli) * pauli
    return 0.5 * (result + result.conj().T)


def _projection_env(target="CNOT"):
    config = _config("coworker_exchange_compiled")
    config["target_gate"] = target
    config["disable_analytic_reference"] = True
    config["noise"] = {"enabled": True, "detuning_std": 5e6, "kak_exchange_detuning_segments": "all"}
    config["iterative_constant"] = {
        "include_toggling_z_projection_observation": True,
        "toggling_z_projection_local_only": True,
        "toggling_z_projection_reduce": "segments",
        "toggling_z_frame_point": "midpoint",
        "toggling_projection_control_source": "base",
        "toggling_z_projection_scale": 5e6,
        "clip_toggling_z_projection_observation": False,
        "include_local_action_toggling_projection_observation": True,
        "local_action_toggling_role": "y",
        "clip_local_action_toggling_projection_observation": False,
    }
    return TwoQubitWalshEnv(config)


def test_local_projection_null_guard_is_relative_and_not_clipping():
    nonlocal_axis = np.kron(qops.Z, qops.Z)
    local = TwoQubitWalshEnv._local_pauli_component
    for scale in (1e-6, 1.0, 1e6):
        assert not np.any(local(scale * (nonlocal_axis + 1e-15 * qops.Z1)))
        real_axis = scale * (nonlocal_axis + 1e-8 * qops.Z1)
        np.testing.assert_array_equal(local(real_axis), _legacy_local_component(real_axis))
    np.testing.assert_array_equal(local(1e-15 * qops.Z1), 1e-15 * qops.Z1)
    assert not np.any(local(np.zeros((4, 4), dtype=complex)))
    env = _projection_env()
    generator = qops.liouvillian(5e8 * qops.Z1, [])
    with patch.object(env, "_residual_generator", return_value=generator):
        values = env._projection_values_for_operators([local(nonlocal_axis), qops.Z1])
    np.testing.assert_allclose(values, [0.0, 5e8])
    env.close()


def test_cnot_and_cz_projection_features_are_well_conditioned():
    for target in ("CNOT", "CZ"):
        env = _projection_env(target)
        env.reset(seed=920000)
        info = env.evaluate_physical_coefficients(env.base_physical_coefficients)
        assert np.isfinite(info["average_gate_fidelity"])
        z_values = env._toggling_z_projection_observation()
        y_values = env._local_action_toggling_projection_observation()
        assert np.isfinite(z_values).all() and np.max(np.abs(z_values)) < 100
        assert np.isfinite(y_values).all() and np.max(np.abs(y_values)) < 100
        if target == "CNOT":
            np.testing.assert_array_equal(z_values[:2], [0.0, 0.0])
            assert y_values[0] == 0.0
        env.close()


def test_random_su4_nonnull_projections_are_unchanged():
    for seed in range(4):
        for role in ("x", "y"):
            env = _projection_env("random_su4")
            env.iterative_constant_config["local_action_toggling_role"] = role
            env.reset(seed=seed)
            env.evaluate_physical_coefficients(env.base_physical_coefficients)
            current_z = env._toggling_z_projection_observation()
            current_xy = env._local_action_toggling_projection_observation()
            env._local_action_toggling_basis_cache = {}
            with patch.object(TwoQubitWalshEnv, "_local_pauli_component", side_effect=_legacy_local_component):
                np.testing.assert_array_equal(current_z, env._toggling_z_projection_observation())
                np.testing.assert_array_equal(current_xy, env._local_action_toggling_projection_observation())
            env.close()


def test_cycle_target_override_changes_only_target_and_does_not_mutate_source():
    source = _projection_env("random_su4").config
    source["target_sampling"] = "haar_su4"
    original = copy.deepcopy(source)
    evaluated = fidelity_only_evaluation_config(source, target_gate="CNOT")
    assert source == original
    expected = copy.deepcopy(original)
    expected["target_gate"] = "CNOT"
    expected.pop("target_sampling")
    assert evaluated == expected
    assert fidelity_only_evaluation_config(source) == source
    try:
        fidelity_only_evaluation_config(source, target_gate="X")
    except ValueError:
        pass
    else:
        raise AssertionError("one-qubit target override was accepted")
