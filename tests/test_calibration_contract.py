from pathlib import Path

import numpy as np
import pytest

from walsh_rl.config import load_config
from walsh_rl.envs.fidelity_only_multi_agent_env import FidelityOnlyEightAgentEnv, FIDELITY_ONLY_STAGE_ORDER
from walsh_rl.training.fidelity_only_multi_agent import generate_base_context_cache, FidelityOnlyContextDataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("contexts")
    config = load_config(ROOT/"configs/su4_segment.yaml")
    generate_base_context_cache(config, path, {"test": (920000, 2)})
    return FidelityOnlyContextDataset(path, "test")


@pytest.mark.parametrize("mode", ["none", "local_pauli", "local_truncated_stage"])
def test_all_stages_accept_only_improvements(dataset, mode):
    config = load_config(ROOT/"configs/su4_segment.yaml")
    config["fidelity_only_eight_agent"]["residual_projection"] = mode
    context = dataset.row(0)
    rng = np.random.default_rng(28)
    for stage in FIDELITY_ONLY_STAGE_ORDER:
        env = FidelityOnlyEightAgentEnv(config, active_stage=stage)
        observation, _ = env.reset_from_context(context)
        assert observation.shape == env.observation_space.shape
        previous = env.current_average_gate_fidelity
        for step in range(env.spec.steps):
            observation, reward, terminated, truncated, _ = env.step(rng.uniform(-1, 1, env.spec.action_dim))
            assert np.isfinite(observation).all() and np.isfinite(reward)
            assert env.current_average_gate_fidelity >= previous
            assert terminated == (step == env.spec.steps-1) and not truncated
            previous = env.current_average_gate_fidelity
        context = env.export_context()
        env.close()


def test_invalid_actions_and_discarded_update_mode_rejected(dataset):
    config = load_config(ROOT/"configs/su4_segment.yaml")
    env = FidelityOnlyEightAgentEnv(config, active_stage="alpha_0")
    env.reset_from_context(dataset.row(0))
    with pytest.raises(ValueError):
        env.step(np.zeros(7))
    config["fidelity_only_eight_agent"]["alpha_update_mode"] = "accumulate"
    with pytest.raises(ValueError, match="accept_reject"):
        FidelityOnlyEightAgentEnv(config, active_stage="alpha_0")
