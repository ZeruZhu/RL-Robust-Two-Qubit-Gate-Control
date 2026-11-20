from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_detuning_contexts import generate
from walsh_rl.config import load_config
from walsh_rl.envs.fidelity_only_multi_agent_env import FidelityOnlyEightAgentEnv
from walsh_rl.training.fidelity_only_multi_agent import generate_base_context_cache, FidelityOnlyContextDataset

ROOT = Path(__file__).resolve().parents[1]


def test_detuning_contexts_preserve_target_and_rate_checks(tmp_path):
    config = load_config(ROOT/"configs/su4_segment.yaml")
    reference, output = tmp_path/"base", tmp_path/"binned"
    generate_base_context_cache(config, reference, {"test": (920000, 1)})
    generate(config, output, reference, bins=2, per_bin=2)
    dataset = FidelityOnlyContextDataset(output, "test")
    magnitudes = np.abs(dataset.arrays["noise"][:, :2])
    bins = np.repeat(np.arange(2), 2)
    assert np.all(magnitudes >= bins[:, None]*5e5)
    assert np.all(magnitudes < (bins[:, None]+1)*5e5)
    ordinary = FidelityOnlyEightAgentEnv(config, active_stage="alpha_0")
    with pytest.raises(RuntimeError, match="noise does not replay"):
        ordinary.reset_from_context(dataset.row(0))
    cached = FidelityOnlyEightAgentEnv(config, active_stage="alpha_0", use_cached_noise=True)
    context = dataset.row(0)
    observation, _ = cached.reset_from_context(context)
    assert np.isfinite(observation).all()
    replay = cached.inner_env.evaluate_physical_coefficients(context["analytic_coefficients"])
    assert abs(replay["average_gate_fidelity"] - context["analytic_agf"]) < 1e-12
    context["noise"][2] += 1
    with pytest.raises(RuntimeError, match="rates do not replay"):
        cached.reset_from_context(context)
