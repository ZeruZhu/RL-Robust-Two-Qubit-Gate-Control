from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from walsh_rl.config import load_config
from walsh_rl.envs.fidelity_only_multi_agent_env import parse_fidelity_only_stage
from walsh_rl.training.fidelity_only_multi_agent import (
    FidelityOnlyTrainingConfig, _make_agent, _save_resume_checkpoint, _restore_resume_checkpoint,
)


def test_numpy_resume_state_round_trip(tmp_path):
    env = load_config(Path(__file__).resolve().parents[1]/"configs/su4_segment.yaml")
    config = FidelityOnlyTrainingConfig(env_config=env, active_stage="alpha_0",
        input_cache_dir=str(tmp_path/"input"), output_dir=str(tmp_path/"output"),
        buffer_size=8, batch_size=2, actor_hidden_layers=(8, 8), critic_hidden_layers=(8, 8))
    spec, device = parse_fidelity_only_stage("alpha_0"), torch.device("cpu")
    original = _make_agent(config, spec, device)
    original.replay.add(np.ones(140), 0, np.zeros(8), 0.2, np.ones(140)*2, 0, False)
    generator = np.random.default_rng(123)
    path = tmp_path/"resume.pt"
    _save_resume_checkpoint(path, original, config, episode=1, training_rows=[],
        evaluation_rows=[], best_mean=0.9, best_episode=1, order=np.arange(4),
        permutation_rng=generator)
    restored = _make_agent(config, spec, device)
    restored_generator = np.random.default_rng(0)
    payload = _restore_resume_checkpoint(path, restored, config, restored_generator, device)
    assert payload["episode"] == 1 and restored.replay.size == 1
    np.testing.assert_array_equal(restored.replay.obs, original.replay.obs)
    np.testing.assert_array_equal(generator.integers(0, 100, 8), restored_generator.integers(0, 100, 8))
    for key, tensor in original.actor.state_dict().items():
        torch.testing.assert_close(tensor, restored.actor.state_dict()[key], rtol=0, atol=0)
