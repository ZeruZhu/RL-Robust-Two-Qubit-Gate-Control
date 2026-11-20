from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

from walsh_rl.config import dump_config, load_config
from walsh_rl.training.fidelity_only_multi_agent import FidelityOnlyTrainingConfig


def test_stage_cli_does_not_relabel_an_existing_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = load_config(root/"configs/su4_segment.yaml")
    config = FidelityOnlyTrainingConfig(env_config=env, active_stage="alpha_0",
        input_cache_dir=str(tmp_path/"input"), output_dir=str(tmp_path/"output"), seed=999)
    run = tmp_path/"output"
    run.mkdir()
    dump_config(asdict(config), run/"config.yaml")
    before = (run/"config.yaml").read_bytes()
    result = subprocess.run([sys.executable, str(root/"scripts/train_stage.py"),
        "--config", str(root/"configs/su4_segment.yaml"), "--active-stage", "alpha_0",
        "--input-cache-dir", str(tmp_path/"input"), "--run-dir", str(run)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "different training configuration" in result.stderr
    assert (run/"config.yaml").read_bytes() == before
    assert not (run/"source_config.yaml").exists()
