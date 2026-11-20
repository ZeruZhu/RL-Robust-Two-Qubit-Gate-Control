"""Generate contexts, train eight policies, and evaluate frozen inference cycles."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from walsh_rl.config import dump_config, load_config
from walsh_rl.envs.fidelity_only_multi_agent_env import FIDELITY_ONLY_STAGE_ORDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, help="Training seed, not the evaluation task seeds")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Tiny networks and datasets; not a scientific result")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
        config["fidelity_only_training"]["seed"] = args.seed
    config["fidelity_only_training"]["device"] = args.device
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    saved = root / "pipeline_config.yaml"
    request = dict(config=config, smoke=args.smoke, cycles=args.cycles)
    if saved.exists() and load_config(saved) != request:
        parser.error("Run directory belongs to a different pipeline configuration")
    dump_config(request, saved)
    effective = root / "env_config.yaml"
    dump_config(config, effective)
    scripts = Path(__file__).resolve().parent
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(key, "1")

    def run(name, *arguments):
        print(f"Running {name}", flush=True)
        subprocess.run([sys.executable, str(scripts/name), *map(str, arguments)],
                       env=environment, check=True)

    smoke = ["--smoke"] if args.smoke else []
    base = root / "contexts" / "base"
    run("prepare_contexts.py", "--config", effective, "--output-dir", base, *smoke)
    cache = base
    stages = []
    for key in FIDELITY_ONLY_STAGE_ORDER:
        stage = root / "stages" / key
        run("train_stage.py", "--config", effective, "--active-stage", key,
            "--input-cache-dir", cache, "--run-dir", stage, "--device", args.device, *smoke)
        stages.extend(["--stage-run-dir", stage])
        cache = stage / "materialized_cache"
    run("evaluate_cycles.py", "--base-cache-dir", base, "--final-cache-dir", cache,
        *stages, "--output-dir", root/"cycles", "--num-cycles", args.cycles,
        "--device", args.device)


if __name__ == "__main__":
    main()
