#!/usr/bin/env python
"""Train one calibration policy and materialize its output contexts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


from walsh_rl.config import dump_config, load_config  # noqa: E402
from walsh_rl.envs.fidelity_only_multi_agent_env import (  # noqa: E402
    FIDELITY_ONLY_STAGE_ORDER,
    parse_fidelity_only_stage,
)
from walsh_rl.training.fidelity_only_multi_agent import (  # noqa: E402
    FidelityOnlyTrainingConfig,
    _training_signature,
    train_fidelity_only_stage,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--active-stage", required=True, choices=FIDELITY_ONLY_STAGE_ORDER)
    parser.add_argument("--input-cache-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()

    env_config = load_config(args.config)
    training = dict(env_config.get("fidelity_only_training", {}))
    if args.device:
        training["device"] = args.device
    spec = parse_fidelity_only_stage(args.active_stage)
    training["buffer_size"] = int(
        training.get(
            "alpha_buffer_size" if spec.family == "alpha" else "proposal_buffer_size",
            500000 if spec.family == "alpha" else 100000,
        )
    )
    training["action_noise_sigma"] = float(
        training.get(
            "alpha_action_noise_sigma"
            if spec.family == "alpha"
            else "proposal_action_noise_sigma",
            0.05 if spec.family == "alpha" else 0.10,
        )
    )
    for key in (
        "alpha_buffer_size",
        "proposal_buffer_size",
        "alpha_action_noise_sigma",
        "proposal_action_noise_sigma",
    ):
        training.pop(key, None)
    if args.smoke:
        training.update(
            {
                "stage_episodes": 2,
                "learning_starts": 1,
                "buffer_size": 32,
                "batch_size": 4,
                "actor_hidden_layers": [16, 16],
                "critic_hidden_layers": [16, 16],
                "checkpoint_every_episodes": 1,
                "eval_every_episodes": 1,
                "checkpoint_eval_episodes": 2,
                "save_replay_buffer": True,
            }
        )
    training["actor_hidden_layers"] = tuple(training["actor_hidden_layers"])
    training["critic_hidden_layers"] = tuple(training["critic_hidden_layers"])
    training.update(
        {
            "env_config": env_config,
            "active_stage": args.active_stage,
            "input_cache_dir": str(Path(args.input_cache_dir).resolve()),
            "output_dir": str(Path(args.run_dir).resolve()),
        }
    )
    config = FidelityOnlyTrainingConfig(**training)
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    saved_config = run_dir / "config.yaml"
    if saved_config.exists():
        previous = FidelityOnlyTrainingConfig(**load_config(saved_config))
        if _training_signature(previous) != _training_signature(config):
            parser.error("Run directory has a different training configuration; use a new directory")
    dump_config(env_config, run_dir / "source_config.yaml")
    dump_config(asdict(config), run_dir / "config.yaml")
    summary = train_fidelity_only_stage(config)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved fidelity-only stage to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
