#!/usr/bin/env python
"""Evaluate a frozen fidelity-only eight-policy bundle for repeated cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


from walsh_rl.training.fidelity_only_cycles import (  # noqa: E402
    evaluate_fidelity_only_cycles,
)
from walsh_rl.envs.fidelity_only_multi_agent_env import FIDELITY_ONLY_STAGE_ORDER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument("--final-cache-dir", required=True)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--stage-run-dir", action="append")
    stages.add_argument("--stage-root", type=Path, help="Directory containing alpha_0 through y_1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-cycles", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()
    stage_dirs = ([str(args.stage_root/key) for key in FIDELITY_ONLY_STAGE_ORDER]
                  if args.stage_root else args.stage_run_dir)
    summary = evaluate_fidelity_only_cycles(
        base_cache_dir=args.base_cache_dir,
        final_cache_dir=args.final_cache_dir,
        stage_run_dirs=stage_dirs,
        output_dir=args.output_dir,
        num_cycles=args.num_cycles,
        device=args.device,
        sample_limit=args.sample_limit,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved fidelity-only cycle evaluation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
