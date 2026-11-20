#!/usr/bin/env python
"""Generate matched analytic contexts for fidelity-only staged training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


from walsh_rl.config import load_config  # noqa: E402
from walsh_rl.training.fidelity_only_multi_agent import (  # noqa: E402
    generate_base_context_cache,
    validate_base_context_cache_match,
)
from walsh_rl.training.logging import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--match-cache-dir")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = dict(config["fidelity_only_eight_agent"])
    splits = {
        "train": (
            int(settings.get("train_seed_start", 800000)),
            int(settings.get("train_contexts", 20000)),
        ),
        "validation": (
            int(settings.get("validation_seed_start", 830000)),
            int(settings.get("validation_contexts", 256)),
        ),
        "test": (
            int(settings.get("test_seed_start", 920000)),
            int(settings.get("test_contexts", 2048)),
        ),
    }
    if args.smoke:
        splits = {
            "train": (990000, 4),
            "validation": (990100, 2),
            "test": (990200, 2),
        }
    summary = generate_base_context_cache(config, args.output_dir, splits)
    if args.match_cache_dir:
        match_summary = validate_base_context_cache_match(
            args.output_dir,
            args.match_cache_dir,
            canonicalize_numerical_drift=True,
        )
        save_json(
            match_summary,
            Path(args.output_dir) / "legacy_match_summary.json",
        )
        summary = {"split_sizes": summary, "legacy_match": match_summary}
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved fidelity-only base cache to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
