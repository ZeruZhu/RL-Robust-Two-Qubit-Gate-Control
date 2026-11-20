#!/usr/bin/env python
"""Evaluate one resumable shard of frozen fidelity-only policy cycles."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np


from walsh_rl.envs.fidelity_only_multi_agent_env import (  # noqa: E402
    FIDELITY_ONLY_STAGE_ORDER,
    FidelityOnlyEightAgentEnv,
)
from walsh_rl.envs.two_qubit_env import (  # noqa: E402
    LOCAL_PROJECTION_NULL_RTOL,
)
from walsh_rl.training.fidelity_only_cycles import (  # noqa: E402
    _run_sample,
    fidelity_only_evaluation_config,
    load_fidelity_only_bundle,
)
from walsh_rl.training.fidelity_only_multi_agent import (  # noqa: E402
    FidelityOnlyContextDataset,
    _environment_signature,
)
from walsh_rl.training.logging import save_json  # noqa: E402


ARRAY_KEYS = (
    "seeds",
    "analytic_average_gate_fidelities",
    "raw_cycle_average_gate_fidelities",
    "retained_cycle_average_gate_fidelities",
    "stage_average_gate_fidelities",
    "cycle_pulse_counts",
    "best_visited_average_gate_fidelities",
)

_WORKER_STATE: dict | None = None


def _atomic_save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _pack(rows: list[dict[str, np.ndarray]], num_cycles: int) -> dict[str, np.ndarray]:
    if not rows:
        return {
            "seeds": np.empty(0, dtype=np.int64),
            "analytic_average_gate_fidelities": np.empty(0, dtype=float),
            "raw_cycle_average_gate_fidelities": np.empty((0, num_cycles)),
            "retained_cycle_average_gate_fidelities": np.empty((0, num_cycles)),
            "stage_average_gate_fidelities": np.empty(
                (0, num_cycles, len(FIDELITY_ONLY_STAGE_ORDER))
            ),
            "cycle_pulse_counts": np.empty((0, num_cycles), dtype=np.int64),
            "best_visited_average_gate_fidelities": np.empty(0, dtype=float),
        }
    return {
        "seeds": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "analytic_average_gate_fidelities": np.asarray(
            [row["analytic_agf"] for row in rows], dtype=float
        ),
        "raw_cycle_average_gate_fidelities": np.stack(
            [row["raw_cycle_agfs"] for row in rows]
        ),
        "retained_cycle_average_gate_fidelities": np.stack(
            [row["retained_cycle_agfs"] for row in rows]
        ),
        "stage_average_gate_fidelities": np.stack(
            [row["stage_agfs"] for row in rows]
        ),
        "cycle_pulse_counts": np.stack([row["pulse_counts"] for row in rows]),
        "best_visited_average_gate_fidelities": np.asarray(
            [row["best_visited_agf"] for row in rows], dtype=float
        ),
    }


def _compact_result(result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "seed": np.asarray(result["seed"]),
        "analytic_agf": np.asarray(result["analytic_agf"]),
        "raw_cycle_agfs": result["raw_cycle_agfs"],
        "retained_cycle_agfs": result["retained_cycle_agfs"],
        "stage_agfs": result["stage_agfs"],
        "pulse_counts": result["pulse_counts"],
        "best_visited_agf": np.asarray(result["best_visited_agf"]),
    }


def _initialize_worker(
    base_cache_dir: str,
    stage_run_dirs: list[str],
    device: str,
    num_cycles: int,
    target_gate: str | None,
    use_cached_noise: bool = False,
) -> None:
    global _WORKER_STATE
    env_config, policies = load_fidelity_only_bundle(stage_run_dirs, device=device)
    env_config = fidelity_only_evaluation_config(env_config, target_gate=target_gate)
    _WORKER_STATE = {
        "base": FidelityOnlyContextDataset(base_cache_dir, "test"),
        "policies": policies,
        "envs": {
            key: FidelityOnlyEightAgentEnv(env_config, active_stage=key, use_cached_noise=use_cached_noise)
            for key in FIDELITY_ONLY_STAGE_ORDER
        },
        "num_cycles": int(num_cycles),
    }


def _evaluate_index(index: int) -> dict[str, np.ndarray]:
    if _WORKER_STATE is None:
        raise RuntimeError("cycle-evaluation worker was not initialized")
    result = _run_sample(
        _WORKER_STATE["base"].row(index),
        _WORKER_STATE["policies"],
        _WORKER_STATE["envs"],
        _WORKER_STATE["num_cycles"],
    )
    return _compact_result(result)


def _unpack(arrays: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    return [
        {
            "seed": np.asarray(arrays["seeds"][index]),
            "analytic_agf": np.asarray(
                arrays["analytic_average_gate_fidelities"][index]
            ),
            "raw_cycle_agfs": arrays["raw_cycle_average_gate_fidelities"][index],
            "retained_cycle_agfs": arrays[
                "retained_cycle_average_gate_fidelities"
            ][index],
            "stage_agfs": arrays["stage_average_gate_fidelities"][index],
            "pulse_counts": arrays["cycle_pulse_counts"][index],
            "best_visited_agf": np.asarray(
                arrays["best_visited_average_gate_fidelities"][index]
            ),
        }
        for index in range(arrays["seeds"].size)
    ]


def _load(path: Path, num_cycles: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = set(ARRAY_KEYS) - set(payload.files)
        if missing:
            raise ValueError(f"incomplete cycle shard {path}: {sorted(missing)}")
        arrays = {key: payload[key] for key in ARRAY_KEYS}
    if arrays["retained_cycle_average_gate_fidelities"].shape[1:] != (
        num_cycles,
    ):
        raise ValueError(f"cycle count mismatch in {path}")
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--stop-index", type=int, required=True)
    parser.add_argument("--num-cycles", type=int, default=10)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--stage-run-dir", action="append")
    stages.add_argument("--stage-root", type=Path, help="Directory containing alpha_0 through y_1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-cached-noise", action="store_true", help="Use cached detunings, retaining target and exact noise-rate replay checks")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--target-gate", help="Explicit fixed-target transfer; leaves checkpoints unchanged")
    args = parser.parse_args()
    if args.stage_root:
        args.stage_run_dir = [str(args.stage_root/key) for key in FIDELITY_ONLY_STAGE_ORDER]

    source_config, policies = load_fidelity_only_bundle(
        args.stage_run_dir, device=args.device
    )
    env_config = fidelity_only_evaluation_config(source_config, target_gate=args.target_gate)
    base = FidelityOnlyContextDataset(args.base_cache_dir, "test")
    if _environment_signature(base.env_config) != _environment_signature(env_config):
        raise ValueError("cycle base cache environment mismatch")
    if base.metadata.get("stage_provenance") != "analytic":
        raise ValueError("cycle base cache must be analytic")
    start = int(args.start_index)
    stop = min(int(args.stop_index), base.size)
    if not 0 <= start < stop <= base.size:
        raise ValueError(f"invalid shard interval [{start}, {stop})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{start:06d}_{stop:06d}"
    final_path = output_dir / f"{stem}.npz"
    partial_path = output_dir / f"{stem}.partial.npz"
    request_path = output_dir / f"{stem}.request.json"
    seeds = np.asarray(base.arrays["seeds"][start:stop], dtype=np.int64)
    request = {
        "version": 2,
        "source_environment_signature": _environment_signature(source_config),
        "environment_signature": _environment_signature(env_config),
        "target_gate_override": args.target_gate,
        "use_cached_noise": args.use_cached_noise,
        "local_projection_null_rtol": LOCAL_PROJECTION_NULL_RTOL,
        "base_cache_dir": str(Path(args.base_cache_dir).resolve()),
        "start_index": start,
        "stop_index": stop,
        "num_cycles": int(args.num_cycles),
        "seed_sha256": hashlib.sha256(seeds.tobytes()).hexdigest(),
        "policy_sha256": [policy.checkpoint_sha256 for policy in policies],
    }
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError(f"cycle shard request mismatch in {request_path}")
    else:
        save_json(request, request_path)

    if final_path.exists():
        arrays = _load(final_path, int(args.num_cycles))
        if not np.array_equal(arrays["seeds"], seeds):
            raise ValueError(f"completed cycle shard seeds mismatch in {final_path}")
        print(f"cycle shard already complete: {final_path}", flush=True)
        return

    rows: list[dict[str, np.ndarray]] = []
    if partial_path.exists():
        partial = _load(partial_path, int(args.num_cycles))
        expected = seeds[: partial["seeds"].size]
        if not np.array_equal(partial["seeds"], expected):
            raise ValueError(f"partial cycle shard seeds mismatch in {partial_path}")
        rows = _unpack(partial)

    pending_indices = range(start + len(rows), stop)
    if int(args.workers) == 1:
        envs = {
            key: FidelityOnlyEightAgentEnv(env_config, active_stage=key, use_cached_noise=args.use_cached_noise)
            for key in FIDELITY_ONLY_STAGE_ORDER
        }
        try:
            results = (
                _compact_result(
                    _run_sample(base.row(index), policies, envs, int(args.num_cycles))
                )
                for index in pending_indices
            )
            for result in results:
                rows.append(result)
                if len(rows) % int(args.checkpoint_every) == 0:
                    _atomic_save(partial_path, _pack(rows, int(args.num_cycles)))
                    print(
                        f"{stem}: completed {len(rows)}/{stop - start}", flush=True
                    )
        finally:
            for env in envs.values():
                env.close()
    else:
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(
                str(Path(args.base_cache_dir).resolve()),
                [str(Path(path).resolve()) for path in args.stage_run_dir],
                args.device,
                int(args.num_cycles),
                args.target_gate,
                args.use_cached_noise,
            ),
        ) as executor:
            for result in executor.map(_evaluate_index, pending_indices, chunksize=1):
                rows.append(result)
                if len(rows) % int(args.checkpoint_every) == 0:
                    _atomic_save(partial_path, _pack(rows, int(args.num_cycles)))
                    print(
                        f"{stem}: completed {len(rows)}/{stop - start}", flush=True
                    )

    arrays = _pack(rows, int(args.num_cycles))
    if not np.array_equal(arrays["seeds"], seeds):
        raise RuntimeError("completed cycle shard seed sequence mismatch")
    _atomic_save(final_path, arrays)
    partial_path.unlink(missing_ok=True)
    print(f"saved {stop - start} cycle samples to {final_path}", flush=True)


if __name__ == "__main__":
    main()
