"""Frozen-cycle evaluation for fidelity-only heterogeneous policies."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .. import quantum_ops as qops
from ..config import load_config
from ..envs.fidelity_only_multi_agent_env import (
    FIDELITY_ONLY_STAGE_ORDER,
    FidelityOnlyEightAgentEnv,
    fidelity_only_observation_dim,
)
from ..fidelity_only_controls import log10_infidelity
from .fidelity_only_multi_agent import (
    FidelityOnlyContextDataset,
    FidelityOnlyPolicy,
    POLICY_FILENAME,
    _environment_signature,
)
from .logging import save_json


CYCLE_ONE_REPLAY_ATOL = 1.0e-6


def fidelity_only_evaluation_config(config: dict, *, target_gate: str | None = None) -> dict:
    """Override only the target after the source bundle has been validated."""
    result = copy.deepcopy(config)
    if target_gate is not None:
        if qops.gate_by_name(target_gate).shape != (4, 4):
            raise ValueError("cycle target override must be a two-qubit gate")
        result["target_gate"] = target_gate
        result.pop("target_sampling", None)
    return result


def load_fidelity_only_bundle(
    stage_run_dirs: list[str] | tuple[str, ...], *, device: str = "cpu"
) -> tuple[dict, tuple[FidelityOnlyPolicy, ...]]:
    directories = tuple(Path(value) for value in stage_run_dirs)
    if len(directories) != len(FIDELITY_ONLY_STAGE_ORDER):
        raise ValueError("fidelity-only bundle requires exactly eight stage directories")
    configs = tuple(load_config(path / "source_config.yaml") for path in directories)
    signatures = tuple(_environment_signature(config) for config in configs)
    if len(set(signatures)) != 1:
        raise ValueError("fidelity-only bundle stage environments do not match")
    policies = tuple(
        FidelityOnlyPolicy(path / "best_policy" / POLICY_FILENAME, device=device)
        for path in directories
    )
    if tuple(policy.stage_key for policy in policies) != FIDELITY_ONLY_STAGE_ORDER:
        raise ValueError("fidelity-only bundle stage order is invalid")
    if any(policy.environment_signature != signatures[0] for policy in policies):
        raise ValueError("fidelity-only policy environment signature mismatch")
    if any(
        policy.obs_dim != fidelity_only_observation_dim(configs[0], policy.stage_key)
        for policy in policies
    ):
        raise ValueError("fidelity-only policy observation dimension mismatch")
    return configs[0], policies


def _copy_context(context: dict) -> dict:
    copied = {}
    for key, value in context.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


def _atomic_save_sample(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _run_sample(
    context: dict,
    policies: tuple[FidelityOnlyPolicy, ...],
    envs: dict[str, FidelityOnlyEightAgentEnv],
    num_cycles: int,
) -> dict[str, np.ndarray]:
    current = _copy_context(context)
    raw_cycle = np.zeros(num_cycles, dtype=float)
    retained_cycle = np.zeros(num_cycles, dtype=float)
    stage_agfs = np.zeros((num_cycles, len(policies)), dtype=float)
    alpha_actions = np.zeros((num_cycles, 2, 16, 8), dtype=np.float32)
    g_actions = np.zeros((num_cycles, 2, 4, 3), dtype=np.float32)
    x_actions = np.zeros((num_cycles, 2, 4, 8), dtype=np.float32)
    y_actions = np.zeros((num_cycles, 2, 4, 8), dtype=np.float32)
    alpha_accepted = np.zeros((num_cycles, 2, 16), dtype=bool)
    accepted = np.zeros((num_cycles, 6, 4), dtype=bool)
    pulse_counts = np.zeros(num_cycles, dtype=np.int64)
    raw_coefficients = np.zeros((num_cycles, 2, 27), dtype=float)
    retained_coefficients = np.zeros((num_cycles, 2, 27), dtype=float)
    retained_superoperators = np.zeros((num_cycles, 16, 16), dtype=complex)
    retained_alpha_commands = np.zeros((num_cycles, 2, 8), dtype=float)
    retained_g_commands = np.zeros((num_cycles, 2, 3), dtype=float)
    retained_x_commands = np.zeros((num_cycles, 2, 8), dtype=float)
    retained_y_commands = np.zeros((num_cycles, 2, 8), dtype=float)
    stage_gamma_clip_rates = np.zeros((num_cycles, len(policies)), dtype=float)
    stage_g_exceedance_rates = np.zeros((num_cycles, len(policies)), dtype=float)
    best_visited_agf = float(current["current_agf"])

    for cycle in range(num_cycles):
        retained = _copy_context(current)
        retained_agf = float(current["current_agf"])
        proposal_stage = 0
        for stage_index, policy in enumerate(policies):
            env = envs[policy.stage_key]
            observation, _ = env.reset_from_context(current)
            done = False
            while not done:
                step = env.step_index
                action = policy.predict(observation)
                family = env.spec.family
                mode = env.spec.mode_index
                if family == "alpha":
                    alpha_actions[cycle, mode, step] = action
                elif family == "g":
                    g_actions[cycle, mode, step] = action
                elif family == "x":
                    x_actions[cycle, mode, step] = action
                else:
                    y_actions[cycle, mode, step] = action
                observation, _, done, _, info = env.step(action)
                candidate = env.export_context()
                candidate_agf = float(candidate["current_agf"])
                if candidate_agf > retained_agf:
                    retained = _copy_context(candidate)
                    retained_agf = candidate_agf
                best_visited_agf = max(best_visited_agf, candidate_agf)
                if family != "alpha":
                    accepted[cycle, proposal_stage, step] = bool(
                        info["accepted_history"][-1]
                    )
                else:
                    alpha_accepted[cycle, mode, step] = bool(
                        info["accepted_history"][-1]
                    )
            pulse_counts[cycle] += int(info["pulse_evaluations"])
            current = env.export_context()
            stage_agfs[cycle, stage_index] = float(current["current_agf"])
            stage_gamma_clip_rates[cycle, stage_index] = float(
                info["gamma_clip_rate"]
            )
            stage_g_exceedance_rates[cycle, stage_index] = float(
                info["g_nominal_exceedance_rate"]
            )
            if env.spec.family != "alpha":
                proposal_stage += 1
        raw_cycle[cycle] = float(current["current_agf"])
        retained_cycle[cycle] = retained_agf
        raw_coefficients[cycle] = current["current_coefficients"]
        retained_coefficients[cycle] = retained["current_coefficients"]
        retained_superoperators[cycle] = retained["current_superoperator"]
        retained_alpha_commands[cycle] = retained["alpha_commands"]
        retained_g_commands[cycle] = retained["g_commands"]
        retained_x_commands[cycle] = retained["x_commands"]
        retained_y_commands[cycle] = retained["y_commands"]
        current = retained

    return {
        "version": np.asarray(1, dtype=np.int64),
        "seed": np.asarray(int(context["seed"]), dtype=np.int64),
        "analytic_agf": np.asarray(float(context["analytic_agf"])),
        "raw_cycle_agfs": raw_cycle,
        "retained_cycle_agfs": retained_cycle,
        "stage_agfs": stage_agfs,
        "alpha_actions": alpha_actions,
        "g_actions": g_actions,
        "x_actions": x_actions,
        "y_actions": y_actions,
        "alpha_accepted": alpha_accepted,
        "accepted": accepted,
        "pulse_counts": pulse_counts,
        "raw_coefficients": raw_coefficients,
        "retained_coefficients": retained_coefficients,
        "retained_superoperators": retained_superoperators,
        "retained_alpha_commands": retained_alpha_commands,
        "retained_g_commands": retained_g_commands,
        "retained_x_commands": retained_x_commands,
        "retained_y_commands": retained_y_commands,
        "stage_gamma_clip_rates": stage_gamma_clip_rates,
        "stage_g_exceedance_rates": stage_g_exceedance_rates,
        "final_alpha_commands": current["alpha_commands"],
        "final_g_commands": current["g_commands"],
        "final_x_commands": current["x_commands"],
        "final_y_commands": current["y_commands"],
        "final_coefficients": current["current_coefficients"],
        "final_superoperator": current["current_superoperator"],
        "best_visited_agf": np.asarray(best_visited_agf),
    }


def _load_sample(path: Path, num_cycles: int, expected_seed: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    required = {
        "version",
        "seed",
        "raw_cycle_agfs",
        "retained_cycle_agfs",
        "stage_agfs",
        "alpha_actions",
        "g_actions",
        "x_actions",
        "y_actions",
        "accepted",
        "pulse_counts",
        "raw_coefficients",
        "retained_coefficients",
        "retained_superoperators",
        "retained_alpha_commands",
        "retained_g_commands",
        "retained_x_commands",
        "retained_y_commands",
        "stage_gamma_clip_rates",
        "stage_g_exceedance_rates",
        "final_coefficients",
        "final_superoperator",
    }
    missing = required - set(values)
    if missing:
        raise ValueError(f"cycle sample is incomplete in {path}: {sorted(missing)}")
    if int(values["version"]) != 1:
        raise ValueError(f"cycle sample version mismatch in {path}")
    if int(values["seed"]) != int(expected_seed):
        raise ValueError(f"cycle sample seed mismatch in {path}")
    if values["raw_cycle_agfs"].shape != (num_cycles,):
        raise ValueError(f"cycle sample count mismatch in {path}")
    return values


def _aggregate_samples(
    samples: list[dict[str, np.ndarray]],
    output_dir: Path,
    num_cycles: int,
    *,
    save_arrays: bool = True,
) -> dict:
    seeds = np.asarray([int(sample["seed"]) for sample in samples], dtype=np.int64)
    analytic = np.asarray([float(sample["analytic_agf"]) for sample in samples])
    raw = np.stack([sample["raw_cycle_agfs"] for sample in samples])
    retained = np.stack([sample["retained_cycle_agfs"] for sample in samples])
    stages = np.stack([sample["stage_agfs"] for sample in samples])
    pulses = np.stack([sample["pulse_counts"] for sample in samples])
    accepted = np.stack([sample["accepted"] for sample in samples])
    alpha_accepted = np.stack(
        [
            sample.get("alpha_accepted", np.zeros((num_cycles, 2, 16), dtype=bool))
            for sample in samples
        ]
    )
    gamma_clip_rates = np.stack([sample["stage_gamma_clip_rates"] for sample in samples])
    g_exceedance_rates = np.stack([sample["stage_g_exceedance_rates"] for sample in samples])
    replay_discrepancies = np.asarray(
        [
            float(sample.get("cycle_one_replay_discrepancy", np.nan))
            for sample in samples
        ]
    )
    if save_arrays:
        np.save(output_dir / "eval_seeds.npy", seeds)
        np.save(output_dir / "eval_analytic_average_gate_fidelities.npy", analytic)
        np.save(output_dir / "eval_raw_cycle_average_gate_fidelities.npy", raw)
        np.save(output_dir / "eval_retained_cycle_average_gate_fidelities.npy", retained)
        np.save(output_dir / "eval_stage_average_gate_fidelities.npy", stages)
        np.save(output_dir / "eval_cycle_pulse_counts.npy", pulses)
        for family in ("alpha", "g", "x", "y"):
            np.save(
                output_dir / f"eval_{family}_actions.npy",
                np.stack([sample[f"{family}_actions"] for sample in samples]),
            )
            np.save(
                output_dir / f"eval_retained_cycle_{family}_commands.npy",
                np.stack([sample[f"retained_{family}_commands"] for sample in samples]),
            )
        np.save(output_dir / "eval_proposal_acceptance.npy", accepted)
        np.save(output_dir / "eval_alpha_acceptance.npy", alpha_accepted)
        np.save(
            output_dir / "eval_raw_cycle_physical_coefficients.npy",
            np.stack([sample["raw_coefficients"] for sample in samples]),
        )
        np.save(
            output_dir / "eval_retained_cycle_physical_coefficients.npy",
            np.stack([sample["retained_coefficients"] for sample in samples]),
        )
        np.save(
            output_dir / "eval_retained_cycle_superoperators.npy",
            np.stack([sample["retained_superoperators"] for sample in samples]),
        )
        np.save(output_dir / "eval_stage_gamma_clip_rates.npy", gamma_clip_rates)
        np.save(output_dir / "eval_stage_g_exceedance_rates.npy", g_exceedance_rates)
        np.save(
            output_dir / "eval_cycle_one_replay_discrepancies.npy",
            replay_discrepancies,
        )
        np.save(
            output_dir / "eval_final_physical_coefficients.npy",
            np.stack([sample["final_coefficients"] for sample in samples]),
        )
    summary = {
        "num_samples": len(samples),
        "num_cycles": num_cycles,
        "analytic_average_gate_fidelity": float(np.mean(analytic)),
        "analytic_average_gate_fidelity_std": float(np.std(analytic)),
        "cycle_one_replay_tolerance": CYCLE_ONE_REPLAY_ATOL,
        "cycle_one_replay_max_abs_discrepancy": (
            float(np.nanmax(np.abs(replay_discrepancies)))
            if np.any(np.isfinite(replay_discrepancies))
            else None
        ),
        "cycles": [],
        "pulse_count_formula": "1 + 56 * cycles",
    }
    for cycle in range(num_cycles):
        values = retained[:, cycle]
        raw_values = raw[:, cycle]
        previous = analytic if cycle == 0 else retained[:, cycle - 1]
        summary["cycles"].append(
            {
                "cycle": cycle + 1,
                "raw_average_gate_fidelity": float(np.mean(raw_values)),
                "raw_average_gate_fidelity_std": float(np.std(raw_values)),
                "retained_average_gate_fidelity": float(np.mean(values)),
                "retained_average_gate_fidelity_std": float(np.std(values)),
                "retained_median": float(np.median(values)),
                "retained_p10": float(np.quantile(values, 0.10)),
                "retained_minimum": float(np.min(values)),
                "retained_log10_infidelity": log10_infidelity(
                    float(np.mean(values)), 1.0e-8
                ),
                "mean_marginal_improvement": float(np.mean(values - previous)),
                "strict_win_rate_over_previous": float(np.mean(values > previous)),
                "strict_loss_rate_vs_previous": float(np.mean(values < previous)),
                "tie_rate_vs_previous": float(np.mean(values == previous)),
                "proposal_acceptance_rate": float(np.mean(accepted[:, cycle])),
                "alpha_acceptance_rate": float(
                    np.mean(alpha_accepted[:, cycle])
                ),
                "gamma_clip_rate": float(np.mean(gamma_clip_rates[:, cycle])),
                "g_nominal_exceedance_rate": float(np.mean(g_exceedance_rates[:, cycle])),
                "fidelity_evaluations_per_instance": int(1 + 56 * (cycle + 1)),
            }
        )
    save_json(summary, output_dir / "cycle_summary.json")
    return summary


def evaluate_fidelity_only_cycles(
    *,
    base_cache_dir: str | Path,
    final_cache_dir: str | Path,
    stage_run_dirs: list[str] | tuple[str, ...],
    output_dir: str | Path,
    num_cycles: int = 10,
    device: str = "cpu",
    sample_limit: int | None = None,
) -> dict:
    if int(num_cycles) < 1:
        raise ValueError("num_cycles must be positive")
    env_config, policies = load_fidelity_only_bundle(stage_run_dirs, device=device)
    base = FidelityOnlyContextDataset(base_cache_dir, "test")
    final = FidelityOnlyContextDataset(final_cache_dir, "test")
    if _environment_signature(base.env_config) != _environment_signature(env_config):
        raise ValueError("cycle base cache environment mismatch")
    if _environment_signature(final.env_config) != _environment_signature(env_config):
        raise ValueError("cycle final cache environment mismatch")
    if base.metadata.get("stage_provenance") != "analytic":
        raise ValueError("cycle base cache must be analytic")
    if final.metadata.get("stage_provenance") != FIDELITY_ONLY_STAGE_ORDER[-1]:
        raise ValueError("cycle final cache must come from y_1")
    if not np.array_equal(base.arrays["seeds"], final.arrays["seeds"]):
        raise ValueError("cycle base and final caches use different test seeds")
    for key in ("targets", "noise", "analytic_coefficients", "analytic_agfs"):
        if not np.allclose(base.arrays[key], final.arrays[key], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"cycle base and final cache {key} do not match")
    count = base.size if sample_limit is None else min(base.size, int(sample_limit))
    output_dir = Path(output_dir)
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_digest = hashlib.sha256(
        np.asarray(base.arrays["seeds"][:count], dtype=np.int64).tobytes()
    ).hexdigest()
    request = {
        "version": 1,
        "environment_signature": _environment_signature(env_config),
        "base_cache_dir": str(Path(base_cache_dir).resolve()),
        "final_cache_dir": str(Path(final_cache_dir).resolve()),
        "stage_order": list(FIDELITY_ONLY_STAGE_ORDER),
        "policy_sha256": [policy.checkpoint_sha256 for policy in policies],
        "num_cycles": int(num_cycles),
        "num_samples": int(count),
        "seed_sha256": seed_digest,
    }
    request_path = output_dir / "evaluation_request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("fidelity-only cycle evaluation request mismatch")
    else:
        save_json(request, request_path)
    envs = {
        key: FidelityOnlyEightAgentEnv(env_config, active_stage=key)
        for key in FIDELITY_ONLY_STAGE_ORDER
    }
    samples = []
    for index in range(count):
        context = base.row(index)
        path = sample_dir / f"sample_{index:06d}.npz"
        if path.exists():
            result = _load_sample(path, int(num_cycles), int(context["seed"]))
        else:
            result = _run_sample(context, policies, envs, int(num_cycles))
            expected_cycle_one = float(final.arrays["current_agfs"][index])
            replay_discrepancy = float(
                result["raw_cycle_agfs"][0] - expected_cycle_one
            )
            if abs(replay_discrepancy) > CYCLE_ONE_REPLAY_ATOL:
                raise RuntimeError("cycle-one rollout does not reproduce final stage cache")
            result["cycle_one_cache_agf"] = np.asarray(expected_cycle_one)
            result["cycle_one_replay_discrepancy"] = np.asarray(
                replay_discrepancy
            )
            if not np.array_equal(result["pulse_counts"], np.full(num_cycles, 56)):
                raise RuntimeError("fidelity-only cycle pulse count mismatch")
            if not np.allclose(
                result["raw_cycle_agfs"],
                result["retained_cycle_agfs"],
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "all-accept/reject cycle produced different raw and retained fidelity"
                )
            _atomic_save_sample(path, result)
        samples.append(result)
        if (index + 1) % 16 == 0:
            _aggregate_samples(
                samples, output_dir, int(num_cycles), save_arrays=False
            )
    for env in envs.values():
        env.close()
    return _aggregate_samples(samples, output_dir, int(num_cycles))


__all__ = [
    "evaluate_fidelity_only_cycles",
    "load_fidelity_only_bundle",
]
