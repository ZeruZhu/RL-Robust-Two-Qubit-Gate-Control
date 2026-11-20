"""Run distribution-averaged GRAPE or CRAB on explicitly supplied contexts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from walsh_rl.config import load_config
from walsh_rl.benchmarks import grape, crab, fourier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=Path(__file__).resolve().parents[1]/"configs/benchmarks.yaml")
    parser.add_argument("--contexts", type=Path, required=True, help="NPZ with targets, noise, analytic_coefficients, seeds")
    parser.add_argument("--method", choices=("grape", "crab"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1, help="Number of tasks (default: 1)")
    parser.add_argument("--smoke", action="store_true", help="One optimizer update and one quadrature node")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    env = load_config(args.env_config)
    settings = load_config(args.settings)[args.method]
    if args.smoke:
        settings.update(iterations=1, robust_quadrature_order=1)
        if args.method == "grape":
            settings["num_time_bins"] = 14
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; existing benchmark results are not overwritten")
    with np.load(args.contexts, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in ("targets", "noise", "analytic_coefficients", "seeds")}
    count = len(arrays["seeds"])
    if args.limit > count:
        parser.error(f"Requested {args.limit} tasks, but only {count} are available")
    rates = np.array(env["noise"]["relaxation_rates"] + env["noise"]["dephasing_rates"], dtype=float)
    if not np.allclose(arrays["noise"][:, 2:], rates, atol=0, rtol=0):
        parser.error("Cached dissipative rates do not match the environment configuration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(method=args.method, settings=settings, env_config=env, smoke=args.smoke,
                    context_sha256=hashlib.sha256(args.contexts.read_bytes()).hexdigest(),
                    tasks=args.limit)
    (args.output_dir/"request.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    rows = []
    optimized = {}
    for index in range(args.limit):
        target, base = arrays["targets"][index], arrays["analytic_coefficients"][index]
        seed, detuning = int(arrays["seeds"][index]), arrays["noise"][index, :2]
        folder = args.output_dir/f"seed{seed}"
        folder.mkdir()
        # GRAPE's robust optimizer depends on target/reference, not held-out detuning.
        key = (target.tobytes(), base.tobytes())
        if args.method == "grape":
            mapping = grape.PulseMap(base, env, settings["num_time_bins"])
            dynamics = grape.Dynamics(env, target, settings["num_time_bins"])
            if key not in optimized:
                commands, trace = grape.optimize(mapping, dynamics, "robust", detuning, settings)
                optimized[key] = (commands, seed)
                with (folder/"iterations.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=trace[0])
                    writer.writeheader()
                    writer.writerows(trace)
            commands, source_seed = optimized[key]
            physical = mapping.physical(commands)
            value = dynamics.objective(physical, detuning[None], np.ones(1), gradient=False)[0]
            np.savez_compressed(folder/"pulse.npz", normalized_commands=commands,
                                physical_timebin_controls=physical, target=target, analytic_coefficients=base)
        else:
            pulse = fourier.FourierPulse(base, env, settings["basis_size"], settings["basis_seed"], seed)
            dynamics = fourier.ContinuousDynamics(env, target, pulse, settings)
            nodes, weights = grape.gaussian_ensemble(settings["detuning_std"], settings["robust_quadrature_order"])
            check = crab.validate_gradient(pulse, dynamics, settings, fourier.PTM_CONTROLS, nodes, weights, seed)
            (folder/"gradient_validation.json").write_text(json.dumps(check, indent=2)+"\n", encoding="utf-8")
            best, info = crab.optimize(pulse, dynamics, settings, fourier.PTM_CONTROLS, nodes, weights, folder/"iterations.csv")
            coefficients = pulse.project(best["theta"])
            strict = fourier.ContinuousDynamics(env, target, pulse, dict(settings, rtol=1e-11, atol=1e-13, points_per_period=16))
            value = strict.objective(coefficients, detuning[None], np.ones(1))[0]
            source_seed = seed
            np.savez_compressed(folder/"pulse.npz", theta=best["theta"], coefficients=coefficients,
                                frequencies=pulse.omega, target=target, analytic_coefficients=base)
            (folder/"optimizer.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8")
        rows.append(dict(seed=seed, average_gate_fidelity=value, optimization_source_seed=source_seed))
        print(f"{args.method}: evaluated task {index+1}/{args.limit}, fidelity={value:.8f}", flush=True)
    with (args.output_dir/"results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
