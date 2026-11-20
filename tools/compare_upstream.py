"""Check pulse, feedback, and action-response parity against a research checkout."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from walsh_rl.config import load_config
from walsh_rl.envs.fidelity_only_multi_agent_env import FIDELITY_ONLY_STAGE_ORDER, FidelityOnlyEightAgentEnv
from walsh_rl.envs.two_qubit_env import TwoQubitWalshEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True,
                        help="Root of the original research repository")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip()
    expected = "321d1ee8b038e3399e44d5d38109c52df1011d7c"
    if revision != expected:
        parser.error(f"Expected source revision {expected}; found {revision}")
    sys.path.insert(0, str(args.upstream / "function_basis_rl"))
    prefix = "walsh_reproduction.walsh_rl.envs."
    OriginalSimulator = importlib.import_module(prefix + "two_qubit_env").TwoQubitWalshEnv
    OriginalEnv = importlib.import_module(prefix + "fidelity_only_multi_agent_env").FidelityOnlyEightAgentEnv
    root = Path(__file__).resolve().parents[1]
    errors = {}

    def compare(name, left, right, atol=1e-9):
        left, right = np.asarray(left), np.asarray(right)
        np.testing.assert_allclose(left, right, atol=atol, rtol=1e-11, err_msg=name)
        errors[name] = max(errors.get(name, 0.0), float(np.max(np.abs(left-right))) if left.size else 0.0)

    steps = 0
    for name in ("su4_segment", "cnot_segment", "su4_fidelity", "su4_global"):
        config = load_config(root / "configs" / f"{name}.yaml")
        for seed in (920000, 920001):
            old, new = OriginalSimulator(config), TwoQubitWalshEnv(config)
            for backend in (old, new):
                backend.reset(seed=seed)
                backend.evaluate_physical_coefficients(backend.base_physical_coefficients)
            compare("target", old.target, new.target)
            compare("noise", old.noise_sample.as_vector(), new.noise_sample.as_vector())
            compare("reference_coefficients", old.base_physical_coefficients, new.base_physical_coefficients)
            compare("reference_channel", old.current_operator, new.current_operator)
            score = old._get_info()["average_gate_fidelity"]
            context = dict(seed=seed, target=old.target.copy(), noise=old.noise_sample.as_vector(),
                           analytic_coefficients=old.base_physical_coefficients.copy(), analytic_agf=score,
                           current_coefficients=old.base_physical_coefficients.copy(),
                           current_superoperator=old.current_operator.copy(), current_agf=score,
                           alpha_commands=np.zeros((2, 8)), g_commands=np.zeros((2, 3)),
                           x_commands=np.zeros((2, 8)), y_commands=np.zeros((2, 8)),
                           cumulative_pulse_evaluations=1)
            rng = np.random.default_rng(seed)
            for _ in range(2):
                for stage in FIDELITY_ONLY_STAGE_ORDER:
                    original, extracted = OriginalEnv(config, active_stage=stage), FidelityOnlyEightAgentEnv(config, active_stage=stage)
                    old_obs, _ = original.reset_from_context(context)
                    new_obs, _ = extracted.reset_from_context(context)
                    compare("observation", old_obs, new_obs, 2e-6)
                    for _ in range(original.spec.steps):
                        action = rng.uniform(-1, 1, original.spec.action_dim)
                        old_result, new_result = original.step(action), extracted.step(action)
                        compare("observation", old_result[0], new_result[0], 2e-6)
                        compare("reward", old_result[1], new_result[1])
                        assert old_result[2:4] == new_result[2:4]
                        compare("retained_channel", original.current_operator, extracted.current_operator)
                        compare("retained_fidelity", original.current_average_gate_fidelity, extracted.current_average_gate_fidelity)
                        compare("coefficients", original.current_coefficients, extracted.current_coefficients, 1e-8)
                        steps += 1
                    context = original.export_context()
                    original.close()
                    extracted.close()
            old.close()
            new.close()
            print(f"Verified {name}, task seed {seed}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(source_revision=revision, tasks=8, cycles_per_task=2,
                                          proposal_steps=steps, max_absolute_differences=errors,
                                          passed=True), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
