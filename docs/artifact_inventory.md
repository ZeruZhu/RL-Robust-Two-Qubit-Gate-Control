# Artifact Inventory

This is an extraction inventory, not a claim that the numerical experiments
can already be reproduced from this repository. Paths below are relative to
the upstream `function_basis_rl/walsh_reproduction/` directory unless noted.

| Component | Candidate source | Status |
| --- | --- | --- |
| Method overview (Figure 1) | Separately maintained standalone TikZ figure | Included with a PDF, PNG, and build command |
| Quantum dynamics, fidelity, and Walsh basis | `walsh_rl/quantum_ops.py`, `noise.py`, `walsh_basis.py` | Included; numerical tests pass |
| KAK reference and seven-segment controls | `walsh_rl/baselines/two_qubit_kak.py`, `controls.py`, `fidelity_only_controls.py` | Included; old parameterizations excluded |
| Calibration environment and projections | `walsh_rl/envs/fidelity_only_multi_agent_env.py`, `two_qubit_env.py` | Supported model extracted; source-parity check passes |
| TD3 training and cycle inference | `walsh_rl/training/fidelity_only_td3.py`, `fidelity_only_multi_agent.py`, `fidelity_only_cycles.py` | Included; eight-stage smoke run and frozen-cycle replay pass |
| Context generation, training, and evaluation entry points | `scripts/prepare_fidelity_only_contexts.py`, `prepare_binned_detuning_contexts.py`, `train_fidelity_only_stage.py`, `evaluate_fidelity_only_cycles.py` | Included under concise CLI names with four explicit configurations |
| GRAPE comparison | `GRAPE/scripts/exchange_grape.py` | Numerical core and portable task runner included; full-result verification pending |
| CRAB comparison | `GRAPE/scripts/crab/crab_lbfgs.py`, `continuous_adjoint.py`, and Fourier/dynamics helpers | Numerical core and portable runner included; external source snapshot confirmation pending |
| Numerical result figures and tables | `paper_figures/data/` and corresponding plotting scripts | Awaiting collaborator-provided release selection; no changes here |
| Trained policies and exact evaluation inputs | Referenced by run manifests but not tracked upstream | Missing; see open items |
| Focused numerical tests | Relevant tests under `tests/`, plus extraction-specific tests | Included and exercised locally |

## Inclusion Rules

- Include the main segment-projection method and the fidelity-only and global
  projection ablations used in the reported experiments.
- Include both Haar-random SU(4) and target-specific CNOT experiments, together
  with the reported detuning study and numerical comparisons.
- Retain shared numerical helpers when they are required by these methods;
  remove unrelated options only after checking dependencies and behavior.
- Pin configurations, checkpoints, evaluation contexts, and run membership
  explicitly. A directory gaining a newly completed run must not silently
  change the released figures or statistics.
- Keep figure reproduction separate from retraining and policy evaluation.
- Do not copy the manuscript, submission materials, historical outputs,
  discarded approaches, or unrelated single-qubit experiments.

## Verification Levels

1. **Overview figure:** build the standalone diagram from included sources.
2. **Numerical figures and tables:** regenerate them from frozen, included data
   without retraining or running policy inference.
3. **Policy evaluation:** run released checkpoints on the exact evaluation
   contexts and compare per-task outputs with the reference results.
4. **Training:** provide complete configurations, commands, dependencies,
   compute requirements, and expected variability across seeds.

Figure 1, the numerical code, and a small training/inference workflow are
verified locally. Exact-result reproduction remains pending the external
materials listed above. See `validation.md` for the checks and their limits.
