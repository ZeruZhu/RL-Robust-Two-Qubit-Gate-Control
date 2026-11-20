# Validation

## Checks Performed

- Local editable installation with Python 3.13.0 on Windows x64.
- Unit tests for Walsh functions, channel propagation, KAK construction,
  projection normalization/null directions, stage dimensions, accept/reject
  behavior, and benchmark dynamics/gradients.
- A PyTorch training-resume round trip covering actor weights, NumPy replay
  arrays, and random-state restoration.
- Source-parity comparison for the four released configurations, two task
  seeds each, and two complete cycles: 896 proposal steps. Targets, noise,
  reference coefficients/channels, actor observations, rewards, retained
  channels/fidelities, and calibrated coefficients agreed exactly in this
  environment. See [the parity report](validation/upstream_parity.json).
- A small end-to-end segment-projection run: four training contexts, two
  validation contexts, two test contexts, two training episodes per stage,
  and two inference cycles. All eight stages completed. First-cycle replay
  discrepancy was zero; evaluation counts were 57 and 113 per task.
- A separate ten-cycle inference check on one smoke task, with 561 total
  evaluations and no decrease in retained fidelity.
- A two-worker inference shard on two smoke tasks.
- One-update GRAPE and CRAB CLI checks using the smoke contexts.
- Standalone Figure 1 build and visual inspection.

The smoke test uses small networks and is a workflow check, not evidence for
the reported fidelity improvements. Its generated checkpoints and numerical
outputs stay under ignored `outputs/`, not in the release data.

## Repeat the Checks

```sh
python -m pytest -q
python scripts/run_pipeline.py --config configs/su4_segment.yaml --run-dir outputs/smoke --cycles 2 --smoke
python scripts/run_benchmark.py --env-config configs/su4_segment.yaml --contexts outputs/smoke/contexts/base/test.npz --method grape --output-dir outputs/smoke_grape --smoke
python scripts/run_benchmark.py --env-config configs/su4_segment.yaml --contexts outputs/smoke/contexts/base/test.npz --method crab --output-dir outputs/smoke_crab --smoke
```

Use new benchmark output directories when rerunning; the benchmark CLI refuses
to overwrite existing results. To repeat the optional source comparison, check
out the exact research revision listed in `provenance.md`, then run:

```sh
python tools/compare_upstream.py --upstream /path/to/research-checkout --output outputs/upstream_parity.json
```

The normal tests do not require the research checkout.

## Not Yet Verified

- Full retraining of the reported training seeds.
- Inference from the original reported checkpoints and complete evaluation data.
- Reproduction of numerical figures and tables.
- A clean dependency installation on other platforms, GPU parity, and full-run runtime.
- The external frozen CRAB source snapshot used by the original launcher.

These limits are tracked in [open items](open_items.md). Local parity on the
checked tasks is not a guarantee of bitwise equality across library versions,
hardware, or every possible target and control.
