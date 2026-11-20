# Reproduction

## Environment

Install the package as described in the root README. `pyproject.toml` declares
the dependencies; `requirements-tested.txt` records the direct versions
tested locally. It is not a complete transitive lockfile. The local validation
used Windows x64, Python 3.13.0, and CPU execution, with an editable install in
a virtual environment reusing installed dependencies. A fresh dependency
installation on Linux and GPU execution have not yet been validated.

PyTorch is required for training and checkpoint inference, but not for the
numerical simulator or GRAPE/CRAB. For a numerical-only installation:

```sh
python -m pip install -c requirements-tested.txt -e ".[test]"
```

Only load checkpoints from trusted sources. Policy and training-state files
use PyTorch serialization; verify the release checksums before loading them.
Inference policies use weights-only loading. Full training-resume files also
contain NumPy replay and random-state arrays, and require pickle-based loading;
only resume from files created by your own run or a trusted, verified source.

## Training and Inference

The full pipeline generates the configured contexts, trains the eight stages
in order, materializes their output contexts, and evaluates frozen policies:

```sh
python scripts/run_pipeline.py --config configs/su4_segment.yaml --run-dir outputs/su4_segment_seed17 --seed 17 --cycles 10 --device cpu
```

Use the other configurations to select fixed-target training or an observation
ablation. The seed argument changes the training seed; evaluation task seeds
remain those specified in the configuration. `--device cuda` selects an
available CUDA device, while the quantum environment still uses CPU NumPy/SciPy.

Each configuration specifies 20,000 training contexts, 256 validation contexts,
2,048 test contexts, and 20,000 training episodes per stage. Every alpha stage
has 16 proposals; each other stage has four. Full runs are substantially more
expensive than the smoke test. Production runtime and hardware measurements
are awaiting the original run information; no runtime estimate is asserted.

For an existing bundle, inference can be run without training:

```sh
python scripts/evaluate_cycles.py --base-cache-dir outputs/su4_segment_seed17/contexts/base --final-cache-dir outputs/su4_segment_seed17/stages/y_1/materialized_cache --stage-root outputs/su4_segment_seed17/stages --output-dir outputs/su4_segment_seed17/eval10 --num-cycles 10 --device cpu
```

Each stage directory must contain `source_config.yaml` and
`best_policy/policy.pt`. The evaluator checks configuration signatures,
checkpoint stage order, task alignment, and replay of the first cycle against
the final materialized cache. It records checkpoint hashes with the output.
It does not silently substitute new checkpoints or evaluation tasks.

There are 56 proposals per inference cycle, so the evaluation count per task,
including the initial reference evaluation, is `1 + 56 * cycles`. This count
does not include measurement shots or training-time validation calls.

### Individual Stages

```sh
python scripts/prepare_contexts.py --config configs/su4_segment.yaml --output-dir outputs/example/contexts/base
python scripts/train_stage.py --config configs/su4_segment.yaml --active-stage alpha_0 --input-cache-dir outputs/example/contexts/base --run-dir outputs/example/stages/alpha_0 --device cpu
```

The next stage takes the previous stage's `materialized_cache` as input.
The order is `alpha_0, alpha_1, g_0, g_1, x_0, x_1, y_0, y_1`.
For a tiny stage test, use `--smoke` on both commands. Do not mix smoke and
full-run outputs in the same directory. Existing request checks reject many
configuration mismatches; use separate directories for separate runs.

## Detuning Study

The detuning contexts preserve target sampling and dissipative rates while
changing the sampled coherent offsets. The generator uses 40 bins, 400 tasks
per bin, and the upstream sampling sequence:

```sh
python scripts/prepare_detuning_contexts.py --config configs/su4_segment.yaml --reference-cache outputs/su4_segment_seed17/contexts/base --output-dir outputs/detuning_contexts
python scripts/evaluate_cycle_shard.py --base-cache-dir outputs/detuning_contexts --stage-root outputs/su4_segment_seed17/stages --output-dir outputs/detuning_eval --start-index 0 --stop-index 16000 --num-cycles 10 --use-cached-noise --device cpu
```

`--use-cached-noise` is essential here: the detunings are the explicitly
generated context values, not a fresh Gaussian sample. Target and dissipative
rate validation remain enabled. Shard outputs retain per-task values; the
final plotting/aggregation handoff remains with the numerical-figure package.

## GRAPE and CRAB

The benchmark entry point consumes the same context NPZ files used by the
calibration pipeline. A single task is evaluated by default:

```sh
python scripts/run_benchmark.py --env-config configs/su4_segment.yaml --contexts outputs/su4_segment_seed17/contexts/base/test.npz --method grape --output-dir outputs/grape_example
python scripts/run_benchmark.py --env-config configs/su4_segment.yaml --contexts outputs/su4_segment_seed17/contexts/base/test.npz --method crab --output-dir outputs/crab_example
```

Use `--limit 2048` for the full task set, or `--smoke` for one optimizer update
with a one-node quadrature rule. The GRAPE smoke test also reduces 140 bins
to 14. Smoke outputs are marked in `request.json` and must not be used as
reported results.

`configs/benchmarks.yaml` retains the checked-in numerical settings: GRAPE
uses projected Adam with an Armijo line search, and CRAB uses a continuous
Lindblad adjoint with L-BFGS-B and two Fourier frequencies per control.
Both optimize the noise-distribution objective. Held-out task detunings are
used for evaluation, not to select the optimized pulse. GRAPE reuses its
optimized pulse for identical target/reference pairs, including the fixed
CNOT setting. CRAB retains task-seeded Fourier frequencies.

CRAB finite-differences only the algebraic amplitude-constraint map. Its
dynamical gradient uses the continuous adjoint. Gradient checks, iteration
traces, final pulse parameters, and per-task fidelities are written to the
requested output directory. These are numerical simulations, not assertions
about hardware gradient or fidelity measurement costs.

The original CRAB launcher loaded an external frozen source snapshot that is
not present in the selected Git checkout. The artifact uses the checked-in
Fourier/dynamics implementation and passes numerical checks, but equality
with that external snapshot still requires confirmation (open item A09).

## Result Data and Figures

No scientific results are inferred from smoke runs. Exact released checkpoints,
evaluation contexts, and numerical plot/table inputs remain pending. Do not
regenerate the result figures from newly sampled tasks as a substitute for
that material. Figure 1 is independently reproducible using
`python tools/build_overview.py`.
