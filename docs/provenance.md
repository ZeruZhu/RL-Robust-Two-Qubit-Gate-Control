# Provenance

## Research Code

- Upstream repository: <https://github.com/ZeruZhu/RL_Quantum_Gate_Control>
- Selected branch: `paper-experiments-20261014`
- Selected commit: `321d1ee8b038e3399e44d5d38109c52df1011d7c`
- Selection date: 2026-11-20
- Relevant upstream directory: `function_basis_rl/walsh_reproduction/`

Selected components have been extracted from this revision. The
`source_manifest.json` records their original paths, source hashes, and changes.
The research repository remains unchanged. This revision does not by itself
specify the checkpoints or the set of runs behind each result.

## Extraction Changes

- Installed-package imports replace research-workspace path manipulation.
- The simulation backend is restricted to the 27-control, seven-segment
  exchange model. Earlier control parameterizations and observation modes
  are not included. Reference-midpoint projection kernels retain the upstream
  numerical calculations and null-direction safeguard.
- The KAK module retains the exchange-compiled reference convention. The
  original convention name in configurations is preserved for checkpoint
  compatibility; it does not select a separate method in this repository.
- TD3 architecture, action scales, observations, reward, stage order, checkpoint
  format, and materialized-context protocol are retained. The discarded
  alpha-accumulation mode is rejected; all families use accept/reject updates.
- PyTorch loading modes are explicit: inference checkpoints use weights-only
  loading, while trusted training-resume files retain their NumPy/RNG payload.
  A resume round-trip test covers this portability correction.
- Internal `fidelity_only_*` names and serialized keys are retained to preserve
  compatibility with existing checkpoints. They support all three released
  feedback settings, not only the fidelity-only ablation.
- GRAPE and CRAB numerical kernels are separated from cluster orchestration.
  The new benchmark CLI consumes explicit contexts and settings. It does not
  copy cluster paths or the old plotting/aggregation pipeline.
- CLI filenames and experiment labels are simplified. Configuration numerical
  values are retained. Training entry points reject reuse of a run directory
  with a different training configuration before writing metadata. New pipeline
  orchestration, packaging, tests, and documentation are local additions.

The exact source comparison is recorded in `validation/upstream_parity.json`.
The separate frozen CRAB source snapshot used by the original benchmark
launcher is not tracked upstream; its equivalence remains open item A09.

Licensing, contributor attribution, and any third-party notices must be
confirmed before public distribution. No author identity or license is
invented during extraction.

## Overview Figure

The six standalone TikZ source files in `figures/method_overview/source/`
were copied without content changes from the project-maintained overview
figure on 2026-11-20. They are maintained separately from the upstream research
repository. The PDF and PNG in this repository are rebuilt from these sources
using `tools/build_overview.py`; they are not extracted from a manuscript.

No manuscript, submission template, historical figure versions, build logs,
or machine-specific source paths are included.
