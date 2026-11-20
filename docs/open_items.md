# Open Items

Recorded on 2026-11-20. Suggested owners are provisional; the project authors
can assign them. Missing items are documented here rather than inferred or
silently replaced. Do not include submission details or private machine paths
in updates to this file.

| ID | Item | Material or decision needed | Suggested owner |
| --- | --- | --- | --- |
| A01 | Released policy checkpoints | Exact checkpoints for each released training seed and stage, with file hashes and configuration associations | Training/results collaborator |
| A02 | Evaluation contexts and outputs | Exact targets, cached noise configurations, and required evaluation arrays, or verified regeneration commands; record ordering and seeds | Training/results collaborator |
| A03 | Numerical figure and table package | Authoritative source data, plotting scripts, dependencies, and figure-to-data mapping for all figures other than the overview | Plotting collaborator |
| A04 | Frozen run membership | Confirm which completed seeds enter each result and whether newly completed CNOT seed 43 should be included | Results collaborator / authors |
| A05 | Benchmark reproduction inputs | Checked-in optimizer settings and a portable runner are included; exact original inputs, saved optimized pulses, and result-to-configuration mapping are still needed | Benchmark collaborator |
| A06 | Numerical code extraction | Initial extraction complete; 896-proposal source-parity check and the eight-stage smoke pipeline pass. Original checkpoint/data replay remains covered by A01/A02 | Repository preparation |
| A07 | Reproduction environment | Local CPU versions, tests, and commands are documented; clean cross-platform installation, GPU validation, and original production runtime/hardware information remain pending | Repository preparation, with training collaborator |
| A08 | Licensing and citation | Author-approved licensing, third-party notices, contributor attribution, and citation metadata | Authors |
| A09 | Frozen CRAB source snapshot | Original L-BFGS-B launcher loads a frozen Fourier/dynamics source tree outside the tracked repository. Supply it or confirm that it matches the checked-in kernels used here | Benchmark collaborator |

## Run and Figure Version Notes

The selected upstream plotting package records six SU(4) segment-projection
seeds (`17, 29, 43, 61, 73, 101`) and four target-specific CNOT seeds
(`17, 29, 89, 101`). Fidelity-only and global-projection ablation data, and the
detuning-grid study, use seed 17 in that package. These groups must not be
described as having identical seed coverage without additional data.

The latest upstream revision marks CNOT seed 43 complete, but the frozen
plotting package does not include that run. Completion metadata alone is not
enough to incorporate it: its evaluation outputs and the intended release
selection must be confirmed.

The overview is included here. The numerical observation-ablation histogram
also has different versions in the current project materials. The plotting
collaborator should provide the authoritative figure together with its source
data. No numerical figures, statistics, or manuscript text are changed by this
repository preparation.

## Completed Preparation

- Connected the local artifact repository to the supplied GitHub remote.
- Included the supported numerical code, portable scripts, four RL configurations,
  benchmark settings, tests, and a tested-dependency constraints file.
- Verified the extracted simulator against the source revision and completed
  a small training/inference run. See `validation.md`.
- Kept generated smoke outputs, environments, and checkpoints out of Git.
- Left numerical plotting materials and the manuscript unchanged.

## Data Handoff

For each supplied artifact, record its filename, purpose, source revision or
run identifier, seed, configuration, checksum, and the command that consumes
it. Large files may be distributed separately, but the released repository
must provide their stable location and integrity checks. Do not use private
cluster paths as download or reproduction instructions.
