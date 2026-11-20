# Method Overview

This directory contains the standalone overview diagram (Figure 1), its
editable TikZ sources, a vector PDF, and a PNG preview used by the root README.
No manuscript or submission template is required to build it.

## Sources

| File under `source/` | Contents |
| --- | --- |
| `method_overview.tex` | Standalone document and panel composition |
| `styles.tex` | Shared typography, colors, and drawing styles |
| `calibration_step.tex` | Proposal, evaluation, acceptance, and TD3 training |
| `policy_cycles.tex` | Policy sequence and highlighted action |
| `pulse_model.tex` | Seven-segment pulse and Walsh coefficients |
| `projection_feedback.tex` | Two geometric projections and segment features |

## Build

Requirements are Python 3, `pdflatex`, and `pdftoppm` on `PATH`.
The LaTeX distribution must provide `standalone`, `amsmath`, `amssymb`,
`newtxtext`, `newtxmath`, and TikZ with `arrows.meta`, `calc`, and
`shapes.geometric`. Poppler provides `pdftoppm`.

From the repository root:

```sh
python tools/build_overview.py
```

To change the PNG resolution without changing the vector PDF:

```sh
python tools/build_overview.py --dpi 220
```

The script compiles in a temporary directory and publishes the PDF and PNG
only after compilation and rendering succeed. No Python packages beyond the
standard library are needed. Compilation logs and auxiliary files are not
kept in this repository.

## Reading the Diagram

One proposal is not one inference cycle: a complete cycle has 16 proposals
for each of two longitudinal policies and four proposals for each of six
other policies, giving 56 proposals. Each evaluation uses the complete
seven-segment pulse, not a single segment.

Panel (d) illustrates the longitudinal segment features. Its first sketch
projects a transported reference direction onto a six-dimensional space of
local Hermitian matrices. The commutator map then gives a superoperator
direction. The second sketch projects the final channel residual onto the
one-dimensional span of that superoperator. The drawn plane and line are
schematic representations of subspaces, not the dimensions of the matrices.
The 14 cells represent seven segments and two qubits, not separately measured
within-segment errors. The implementation also uses global-projection features;
the transverse policies use their corresponding reference directions.

Feedback is computed from simulated channels. The figure does not establish
a hardware measurement protocol or quantify its measurement cost.
