# Paired 800-step cutoff versus full 1000-step CSP relaxation

This module replays the same GSS12 compositions, seeds and three native batches
as `csp_trajectory`. It preserves the original1000-step time grid and captures
after800 completed corrector+predictor rounds, rather than changing N to800.
The old saved predictor800 input includes corrector800 and is not that boundary.

`boundary.py` observes native `_mask_replace` outputs without changing returned
values or CPU/CUDA RNG. After1600 replacements it records both `mean800` (the
output omitting the last predictor's added noise, consistent with native terminal
output semantics) and `sample800` (the actual state passed to the next round).
The third branch is the native returned `mean1000`. It records2000 score calls,
2000 replacements per batch, fixed ordered species and the sample800 equality
to the next corrector input. Original terminal differences are diagnostic only:
CUDA execution can be nondeterministic; every comparison is paired within this
single replay, never between an old endpoint and a new early-stop endpoint.

All72 inputs get the same pinned MatterSim5M potential and unconstrained
coordinate+cell relaxation: ASE BFGS/ExpCellFilter, fmax0.01, maxstep0.05,
cap1600 steps, no symmetry constraints. The common isolated physics worker now
saves optimizer logs/trajectories, filter forces and stress for relax requests;
point requests and optimizer settings are unchanged. Geometry-guard exclusions,
unconverged/failed runs remain in all24-parent denominators. Convergence for
endpoint comparison requires both raw atomic and filter maximum force <=0.01.

`run.py` generates cutoff inputs and performs paired GPU relaxations; `report.py`
audits input SHA, all records/NPZ/VASP geometry, force/energy convergence and
pair coverage, then compares per-atom energies, volumes, strict and conventional
StructureMatcher matches, and fixed-order internal MIC displacements. A strict
match alone is not an exact energy-minimum identity claim. No DFT, SUN, CF or
target-property optimization is run.

Submit `run.sbatch` then `report.sbatch` afterok with environment:

```bash
export CSP_EARLY_SOURCE=results/mattergen/csp-trajectory-20261030
export CSP_EARLY_ROOT=results/mattergen/csp-early-relax-20261030
sbatch --export=ALL experiments/mattergen/csp_early_relax/run.sbatch
```

Outputs include `inputs/*.npz`, immutable source hashes, sampling and relaxation
progress, full physics ledger/optimizer trajectories, report JSON/Markdown,
PNG/SVG and an evidence archive. Local numeric/observer tests:
`python -m unittest discover -s experiments/mattergen/csp_early_relax -p 'test_*.py'`.

The coarse optimizer converged on72/72 inputs, but raw Cartesian force checks
passed only21/24 mean800,20/24 sample800 and23/24 mean1000 endpoints. To remove
this residual-force confound, `refine.py` uniformly continues ALL72 endpoints
with fmax0.001 in a fresh root, keeping the original0.01 physical acceptance
criterion unchanged. The report verifies exact coarse-to-fine pose lineage and
shows cumulative steps/time/NFE. It preserves the complete coarse run and never
selects only advantageous outcomes for tighter refinement. Submit `refine.sbatch`
with `CSP_EARLY_COARSE` and a fresh `CSP_EARLY_ROOT`, then the same CPU report.

Final endpoint geometry support is checked independently of force convergence,
with the original domain thresholds (volume/atom2 A^3, pair distance0.5 A,
cell minimum singular value0.3 A). Relaxed cells are reduced before the condition
check because integer basis shear does not change the physical lattice. Collapsed
endpoints remain failed geometries in all24-parent denominators and total cost;
they do not enter ordinary energy/structure comparisons. The initial force-only
report/archive is copied to a sibling backup before this geometry-aware audit.

`physical.py` / `physical.sbatch` add strict primitive-crystal matching without
volume rescaling after the geometry-aware report. They distinguish equivalent
supercell representations from actual crystal differences and preserve the
fixed-cell matcher separately. Outputs are `physical-matches.json`,
`PHYSICAL_COMPARISON.md`, and a separately hashable `verified-evidence.tar.gz`.
