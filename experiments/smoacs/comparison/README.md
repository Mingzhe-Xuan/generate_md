# MatterSim before/after comparison

This module adds a paired comparison to the existing CPU206 reproduction.
The two stages are original generated geometry and MatterSim-relaxed geometry;
they do not denote switching property predictors. DFT remains paused.

Interfaces:

- `singlepoint.py`: immutable candidate inventory + pinned MatterSim5M checkpoint
  -> original-geometry single-point energies and input/model/environment hashes.
  It does not optimize the original geometry. All128 inputs are attempted, even
  those whose later relaxation failed. Existing completed relaxation is reused.
- `evaluate.py`: inventory, single-point directory, existing same-predictor
  predictions.csv/summary.json, relaxation directory, and completed MP reference
  selection -> per-candidate before/after records and comparison summary.
  Both stages use the identical relaxed reference pool, MP2020Compatibility
  algorithm, StructureMatcher settings, and E_hull<0.1eV/atom threshold from
  `../references/assess.py`. Missing competitors and terminals stay explicit.
  The full-inventory baseline also audits selected phases omitted from reference
  relaxation because their proposals failed. These phases count as missing
  competitors for those original geometries; they cannot silently certify S/N.
- `paired_statistics.py`: identity validation, threshold classification, fixed
  denominator summaries, pass/fail transitions and paired continuous changes.
- `singlepoint.sbatch` / `evaluate.sbatch`: GPU inference / CPU hull evaluation
  in Slurm. The latter uses the existing paper environment for compatibility
  correction; it does not run DFT or use DFT band gaps.

The report has two cohorts: the full original inventory, with missing post-relax
values unknown, and the identical subset with successful converged relaxation.
Uniqueness is recomputed within the paired cohort on both sides: an earlier
failed sample must not suppress a surviving sample's before-stage uniqueness.
Each metric includes pass/fail/unknown counts, lower/upper pass-rate bounds and
retained/lost/gained/neither/unknown transitions. Never interpret a lower-bound
change as an exact change when unknown outcomes remain. Continuous mean/median
changes use only candidates with both finite values, and report that pair count.

Metrics are stability, uniqueness, novelty, S.U.N., band-gap target success,
formation-energy target success, structural validity, joint property success
(including the original composition-neutrality and0.5angstrom separation rules),
stable+property and S.U.N.+property. Composition neutrality is read from the
hash-verified author CSV and held fixed because relaxation preserves composition.
Continuous values include raw MatterSim energy/atom, E_hull, predicted gap,
absolute gap-target error, and predicted formation energy. Gap defaults to
2.5+/-0.2eV and formation energy to <-0.5eV/atom for CPU206. This is a
same-predictor sensitivity study, not independent property validation. Stability
is thermodynamic relative to the selected references, not a phonon/MD guarantee.
The expectation of improved stability and reduced property success is a
hypothesis to test, not a result built into the code. MP2020 structural corrections
can vary with geometry; both stages apply the same correction algorithm.

Submit `singlepoint.sbatch` after reference relaxation to avoid GPU contention.
Export `SMOACS_COMPARE_INVENTORY` and `SMOACS_COMPARE_ROOT` first. For assessment,
also export `SMOACS_REFERENCE_SELECTION`, `SMOACS_COMPARE_SINGLEPOINT`,
`SMOACS_COMPARE_PREDICTOR`, `SMOACS_COMPARE_RELAXATION`, and
`SMOACS_COMPARE_AUTHOR_CSV`, `SMOACS_COMPARE_BENCHMARK`; submit with `afterok` dependencies on reference
completion and the single-point job. CPU206 reuses validated predictor-245, so
there is no new model or different before/after property evaluation protocol.
Outputs belong under ignored results/smoacs directories, never normal Git.

Validation: `python -m unittest discover -s experiments/smoacs/comparison`.
The integration fixture checks an analytic elemental hull with complete
reference coverage, a failed relaxation retained in the full denominator,
stability improvement, property loss, joint success, and paired uniqueness.
