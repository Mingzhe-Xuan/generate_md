# GSS paired CSP early stopping and relaxation, 2026-10-30

The800-round cutoff followed by MatterSim relaxation is not an equivalent
replacement for full1000-round sampling in this24-path experiment. Most paired
energies are nearly identical, but some crystal geometries and local endpoints
differ; early stopping also increases total measured relaxation work.

Protocol: twelve GSS fixed-composition systems, two paths each, three native
coupled batches. Preserve the native1000-step time grid and observe after800
completed corrector+predictor rounds. Compare the native last-predictor mean at
800 with the mean at1000; sample800 is a separate stochastic-state control.
Every comparison is paired within the same replay. CUDA replay did not exactly
reproduce the old run (maximum old/new cell difference4.05 A), so old endpoints
are not mixed into any pair and conclusions are limited to this replay cohort.

All72 inputs use identical MatterSim5M coordinate+cell BFGS/ExpCellFilter
relaxation (.01 filter threshold, maxstep.05, cap1600), followed by identical
tighter continuation (.001) for ALL72 endpoints. Original .01 raw+filter force
acceptance is unchanged. All72 pass force acceptance after continuation, but
mean800 parent9 and sample800 parent8 have collapsed carbon cells and fail the
existing volume/atom>=2 A^3 geometry domain. Failed geometries remain in total
cost and all24-parent denominators, not in ordinary paired energy comparisons.

| Metric | mean800 + relaxation | sample800 + relaxation | mean1000 + relaxation |
|---|---:|---:|---:|
| Force + geometry-valid endpoints | 23/24 | 23/24 | 24/24 |
| Strict physical crystal matches to1000 | 16/23 | 16/23 | Reference |
| Paired energy difference within1meV/atom | 19/23 | 18/23 | Reference |
| Mean signed paired energy difference, meV/atom | +12.615 | +12.648 | Reference |
| Largest paired energy increase, meV/atom | +272.007 | +272.007 | Reference |
| Mean cumulative BFGS steps (valid endpoints) | 94.57 | 109.43 | 48.21 |
| Median cumulative BFGS steps (valid endpoints) | 63 | 62 | 29 |
| Total cumulative relaxation seconds (all24 attempts) | 170.93 | 263.01 | 63.23 |
| Estimated generation + relaxation seconds (all24 attempts) | 336.18 | 428.26 | 269.58 |
| Cumulative MatterSim evaluations (all24 attempts) | 2648 | 3957 | 1205 |

The mean energy increase is dominated by one Si parent17 (+0.2720 eV/atom).
For mean800, other >1meV increases are AlPN parent4 (+7.91meV/atom), C parent8
(+2.48meV/atom), and LiCoO2 parent15 (+7.96meV/atom). No early endpoint improves
by more than1meV/atom in the valid paired cohort. Seven pairs per early branch
fail strict physical matching; this alone does not establish seven new phases.

Physical matching uses ltol.02/stol.05/angle1deg, primitive reduction enabled,
volume rescaling disabled. This recovers equivalent supercell representations
that the fixed-simulation-cell strict matcher labels differently (14/23 fixed
cell matches versus16/23 physical matches). It is distinct from the conventional
looser matcher. Geometry guards are applied symmetrically; cell conditioning
uses a reduced lattice basis. These are MatterSim energies, not DFT/hull results.

Generation timing uses the three native replay prefixes:165.25s at800 versus
206.36s at1000. Adding all actual relaxation work gives an estimated24.7% cost
increase for mean800,58.9% for sample800. This excludes model loading/queue and
includes failed geometry costs. It is not a separate end-to-end timing rerun.

Successful Slurm phases:496(coarse),498(initial force audit),499(all72 tighter
continuations),501/502(geometry-aware audits),503/504(physical matching/export).
The initial497 audit dtype error was repaired by exact native float32 norm
reconstruction, with unchanged audit/physical thresholds. Prior force-only
reports and archives are preserved. All10 local tests and independent local
archive/input/energy/cohort/structure/force audits pass; cumulative plot inspected.

Final evidence on Guqq2: `results/mattergen/csp-early-relax-20261030-refined`.
Local final evidence: `results/mattergen/evidence/csp-early-relax-20261030-final`.
Final verified archive SHA256:
`0e6c11a203bd65dedb3e72027aa6f275903d72d3874aa9f5994596fdfa609318`.
Local artifacts: `PHYSICAL_COMPARISON.md`, `REPORT.md`, `comparison.png/svg`,
`physical-matches.json`, `report.json`, `complete.json`, `local-audit.json`.
