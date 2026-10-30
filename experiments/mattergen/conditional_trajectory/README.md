# Bandgap-conditioned native trajectories

`run.py` observes the unchanged official 1000-round joint generator using cached
`dft_band_gap` weights. Three common batches of four paths are sampled under CFG2
(target 2.5 eV) and CFG0 (same weights, unconditional embeddings), with paired seeds
and exactly audited common initial priors. Atom counts use native ALEX_MP_20 prior.
This is not fixed-composition CSP. Native CFG2 collates conditional/unconditional
graphs into one doubled forward; both arms make 2000 native forwards per batch.
The RNG-neutral observer records prior, predictor inputs every five rounds plus
999, and the actual returned terminal mean at 1000 (203 states/path).

`evaluate.py` performs fresh-graph pinned ALIGNN gap/formation-energy inference on
GPU, and compatible MatterSim single points in an isolated worker. MASK101 is
never replaced by terminal types; ALIGNN supports Z<=98, MatterSim Z<=94. Geometry
guards require finite arrays, positive determinant, minimum cell singular value
0.3 A, V/N>=2 A^3, and distinct-atom minimum periodic separation>=0.5 A. Atomic
species/logits can remain unresolved early; invalid/unsupported states are recorded
as missing. Fixed prespecified terminal parents 0/4/8 per arm receive native ALIGNN
architecture controls. There is no relaxation, CF, training, DFT or SUN.

`metrics.py` supplies coverage, available-case and paired-parent target MAE, plus
cohorts evaluable at EVERY observed state from a fixed cutoff in BOTH arms.
`report.py` audits raw states, priors, model support, hashes and physics ledgers,
then exports CSV/JSON/PNG/SVG/PDF. Phase windows are descriptive, not inferred
physical boundaries. ALIGNN target MAE measures adherence to 2.5 eV, not DFT
prediction error. Noisy-state ALIGNN is uncalibrated; changing composition also
confounds comparisons of absolute MatterSim energy per atom. Three coupled
sampling batches per arm do not support treating 12 paths as independent replicates.

Run from repository root, after local tests and code synchronization:

```bash
export GAP_TRAJECTORY_ROOT=results/mattergen/conditional-gap-20261030
job=$(sbatch --parsable --export=ALL experiments/mattergen/conditional_trajectory/run.sbatch)
sbatch --dependency=afterok:$job --export=ALL experiments/mattergen/conditional_trajectory/report.sbatch
```

Tests: `python -m unittest discover -s experiments/mattergen/conditional_trajectory -p 'test_*.py'`.
Native observation regression tests live in `random_score_probe/test_trajectory.py`.
All remote compute, including audit and archive creation, must use Slurm.

`phase.py` adds event descriptions from the existing observations: first fully
known composition, first eligible property prediction, final composition sustained
through the terminal, same-path first-to-terminal errors and nonmonotonic changes.
Its second figure aligns errors to type completion and shows nearest separation,
MatterSim force and ALIGNN formation energy. These are descriptive probes; native
conditioning is active throughout sampling, including stages where an external
bandgap prediction is undefined. Every-five-round observations limit event timing
resolution. `phase.sbatch` can depend on the completed primary audit job; it never
changes or reruns generation/inference.
