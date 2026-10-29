# MatterGen score / physical and property direction probe

This module diagnoses frozen MatterGen scores. It does not change sampling or
perform TTO, relaxation, SUN, DFT, training, or model selection.

`run_probe.py` freezes the existing property-blind16 cohort from the directed-TTO
plan. Each raw MatterGen and direct-MatterSim relaxed anchor supplies three
position-only diffusion times and two noise replicas: 192 states. Atomic numbers
and each anchor's cell stay fixed. Paired raw/relaxed anchors share noise seeds.
Time is the official diffusion time, not a Cartesian perturbation magnitude;
actual Cartesian RMS displacement is recorded. Cells use the existing symmetric
lattice orientation, preserving the physical structure and fractional positions.

`probe_models.py` reuses cached, hash-pinned models and existing environments:
MatterGen and F64 ALIGNN in `.venv-mattergen`; two CUDA MatterSim workers in
`.venv-smoacs-postprocess`. ALIGNN uses a fresh signed periodic kNN graph in every
state. The first replica at t=0.15 of the first four raw parents requires native-forward agreement, two-scale F64
fixed-graph finite differences, rebuilt-graph diagnostics and two-scale MatterSim
force/energy controls. Graph discontinuities are reported rather than converted
into a physical gradient claim. Small-step evaluations check local predictions;
F32 MatterSim energy differences at very small steps can be quantized.

F is negative MatterSim total energy gradient. G is negative ALIGNN
half-squared bandgap target error gradient (target 2.5eV), including its existing
nonnegative gap clamp. A clamped/stationary G is recorded as zero/unknown cosine.
All atom vectors are Cartesian covectors with translation removed. A fractional
score is converted with inverse-transposed cell; raw network output must first
be converted using the official noise standard deviation. Discrete type logits
and lattice scores are outside this diagnostic.

Both representations are saved: the Cartesian score covector `s_f L^{-T}`,
and the Cartesian pushforward of the sampler's positional update `s_f L` (named
`drift_*`). They differ for anisotropic/skew cells: coordinate diffusion induces
a Cartesian metric. For the fixed-cell positional VE process the positive scalar
diffusion variance does not change the update direction. Fits are reported for
both, avoiding conflation of a density gradient with a sampling displacement.

The conditional and disabled-condition branches use the SAME fine-tuned
`dft_band_gap` checkpoint, not two separately trained networks. Delta is their
difference, and guided is unconditional + 2*delta. The ideal conditional-score
identity does not establish real bandgap gradient accuracy at noisy structures.

`probe_math.py` and `report_probe.py` compare unit directions. Per-state projection
is descriptive; predictive fits use global coefficients, no intercept, and
leave-one-parent-out validation separately at each anchor/time. All replicas of
a parent stay in one fold. Unit target vectors make zero prediction's MSE one;
negative held-out R2 is retained. Force-only, property-only, joint, known noise
restoration and combined controls distinguish geometric restoration from
additional property alignment. Coefficients are unconstrained; a negative
property coefficient indicates opposition, not successful guidance. Near
collinearity is exposed through ranks/singular values. Coordinate-permuted
property vectors are a norm-preserving negative alignment control, not a
composition-exchangeable statistical test. Parent bootstrap intervals describe
paired errors; they do not refit cross-validation folds.

Outputs: immutable input/source hashes in `plan.json`, per-state NPZ vectors,
`progress.json`, native controls/provenance in `complete.json`, fitted tables in
`report.json` / `report.md`, and `probe.png` / `probe.svg`. Failure roots are
preserved and new attempts must use a fresh root. `report_probe.py` verifies
coverage, vector hashes, finite data and branch identities before analysis.

Run all model compute through Slurm from the repository root:

```bash
export SCORE_PROBE_SOURCE=results/mattergen/directed-tto-20261026-doublefd
export SCORE_PROBE_ROOT=results/mattergen/score-probe-20261029-seed49
sbatch experiments/mattergen/score_probe/run.sbatch
```

Local tests (no real models):

```bash
python -m unittest discover -s experiments/mattergen/score_probe -p 'test_*.py' -v
```
