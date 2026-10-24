# Fixed-element MatterGen optimization with ALIGNN

This module substitutes the actual optimization predictor with the pinned
MEGNet-trained ALIGNN bandgap and formation-energy pair already used for
independent scoring. Input is the same immutable128 raw MatterGen structures;
ordered per-site elements remain exact onehot, with only scaled lattice
lengths, scaled angles and fractional coordinates optimized. The65/63 original
neutral-mask eligibility strata are retained for comparison; both fixed-element
strata bypass elemental/oxidation optimization.

The loss is `relu(abs(ALIGNN_gap-2.5)-0.04)+ALIGNN_eform`; gap predictions are
clipped at zero exactly as in frozen evaluation. Adam base rates0.002 for
lattice variables and0.005 for coordinates,200-update cosine, batch-gradient
normalization by cohort size, bounds and physical B profile match CF200.
Elemental/oxidation interface gradients remain zero. No random perturbation,
Best-of-N, SUN, DFT, phonon or MD stage is included.

`geometry.py` recovers signed integer periodic images from native edge vectors
and recomputes differentiable lengths/cosines. The vendored graph's reverse
image metadata cannot be used directly. `model.py` verifies ZIP/payload hashes,
loads frozen ALIGNN, rebuilds kNN/line-graph connectivity on every forward,
and compares explicit-feature predictions with native ALIGNN. Connectivity
selection is discrete: gradients differentiate geometry on the current graph,
not through neighbor selection. `probe.py` tests lattice/coordinate directional
finite differences on a frozen graph at interior float64 probe coordinates,
plus native-forward equivalence on original and deformed structures.
`device_control.py` compares CPU/GPU arithmetic on identical signed neighbor
connectivity at the unchanged1e-3 tolerance, then records fresh-graph topology
and prediction changes separately. Floating-point geometry differences can
change kNN choices near tied neighbor distances. The optimizer always uses
fresh GPU graphs; the replay is restricted to the numerical device control.
Native architecture equivalence uses identical atom descriptors, lengths and
cosines at the existing1e-5 tolerance. Native graph feature rounding is checked
separately at1e-5 and its prediction effect is recorded. Native ALIGNN consumes
only the norm of edge vectors, so the architecture control encodes those exact
lengths along one axis and supplies the same cosine inputs. This control does
not change any optimization prediction or derivative.

`run.py` reuses verified CF template initialization, the shared GPU Adam200
controller and the same MatterSim workers. ALIGNN supplies every optimization
loss/gradient. CF is a frozen observer at exported poses, so the established
CF inventory controls/evaluation remain valid; its scores never enter the
ALIGNN loss. `alignn-adapter-audit.json` records optimizer-predictor scores at
exported phases, graph refresh counts and native controls separately.

`pilot.sbatch` runs the original4 size/ID-selected templates, the numerical
gradient probe,200 actual GPU updates, B, and both frozen scorers. The full
`full.sbatch` requires `ALIGNN_GRADIENT_PROBE` pointing to the successful pilot
`gradient-probe.json` with matching adapter source hashes. It runs ALIGNN200
on both original strata, then a fresh matched fixed-element CF200 control.
All computation is inside Slurm and existing verified environments/caches.
This CF control is necessary because the old full128 CF200 allowed element
changes, whereas this predictor-swap experiment preserves elements.

`report.py` combines full-denominator raw/A/B property pass counts, joint pass
counts, MAE/RMSE, unknowns and same-ID paired comparisons. ALIGNN is a training
score for ALIGNN200 and CF is its independent architecture check; the roles
reverse for CF200. These frozen predictions are not ground-truth accuracy.
Results are under `results/mattergen/alignn-pilot-<job>` and
`results/mattergen/alignn-full-<job>` with source provenance and evidence SHA.

For a control-only continuation, set `ALIGNN_FULL_RUN` to the existing run and
`ALIGNN_CONTINUE_JOB` to its failed terminal Slurm job. `continuation.py` requires
the neutral65 phase to be completely evaluated, proves unchanged optimizer and
geometry/inference code through Git/AST comparison, and protects its files by
hash. It archives only an unmasked initialization with no A trace/checkpoint/
results/cycle. A fresh valid gradient-probe report remains required. Scientific
partial optimization is never silently archived and restarted.
