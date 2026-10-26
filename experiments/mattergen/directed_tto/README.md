# Directed gradient TTO surviving full relaxation

This module pursues the new gradient-TTO goal. It is separate from the completed
CF-ranked random/beam basin-search experiment. A frozen new16 is sampled uniformly
from the original strict-neutral and canonical-B-converged parents (64 available),
using cohortseed20261026 and original128 order, without property or old outcome
filters. Fixed ordered elements remain unchanged. Every final endpoint uses the
original full free-cell MatterSim5M FixSymmetry/ExpCellFilter/BFGS profile:
fmax0.01, maxstep0.05, cap1600. No short/fixed-cell relaxation is substituted.

## Stage-one experiment

`geometry.py` uses RMS-scaled Cartesian offsets plus six orthonormal symmetric
strain variables in Angstrom-scaled coordinates. The potential gradient includes
the correction between affine stress and offsets held Cartesian. `cf_gradient.py`
uses the pinned original CF models on CUDA, with pure squared gap error to2.5eV;
formation energy is a final hard guard, not a competing gradient objective.
The original Torch Gaussian custom backward omits distance-basis derivatives.
Input-gradient calls reuse the already validated ExactGaussianAverage adapter
from block_alternating/gaussian.py. Ordinary predictions and finite differences
retain the native forward; all16 forward controls must agree at1e-5. The adapter
hash and actual calls are recorded. Earlier Adam experiments already used this
adapter; initial directed gate405 omitted its integration and stopped before any
direction trials. Its failed operators must not be reused.
Gate407 then found one coarse CF difference inconsistent at0.004A while
0.002/0.001/0.0005A converge to the corrected derivative. CF validation therefore
uses both0.001/0.0005A steps with the unchanged absolute0.001/relative0.02 error
criterion, retaining coarse0.004/0.002A differences as diagnostics. Physical and
unrolled derivative checks retain their original0.004/0.002A steps/tolerances.
All14 required primary checks must exist and pass; omitted/duplicate checks fail.
The next gate409 passed0000 but found fine-step inconsistencies on0010.
`numerics.py`/`numerics.sbatch` diagnose the same first4 parents using identical
frozen weights in float32/float64 arithmetic, both one-sided and centered
differences over eight resolutions. This diagnostic never authorizes scientific
trials, changes tolerances or overwrites a failed proof.
Diagnostic411 resolved the ambiguity:24 float64 checks at1e-5/1e-6A match
autograd within5.73e-8, with float32/float64 whole-gradient relative differences
at most1.83e-5 and forward gap differences at most1.04e-6eV. The gate now compares
the actual float32 optimization derivative against these double-arithmetic finite
differences with the original0.001/0.02 tolerances. Mandatory all16 controls also
require same-weight float32/float64 forward agreement1e-5, gradient relative
agreement0.001, and native/adapter double forward agreement1e-10. Scientific
optimization and final scoring remain float32. Old coarse checks remain visible.

`directions.py` compares three actual derivatives: ordinary CF gradient, a positive
soft-Hessian preconditioner, and the gradient through eight physical gradient
steps via finite-force Hessian-vector products. The latter differentiates a
finite lookahead, not the full converged quench. All three face the same finalB.
Implicit differentiation of a fixed isolated minimum with respect to its initial
coordinates can vanish; differentiating short relaxation is a hypothesis to
test, not a promise of nonzero full-relaxation sensitivity.

Each family has descent, exact reverse and random directions. Random atomic and
strain block norms match the descent vector. All three start from one shared
small symmetry-breaking seed (.03A atomic RMS/.003 cell strain), whose directB
zero-gradient control is also evaluated. Four attempted step sizes are
0.05/0.20/0.50/0.80A in the combined metric. Geometric rejections, failures and
stationary directions consume their attempts. This gives576 attempts on16
parents; real MatterSim calculator calls include all curvature/lookahead costs.

`plan.py` freezes IDs, source hashes and policy. `mechanics.py`/`runtime.py` share
two CUDA potential workers for counted energy/force/stress queries and fullB.
`probe.py` checks all16 baseline CF scores/canonicalB replays, Hessian symmetry,
and finite differences for CF inputs, physical pullbacks and unrolled gradients
on the first four predetermined parents, using two epsilon values. `probe.sbatch`
must succeed before `run.sbatch`; no source change is permitted through this
gate without a fresh matching proof. `experiment.py` saves every attempted ray,
then exports equal-budget Best4 selections with the original baseline/seed as
shared fallbacks. `report.py` requires independent frozenCF/ALIGNN controls and
all16 final endpoints, and separately compares gradient against random/reverse.
`direction_stats.py` also compares all unselected proposals afterB, averaging
four scales within each parent before parent bootstrap. Failed/ineligible
attempts retain the common seed/baseline fallback; a complete eligible-pair-only
sensitivity reports its coverage. Per-scale means and failure counts accompany
these descriptive diagnostics. They do not change the Best4 primary criterion.

Initial operational discovery evidence requires at least4 retained error gains
of0.1eV beyond both directB and seed-onlyB, and pairedBest4 CFMAE advantages of
at least0.05eV with negative95%CI against BOTH random and reverse controls.
These controls are necessary to distinguish gradient sign information from
mere extra proposals or a symmetry-breaking noise benefit. Discovery alone is
not goal completion: an independent search seed, repeated optimization steps
and endpoint perturbation/re-quench retention must verify the practical result.
ALIGNN remains held out and reported. No SUN/DFT or phonon claims are made.

If stage one is insufficient, subsequent work should implement feedback gradient
steps and relaxed-ion strain/loading-unloading strategies; do not redefine the
goal as CF-ranking-only random search or leave final structures unrelaxed.

## Iterative feedback experiment

`feedback.py` continues the exact cached plain-family first round of a complete
stage-one run, then adds rounds2/3 for descent, reverse and block-norm-matched
random policies. Each policy recomputes the actual CF gradient at its own fully
relaxed incumbent plus a common small seed, evaluates four unchanged step sizes,
and retains the best physically eligible fullB endpoint including its incumbent
and seed-onlyB. The seed schedule is source_seed+1000*(round-1), shared across
arms; all failures and geometric rejections consume attempts. Original parent
energy/volume references remain fixed across rounds. There are384 fresh ray
attempts and96 fresh seed quenches; the first192 rays and16 common seed quenches
are reused exactly, not silently rerun or relabelled as independent data.

`feedback.sbatch` runs only after the complete source evaluation. It verifies
source/native hashes, cached species and fresh CF scores before continuation.
`feedback_report.py` validates the complete adaptive ledger and independently
scores all16 final structures under both CF and held-out ALIGNN. It reports
round3-versus-round1 changes alongside descent-versus-random/reverse controls.
More rounds alone do not prove gradient direction information. The operational
discovery criterion is unchanged; this extension is exploratory development,
followed by independent confirmation and endpoint re-quenching if positive.

## Relaxation-aware directional sign audit

`relaxation_gradient.py` treats h(x) as MAE/(1+MAE) after a fullB quench when
physical/formation guards pass, and1 otherwise. For a locally fixed CF-informed
axis v, define H(s)=(1/(2sigma)) integral[-sigma,sigma] h(x+(s+t)v)dt. The
fundamental theorem of calculus gives H'(0)=[h(x+sigma*v)-h(x-sigma*v)]/(2sigma),
even if the unsmoothed quench map is locally constant. This is a derivative of
the box-smoothed post-quench objective, not a promise about the unsmoothed fullR.

The first audit reuses stage-one +/-0.5A endpoints to calibrate its sign, with a
0.005 utility deadband, and chooses exactly one held-out +/-0.8A endpoint. All
policies share the original, seed and both calibration points as their fallback
pool. It compares calibrated sign with its reverse, rawCF sign and a predetermined
random sign on all16 paired parents. Held-out scores cannot choose the sign.
This is exploratory posthoc method development; fresh random-axis controls and
native iterative optimization would still be required. `sign_calibration.sbatch`
is a CPU-only afterok analysis, not new CF inference or a changed ongoing run.
General smoothing inspiration: https://arxiv.org/abs/1103.4296 (convex theory;
its convergence guarantees are not claimed for this nonconvex crystal search).

Primary inspiration (not literal reproduction):
- https://arxiv.org/abs/2105.15183
- https://www.nature.com/articles/s41524-024-01506-0
- https://www.nature.com/articles/s43246-024-00583-4
