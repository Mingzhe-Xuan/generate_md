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

## Tighter-force endpoint diagnostic

`tight_relaxation.py` selects all material plain-CF gains (>=0.1eV beyond both
originalB and seed-onlyB, physically eligible), not handpicked favorable cases.
For both original and optimized endpoints it re-quenches the unperturbed pose
plus three paired0.02A/0.003strain perturbations using the same MatterSim5M
potential on CPU, fmax0.001/maxstep0.05/cap1600 and unconstrained free cells.
Fresh CPU/GPU source energy/raw-force controls must pass1e-4 before trials.
`tight_relaxation.sbatch` can overlap GPU experiments without accessing a GPU.

`tight_evaluation.py` exports a variable-size trial inventory, checks original
frozenCF scores on GPU, then uses the established independentCF/ALIGNN evaluator.
`tight_evaluation.sbatch` runs after both the CPU diagnostic and active GPU work.
Reports group the four correlated trials within each selected parent, keep all
failures and require both raw/filter forces<=0.001 plus CFformation<-.5. This
stronger profile diagnoses residual-force or symmetry artifacts; it does not
replace the frozen0.01-force benchmark or certify DFT/phonon/global stability.

## Independent remaining48 validation

`validation_plan.py` freezes ALL remaining48 parents from the original64
strict-neutral/canonical-B-converged pool after excluding the complete pilot16.
It uses original inventory order and searchseed20261027, without property or
old-method outcome filters. `validation_plan.sbatch` runs this preregistration
before any new48 directional outcomes. PlainCF is selected during pilot method
development; the unsuccessful pilot criteria/results remain unchanged.

`validation_policy.py` defines a new primary sufficient-decrease acceptance:
retain the CF Best4 winner only if its fully relaxed error improves>=0.1eV beyond
BOTH originalB and seed-onlyB; otherwise return originalB. Apply the same rule to
descent/reverse/random controls, never using ALIGNN for ranking. OrdinaryBest4 is
a separately reported secondary policy, including smaller threshold crossings.
This conservative variant is motivated by pilot near-tie geometric changes and
weak aggregate cross-model retention; its new48 outcomes must not tune it. The
new48 primary comparisons, raw query costs, held-out ALIGNN and tighter physical
retention are required before attributing practical value to gradient direction.

`validation_run.py` enforces the preregistered manifest SHA and native mathematical
source, then checks all48 frozenCF baselines and same-pose potential values before
any new rays. Raw-start replays remain explicit diagnostics and a separately
evaluated no-CF control, as described below. Same-weight double-arithmetic checks validate actual float32 gradients;
the first four heldout parents require all six atomic/strain/mixed directional
differences. No Hessian or finite-lookahead work is needed for this plain family.
There are576 rays and48 seed quenches, plus48 baseline replays. The feasible
seed-only comparator is best(originalB,rawseedB), matching the pilot control.
All stationary/rejected/failed attempts remain in the ledger and48 denominator.

`evaluation_export.py` exports the actual cohort size with input/terminal hashes,
selection signatures and protected idempotence; incomplete exports are preserved
before rebuilding. `validation.sbatch` runs the native gate/search then seven
unchanged frozenCF/ALIGNN evaluations (two policies times three arms, plus original
baseline), and an eighth fresh raw-replay baseline. `validation_report.py` audits all48 selections, original volume/energy,
ordered species and independent scores, and reports parent-paired confidence
intervals. It checks fresh baseline predictions against available archived values;
ALIGNN unknowns retain their coverage and are never silently dropped from pass
rate denominators. A positive directional flag still requires separate physical
retention checks and cross-model interpretation before goal completion.

`baseline_diagnostic.py` is a separate read-only GPU diagnostic when a raw-start
quench does not reproduce an archived endpoint. It checks the same frozen poses'
energies and raw forces with the unchanged1e-4 tolerances on all48 parents, while
recording raw-path energy/geometry differences separately. Its report alone does
not amend the validation gate or authorize candidate generation; source and
potential consistency must be understood before proceeding.

`baseline_amendment.py` records the explicit pre-outcome amendment after native421:
all48 frozen poses passed unchanged1e-4 energy/raw-fmax controls, while three raw
replays reached different endpoints. This demonstrates same-pose consistency;
the cause of path variability is not established. The original raw-path gate
failed and is never relabeled as passed. The diagnostic is pinned by exact SHA.
The preparation CLI takes `--source` failed420root and `--run` a NEW execution
folder, checks all48 cached replay tasks/record hashes/input/terminal/checkpoint/
profile, copies the original manifest bytes unchanged and records the amendment
before proposals. It refuses any source with scientific candidate outcomes.

Every new native run repeats all48 fixed-pose energy/raw-force controls and the
unchanged CF/gradient checks. All seven original selections remain unchanged;
`raw-replay-baseline` independently scores all48 new direct-B endpoints with CF
and ALIGNN. Reports include parent-paired contrasts against BOTH original and
fresh raw-B, unknown-model coverage, and material gains that also exceed fresh
raw-B by0.1eV. The additional robustness flag requires the original directional
criterion, at least four such gains, and a negative upper pairedCF CI against
fresh raw-B. It cannot turn a failed primary criterion into success. Cached
replay costs, current native/search work and the prior48-call diagnostic are
reported separately. Failed420 and native421 files are read-only throughout.

`validation_retention.py` follows the completed48-parent experiment without
changing its primary result. It includes ALL materially improved retained-descent
parents, including cases that do not beat fresh raw-B or agree with ALIGNN. For
each parent, originalB, fresh raw-B and the selected endpoint each receive an
unperturbed and three paired0.02A/0.003strain unconstrained re-quenches at fmax
0.001/maxstep0.05/cap1600. All three original endpoint energy/raw-fmax CPU/GPU
controls must pass1e-4 before trials. `validation_retention.sbatch` runs this CPU
stage after the source validation; `validation_retention_evaluation.sbatch` runs
frozen sourceCF controls and independentCF/ALIGNN evaluation on GPU afterwards.
Set `RETENTION_SOURCE_ROOT` and a fresh `RETENTION_RUN_ROOT` for the two jobs.

The retention plan pins all six source artifacts and its own implementation.
Every parent has12 trials; failures and unknown predictions stay in the inventory.
Summaries compare selected endpoints against BOTH matched tighter original and
fresh baselines, requiring all four physically eligible pairs for a robust-parent
flag. Trials remain correlated within each parent. Zero material parents produces
an explicit zero-result report, without running an empty model evaluation. These
diagnostics do not turn unsuccessful population directionality into success.

Primary inspiration (not literal reproduction):
- https://arxiv.org/abs/2105.15183
- https://www.nature.com/articles/s41524-024-01506-0
- https://www.nature.com/articles/s43246-024-00583-4
