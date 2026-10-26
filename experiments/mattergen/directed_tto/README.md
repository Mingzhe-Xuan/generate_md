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

Primary inspiration (not literal reproduction):
- https://arxiv.org/abs/2105.15183
- https://www.nature.com/articles/s41524-024-01506-0
- https://www.nature.com/articles/s43246-024-00583-4
