# CF-guided gradients of the post-quench objective

This is a new method-development experiment on the unchanged original16 neutral
parents. It does not replace or reinterpret the independent48 validation. CF and
MatterSim run on the Slurm GPU; ALIGNN is an evaluator, never a ranking signal.

`guided_settings.py` freezes parents, seed20261031, budgets and decision criteria.
`guided_math.py` samples trace-one Gaussian perturbations in the original
translation-free atomic-RMS/cell-strain metric. With dimension d=3N+3, covariance
is alpha*P/d+(1-alpha)*u*u.T: u is the normalized analytic CF gradient axis for
CF-guided ES or a block-norm-matched random axis for the control. Isotropic ES
uses alpha1; guided methods use alpha0.5. All share the same normal/scalar draws.
A stationary CF guide makes all three covariance variants isotropic.

Each family spends eight probes(four antithetic pairs), with sigma0.8A RMS.
Every probe is fully relaxed before computing bounded utility MAE/(1+MAE).
Geometric rejection, nonconvergence, prediction failure or failed physical/CF
formation guards receives utility1 and consumes its attempt without resampling.
The estimator is sum(noise*(utility_plus-utility_minus))/(2*pairs*sigma^2).
This is a preconditioned gradient estimate of a smoothed post-quench objective;
it is not exact differentiation through B. The CF prior supplies an axis, and
the relaxed outcomes supply the estimated sign. No descent guarantee is claimed
for the finite-sample, normalized, constrained crystal update.

Four heldout step sizes0.2/0.5/0.8/1.2 test estimated descent, exact reverse,
block-norm-matched random and raw analytic CF descent. Every arm within a family
shares the original/seed/eight-probe best fallback. Thus better sampled probes
alone cannot establish value from a gradient update. All16 parents remain:
384 probe attempts+768 heldout rays=1152, plus16 shared seed attempts. Final B
keeps the original FixSymmetry/ExpCellFilter/BFGS profile, fmax0.01/maxstep0.05/
cap1600, rawforce0.01, formation<-0.5, original energy+0.05eV/atom and volume
0.65..1.4 guards. Ordered elemental identities stay fixed.

`guided_common.py` checks old-source hashes and exposes the unchanged native
interfaces. `guided_run.py` freezes a plan in a fresh results folder, verifies
all16 frozenCF/energy/raw-force baselines and same-weight float32/float64 gradient
controls, and requires the first four parents'24 tiny finite differences before
any scientific candidate. It then runs probes/search and exports16 output sets
through the established variable-size CF/ALIGNN evaluator. `guided_validate.py`
reconstructs all random draws, antithetic estimates, directions, common-pool
choices and exact query/NFE coverage. Failed native gates preserve their evidence
and authorize no proposals; use a fresh run for a corrected native experiment.
`--export-only` can finish exports of a complete search without repeating B.

`guided_report.py` reports all16-denominator properties, heldout ALIGNN and
parent-paired confidence intervals. Development evidence requires four material
CF gains>=0.1 beyond the already-queried probe best, an average>=0.05eV beyond
probe best, and>=0.05eV advantages with negative95%CI against reverse/random.
Each family uses its own physical-eligibility mask for this update test.
CF-prior importance separately requires>=0.05eV and negativeCI against BOTH
isotropic/random-guided ES, for BOTH final MAE and improvement beyond each own
probe pool. These strict pilot flags never certify goal completion; independent
confirmation, physical retention and honest cross-model interpretation remain.

Submit `guided.sbatch` with `QUENCH_GUIDED_SOURCE` pointing to the completed
original directed16 run and `QUENCH_GUIDED_ROOT` pointing to a fresh results
folder. All source edits are local and synchronized through Git before submission.
No new models, environments, SUN or DFT are required.

Primary method reference: [Maheswaranathan et al., ICML2019, Guided Evolutionary
Strategies](https://proceedings.mlr.press/v97/maheswaranathan19a.html), equations1-2
and the guiding covariance. Its benchmark results are not crystal-search evidence.
