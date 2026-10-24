# Fixed-species CF/MatterSim force blending

For each original MatterGen pose, combine F2 + lambda F1 at every Adam update,
lambda0.1/0.2/0.3. F1 is **minus** the gradient of the original numerical CF
loss `relu(abs(gap-2.5)-0.04)+CF_Ef`; F2 is the raw MatterSim5M negative
**total** energy gradient. No per-atom, RMS or L2 normalization is introduced.
Both contributions receive the same original65/63 cohort mean. This implements
the gradient of `E_MS_total + lambda*L_CF`, input to the existing200-update Adam
cosine schedule, rather than a separate plain Euler displacement.

Elements are exact fixed onehot. Lattice lengths, angles and coordinates use
the same scalers/bounds/base learning rates as the prior fixed CF200 control.
`geometry.py` pulls native unconstrained physical forces/stress back via
`dE/ds = -F H^T`, `dE/dH|s = V H^{-T} sigma`, then through the lattice scalers.
Stress already includes affine atomic motion: adding another Cartesian-force
cell term would double count it. ASE stress follows its tensile-positive
[filter convention](https://docs.ase-lib.org/_modules/ase/filters.html).

`worker.py`/`client.py` maintain two GPU MatterSim services in the existing
postprocess environment, with pinned checkpoint hashes/encoder guards,
stateless current-cell/coordinate/ordered-species hashes and raw E/F/stress.
They do not relax or constrain the optimization pose. `provider.py` combines
the geometry gradients before Adam, records all200 per-case scalar force
diagnostics and stores gradient/pose snapshots at1/25/.../200 for independent
mixing audits. Final endpoint raw forces are recorded; force services stop
before the unchanged final MatterSim B workers start.

`probe.py` checks total-energy directional finite differences of scaled
lengths/angles/fractional coordinates on four slightly perturbed raw parents.
Geometry is float64; the native GPU force model is float32. Epsilon0.001 and
0.003 each must satisfy the predeclared `abs0.05 + rel0.03` tolerance, accounting
for float32 energy noise. Analytic tests are stricter and catch force sign,
stress transpose, normalization and affine double counting. A validated probe
with matching code/model hashes is mandatory for full runs.

Submit `pilot.sbatch` first for the same original4 size/ID-selected poses,
three lambda branches,200 updates/final B/frozen CF and ALIGNN. It reuses the
old matched four-case fixed CF200 control. `full.sbatch` requires
`FORCE_MIX_PROBE` and evaluates each lambda on all original128, keeping the
65/63 strata and failed attempts. It reuses the completed fixed CF200 control
under `results/mattergen/alignn-full-361/cf200`, without rerunning its A/B.
Every model/force evaluation and batch export runs through Slurm. Existing
blackwell/postprocess environments and caches are reused. No SUN/DFT/MD.

`report.py` retains full-denominator property/joint rates, unknowns, MAE/RMSE,
same-ID paired CF200/direct-B comparisons and force magnitude/opposition
statistics. Both scorers are frozen saved-pose evaluations; CF is the training
score and ALIGNN the independent architecture check, neither DFT truth.
The surrogate already mixes bandgap eV and formation-energy eV/atom numerically;
lambda is the explicitly defined numerical weight against total potential
energy, not a universal dimensionless calibration across alternative losses.
