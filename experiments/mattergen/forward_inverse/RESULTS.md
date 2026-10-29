# Forward prediction and inverse generation: ongoing mechanistic study

This is an exploratory pilot, not completion of the broader research goal.
No new guidance improvement, benchmark generalization, physical temperature,
full phonon stability or DFT result is claimed.

## Connections with different assumptions

For fixed species/cell, let x be continuous coordinates in a local chart,
C the Gaussian corruption covariance and s its exact noisy density score.

- **No thermal assumption:** the clean posterior mean m and covariance V obey
  `m=x+C s`, `D m=I+C J_s`, and `V=C+C J_s C`.
- **Forward properties:** `grad E[y_0|x]=C^{-1}Cov(x_0,y_0|x)`. For a linear
  clean property `a^T x_0`, this becomes `(I+C J_s)^T a`.
  Nonlinear `y(E[x_0|x])` generally differs from `E[y(x_0)|x]`.
- **Additional Boltzmann assumption:** `s_0=beta F` and, after Gaussian
  corruption, `s_t=beta E[F(x_0)|x_t]`, rather than generally `beta F(x_t)`.
  Within a harmonic well, `-J_s=(beta^{-1}H^{-1}+C)^{-1}`.
- **Relaxation:** for a converged structural relaxation map R, a property of
  its output has gradient `J_R^T grad y(R(x))`. A raw property gradient need
  not survive relaxation inside a basin.

The posterior identities, anisotropic harmonic special case, and explicit
counterexamples to naive force projection have exact or finite-difference
unit checks. These verify the derivations, not the learned model assumptions.
Curated equilibrium structures are not automatically a thermal ensemble.

Primary literature supports the distinction: [Two for One](https://arxiv.org/abs/2302.00600)
learns force fields from MD distributions and studies conservative networks;
[denoising pretraining](https://arxiv.org/abs/2206.00133) relates equilibrium
structure denoising to useful property representations under a distribution
approximation; [Fractional Denoising](https://arxiv.org/abs/2307.10683) emphasizes
anisotropy and corruption design. [GSS](https://arxiv.org/abs/2604.27636) already
combines crystal score and physical-force search: that combination is prior art.

## Measured pilot

Source selection seed20261052, GSS references hash-ordered without property or
energy selection. One eligible2..12atom structure each for Si, C, TiO2 and
CaTiO3. Same checkpoint, original poses and random test directions throughout.
Species fixed. MatterSim5M anchors relaxed without symmetry constraints to
0.001eV/angstrom; probes displaced0.01angstrom atomic RMS. Full centered
Cartesian matrices remove translations only. Times0.001,0.05,0.15,0.30.

| Stage | Authoritative job | Outcome |
|---|---|---|
| Original selection |470|4 fixed anchors|
| Initial execution |471|Failed before physics: virtualenv executable symlink resolution; fixed without installing packages|
| FP64-primary attempt |472|Stopped at original0.005 native-value equivalence gate; preserved|
| Precision field diagnostic |473|Same graph, non-negligible FP32/FP64 field discrepancy for Si; no equivalence assumption|
| Native full-matrix pilot |474|4/4 execution complete, original full-matrix numerical QC0/16|
| Raw audit/archive |475|Hash and matrix reconstruction passed; full-QC physical-law cohort unavailable|
| Same-pose nine-step response |476|Both dtypes, full AD, two original random directions, all16 case-times complete|

MatterSim Hessian FD-scale changes are about0.03%..0.09%. The score pilot mixes
neighbor-graph transitions and FP32 sensitivity; automatic differentiation
alone was therefore not accepted as independent numerical validation. Its
full-matrix QC remains failed, and those failures were not relabeled by the
subsequent narrower diagnostic.

The response diagnostic saves nine displacement norms0.000125..0.032angstrom,
all rebuilt graph fingerprints, repeats, full adjoints, and raw plus/minus
fields. Native and same-weight FP64 are reported separately. Si remains
precision-sensitive; its native results cannot be replaced by FP64 findings.

For an exploratory reporting window0.0005/0.001angstrom, both original random
directions must retain their graph and have <=0.02 relative AD/FD error. Nonzero
cross-curl is corroborated only if it has the same sign at both step sizes and
exceeds five times their observed AD/FD discrepancy or step-to-step spread.
This is empirical numerical corroboration, not a rigorous derivative error bound.
The window was chosen after inspecting the pilot; freeze it before independent
confirmation and do not treat counts as population prevalence.

| System | Native nonzero cross-curl corroborated /4 times | Interpretation |
|---|---:|---|
| Si |0/4|Precision-sensitive native response remains inconclusive|
| C |4/4|Two-direction nonconservative response corroborated|
| TiO2 |4/4|Two-direction nonconservative response corroborated|
| CaTiO3 |3/4|At time0.05 the selected-plane curl is too small relative to numerical discrepancy|
| All selected |11/16|Pilot coverage retained; not a whole-space or population claim|

FP64 corroborates15/16 by the same rule. This is secondary evidence, not an
upgrade to native equivalence. Native AD antisymmetric fractions on the three
non-Si structures range about0.107..0.630; only the tested planes have independent
FD corroboration, not every matrix component.

Where confirmed, nonzero local curl excludes an exact scalar-density or
scalar-energy gradient representation of the learned vector field at that
location. Approximate score-matching networks can still generate useful samples;
this does not show that forward and inverse models have no relationship.
It does show why exact force or posterior-covariance interpretations require
more than a high cosine on a generated trajectory.

The algebraic posterior covariance proxy also has negative symmetric eigenvalues
in this pilot. Because full-matrix FD QC failed, this remains an exploratory
matrix diagnostic, not a separately validated uncertainty measurement.

## Reproducible evidence and remaining work

Remote roots:

- `results/mattergen/forward-inverse-20261029-pilot-v3`
- `results/mattergen/forward-inverse-20261029-pilot-v3-analysis`
- `results/mattergen/forward-inverse-response-20261030`

Local evidence roots under `results/mattergen/evidence/` have the same suffixes
`forward-inverse-20261029` and `forward-inverse-response-20261030`. The latter's
`reviewed/report.json` and `reviewed/response.png` contain independently rebuilt
directional statistics and the precision/step-size figure. Original evidence
archives are immutable and hash-checked.

Pilot archiveSHA256:
`eefd7e2aed1411da96adfe4604414a5cbb80833f5cfe51aa6f1aa9c8bedc365a`.
Response archiveSHA256:
`018635659628cdd72213957c7b48b1d397b5bdc469f977a7f25cabd5555e9ae5`.
Local pilot audit checks13 source blobs and independently reconstructs64 raw
matrices. The response report checks all vector hashes, raw directional
responses and adjoint orientations. Current module suite:37 tests.

The subsequent twelve-system confirmation and relaxation-map measurements are
reported in [FOLLOWUP.md](FOLLOWUP.md), including the completed graph
counterfactual. Remaining: assess a noise-aware forward readout with held-out labels,
controls and uncertainty. MP20 checkpoint structure exposure remains unknown,
so frozen-backbone comparisons must not be advertised as unseen-structure CSP
generalization. The broader goal remains active.
