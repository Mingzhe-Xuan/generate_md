# Forward prediction and inverse crystal generation

Research module for the active user goal: derive and empirically test deeper
links between predictive energy/property models and a generative CSP backbone.
The existing 192-state and16-trajectory probes show direction correlation, not
that MatterGen is a physical force field. New work must go beyond that evidence.

## Theoretical distinctions and falsifiable connections

Let `x` denote continuous coordinates with fixed species/cell in a local chart,
`F=-grad U`, and `s_t=grad log p_t`. Noise covariance and coordinate metric must
be specified. Fractional row coordinates obey `r=f L`; a score covector becomes
`s_r=s_f L^{-T}`, while a sampling displacement becomes `d_r=d_f L`.

1. If and only if the modeled clean density is the physical Boltzmann density
   under the stated measure, its clean score is `s_0=beta F`. Gaussian noise
   gives `s_t(x)=beta E[F(X_0)|X_t=x]`, under the integration-by-parts boundary
   conditions. This is a posterior average, not generally `beta F(x)`.
   Curated equilibrium-minimum databases are not demonstrated thermal ensembles.
2. In an SPD harmonic basin with Hessian H and additive covariance C, the exact
   score curvature is `K_t=(beta^{-1} H^{-1}+C)^{-1}`. An empirical delta at a
   minimum instead gives `K_t=C^{-1}`. Both can restore a perturbed structure;
   high force/score cosine alone distinguishes neither. Local Jacobians,
   Hessians, noise dependence and held-out predictions can test this connection.
3. For Gaussian corruption, `E[X_0|x]=x+C s_t(x)` and
   `grad E[Y_0|x]=C^{-1} Cov(X_0,Y_0|x)`. Tilting the clean distribution by a
   likelihood w gives `s_w-s=C^{-1}(E_w[X_0|x]-E[X_0|x])`. Thus a noisy-state
   property guide should be a posterior quantity; a clean-only predictor's raw
   gradient on arbitrary noisy structures is not automatically this guide.
4. For a converged relaxation map R into an isolated basin,
   `grad(y composed with R)=J_R^T grad y(R)`. Within a basin J_R can contract
   strongly; raw property descent can disappear after relaxation. Basin changes
   and finite optimizer tolerances must be measured, not assumed.
5. Even exact conservative s and F can yield a nonconservative field after
   pointwise removal of their parallel component. Orthogonality implies zero
   first-order energy change; finite steps can increase energy at second order.

There is also a second-order link which needs no thermal assumption. For fixed
Gaussian corruption covariance C, the clean posterior mean m and covariance V
satisfy `D m=I+C J_s` and `V=C+C J_s C`. An exact score therefore has a symmetric
Jacobian and a positive-semidefinite covariance proxy. A learned vector field
need not satisfy either condition. Numerical derivative checks must precede
interpreting violations, and the proxy must not be symmetrized/clipped and then
presented as a validated posterior covariance.

For a linear clean property `y=a^T x_0`, the noisy posterior property gradient
is `(I+C J_s)^T a = C^{-1} V a`. In the harmonic Boltzmann special case it is
`(I+beta H C)^{-1}a`: uncertain/soft and stiff coordinates receive different
guidance. For nonlinear properties, `y(E[x_0|x])` generally differs from
`E[y(x_0)|x]`; applying this linear formula to an arbitrary clean predictor is
an approximation requiring a separate test. `test_posterior_geometry.py` checks
the two identities independently using anisotropic Gaussian mixtures, central
differences, and the law of total covariance. The practical opportunity is a
noise-aware posterior property predictor with uncertainty, not an assumed
physical interpretation for an arbitrary learned score.

These are analytic connections/conditional hypotheses, not claims that this
checkpoint obeys a Boltzmann law, has a known temperature, is integrable, or that
a new guidance method improves CSP. Wrapped coordinates admit the Gaussian
identities only in a local chart or with the appropriate periodic formulation.
Joint lattice diffusion makes fixed-cell probes conditional slices, not tests
of the complete joint density. Negative Hessian modes invalidate a stable-well
Boltzmann fit and must remain visible in reports.

## Prior primary literature

- [Zaidi et al., pre-training via denoising](https://arxiv.org/abs/2206.00133):
  equilibrium-structure denoising, Gaussian-mixture force interpretation and
  improved forward property representations.
- [Two for One](https://arxiv.org/abs/2302.00600): physical force connections for
  MD-trained diffusion models; conservative versus vector-valued networks.
- [Fractional Denoising](https://arxiv.org/abs/2307.10683): isotropic noise is
  limited; molecular anisotropy and noise design matter.
- [Generative Pseudo-Force Fields](https://arxiv.org/abs/2605.19050): geometric
  restoration pseudo-forces need not equal the physical PES gradient.
- [Boltzmann Generators](https://arxiv.org/abs/1812.01729): energy/distribution
  connections require thermodynamic training/reweighting assumptions.
- [GSS](https://arxiv.org/abs/2604.27636): joint score/MLFF search is prior art;
  its existence does not establish the mechanisms tested here.

## Interfaces and validation plan

`relations.py` supplies coordinate/translation bases, exact harmonic and mixture
posterior calculations, finite-difference Jacobians and integrability controls.
`test_relations.py` checks identities with anisotropic, noncommuting matrices,
finite differences and explicit counterexamples. Native-model evidence belongs
in ignored `results/mattergen/forward-inverse-*` with source/asset hashes.

The planned local-response experiment uses the pinned CSP checkpoint and
MatterSim5M on property-blind GSS anchors. It compares full centered Cartesian
score Jacobians and physical Hessians at two displacement scales; reports
antisymmetry, stable/unstable modes and covariance-aware predictions, then tests
raw versus post-relaxation property responses. Pilot gates precede expansion
across the12 chemical systems. All model work is CUDA under Slurm. Existing
environments/data are reused; no DFT or old paused dimer job is resumed.

`prepare_anchors.py` selects hash-ordered eligible references (2..12 atoms,
ordered, MatterSim-supported) with no property/energy ranking. `csp_model.py`
loads strict native FP32 and an FP64 copy with the pre-existing dtype-only
compatibility helpers, recording rebuilt periodic graph fingerprints and
raw-to-score conversions. No stable-offset arithmetic rewrite is enabled.
`physics_worker.py` uses unconstrained BFGS/ExpCellFilter at0.001eV/angstrom,
800-step cap; this research map differs from the earlier symmetry-constrained
canonical B. Its counted RPC is isolated in the existing postprocess environment.

`probe.py` builds full translation-removed native-FP32 autograd Jacobians and
finite-difference Jacobians/Hessians at two displacement scales, with fresh
graphs at every input. FP32 replay, full-matrix AD versus FD, directional adjoint
FD, and force versus energy checks assess the pilot. Neighbor changes and
negative physical modes remain explicit. FP64 is a secondary diagnostic: an
observed same-graph0.69% FP32/FP64 field difference failed the original0.5%
equivalence gate, so FP64 copies are not substituted for the requested native
model. That diagnostic threshold is unchanged. Native derivative QC is reported
separately from execution success; a run is not physically validated merely
because its code finished. All raw adjoint and FD vectors are retained. The probe
center is displaced0.01angstrom atomic RMS from a relaxed anchor to avoid
assuming a learned density maximum exactly coincides with a physical minimum.

`analyze.py` reconstructs every saved physical/score finite-difference matrix
and native adjoint matrix from raw vectors, verifies vector hashes, and retains
all selected structures in coverage. Noise-law fits use all cases and a separate
native-QC cohort. Harmonic-law fits require the entire centered physical Hessian
to be positive definite (minimum eigenvalue >1e-4eV/angstrom^2); no negative
modes are removed to manufacture a harmonic distribution. A second cohort also
requires native derivative QC and physical FD/antisymmetry <=0.02. All methods
in a cohort share its structures. Coefficients leave entire chemical systems
out; fewer than three eligible systems produces an unavailable result. This is
a fixed-cell Gamma-point local curvature probe, not a full phonon or thermal
stability assessment. `analysis.sbatch` creates report.json and PNG/SVG plots in
a new directory via FORWARD_INVERSE_ROOT and FORWARD_INVERSE_ANALYSIS.

`precision_response.py` reuses every original completed probe pose, centered
basis and random test directions. It compares full native/FP64 autograd matrices
and directional central differences at0.000125..0.032angstrom, saving both dtypes'
raw fields, adjoints, graph hashes and repeats. This separates precision noise,
graph changes and finite-step nonlinearity; a failed0.005 native-value equivalence
check stays failed. It does not rerun anchor relaxation or replace native output
with FP64 results. Submit response.sbatch with FORWARD_INVERSE_ROOT and a fresh
FORWARD_INVERSE_RESPONSE directory. Numerical failures are not evidence against
a physical hypothesis until an adequate measurement regime is demonstrated.

`response_report.py` independently rebuilds the directional responses and
adjoint matrices, verifies hashes and separates graph/precision checks from
nonzero cross-curl evidence. The0.0005/0.001angstrom reporting window is an
exploratory pilot choice, not a preregistered population test. Raw fields and
all nine step sizes remain available. Nonzero two-direction curl does not
certify the entire Jacobian. See [RESULTS.md](RESULTS.md) for current evidence
and the still-incomplete broader research plan.

`property_model.py` reuses the pinned ALIGNN geometry adapter/weights with
native architecture controls and FP64 Cartesian input derivatives on GPU.
`property_pullback.py` tests raw versus post-relaxation property responses on
hash-selected references from all12 GSS systems. Each full-cell anchor is
perturbed0.03angstrom atomic RMS; around this same probe, random, target-gap-loss
descent and formation-energy descent directions have unit atomic RMS. Both signs
at0.005/0.02/0.05angstrom are evaluated before and after fixed-cell unconstrained
MatterSim relaxation at0.001 and0.0002eV/angstrom. Zero gradients and unconverged
structures remain explicit in coverage. Fixed/fresh graph FD checks accompany
the property gradients. Ordered-site MIC displacement is a continuous response
measure, not proof of basin or crystallographic equivalence. This is a local
mechanism test, not a full gradient optimizer or a held-out property-accuracy
benchmark. `pullback_math.py` computes finite-map contraction, raw/relaxed slopes
and pullback checks; undefined ratios at raw slope<1e-5 stay undefined. Submit
select.sbatch with FORWARD_INVERSE_SYSTEMS=all, then property.sbatch afterok,
using a fresh FORWARD_INVERSE_ROOT. All model computation remains in Slurm.

`property_report.py` independently reconstructs raw/relaxed property slopes,
checks endpoint forces/species/cell and input-gradient FD controls, and reports
all completed versus gradient-QC cohorts separately. Ratios retain zero-response
denominators; bootstrap intervals resample whole parents. Its Slurm wrapper uses
FORWARD_INVERSE_ROOT and fresh FORWARD_INVERSE_REPORT.

`confirm_curl.py` consumes the property experiment's immutable12-system probes,
with new seed20261055 orthonormal random planes and frozen0.0005/0.001angstrom
steps. It checks two-plane AD versus FD in native inference and grad-enabled
paths separately, reports FP64 as a distinct diagnostic, and evaluates MatterSim
on the same plane as a conservative control. This is not full-matrix derivative
certification. Counts for the eight systems beyond the initial pilot stay
separate. Submit confirm.sbatch after the property job with
FORWARD_INVERSE_SOURCE and fresh FORWARD_INVERSE_CONFIRM; all inference is GPU
under Slurm. No anchors are relaxed again and no structure is reselected.

If property execution stops after terminal parents, `--resume-from` (or
FORWARD_INVERSE_RESUME in property.sbatch) requires a new output root and the
exact same selection bytes/scientific policy. Every terminal parent is carried,
including unsuccessful ones; completed physics is never rerun. Pinned property
native controls and saved predictions are rechecked, and validated existing
anchors are reused. Only a specifically identified native graph-geometry control
failure becomes an explicit per-parent failure; its threshold stays1e-5, other
unexpected errors still abort. The original failed run and its ledger are
archived with the continuation. This preserves all12 attempted parents rather
than selecting replacements or weakening a failed control.

```bash
export FORWARD_INVERSE_ROOT=results/mattergen/forward-inverse-20261029-pilot
selection=$(sbatch --parsable experiments/mattergen/forward_inverse/select.sbatch)
sbatch --dependency=afterok:$selection experiments/mattergen/forward_inverse/probe.sbatch
```

```bash
python -m unittest discover -s experiments/mattergen/forward_inverse -p 'test_*.py'
```
