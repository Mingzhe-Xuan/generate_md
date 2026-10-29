# Cross-system confirmation and property response through relaxation

This stage extends the four-system pilot in [RESULTS.md](RESULTS.md), retaining
all failures and original thresholds. The final synthesis and subsequent
forward-readout experiment are in [REPORT.md](REPORT.md).

## Empirical protocols and coverage

Selection477 kept the GSS hash-selection rule and selected one eligible
reference in every system. Property run478 preserved ten completed parents
before a Ti native graph-geometry check failed at1.08489e-5 against the unchanged
1e-5 threshold. Dependent479/480 never started and were cancelled. Strict
continuation481 reused all ten results and the Ti anchor, rechecked native
predictions, recorded Ti as failed and completed the remaining TiO2 parent.
Audit482 completed. All12 attempted parents remain in coverage:11 property
results and1 explicit native-control failure.

Each property probe is0.03angstrom atomic RMS from its relaxed anchor. Fixed-cell,
fixed-species perturbations use random, target-gap-loss descent and formation
energy descent directions, both signs, amplitudes0.005/0.02/0.05angstrom and
force tolerances0.001/0.0002eV/angstrom. There are192 completed pairs and384
converged endpoints. Local independent auditing checks all12 original reference
hashes, ten reused result hashes, both source snapshots, raw/relaxed property
slopes and every endpoint's species, lattice and force convergence.

## Property response retained after relaxation

At amplitude0.02angstrom, median absolute post-/pre-relaxation two-sided
property response in its corresponding descent direction is:

| Property | Gradient-QC parents | fmax0.001 | fmax0.0002 |
|---|---:|---:|---:|
| Gap |9|0.1604%|0.00783%|
| Formation energy |10|0.1365%|0.00587%|

Strict-tolerance parent-bootstrap95% intervals are0.00177%..0.05097% for gap
and0.00145%..0.12605% for formation energy. These small fixed-study cohorts do
not establish population prevalence. At the strict tolerance and0.02angstrom,
all64 endpoint-to-reference ordered MIC distances have median2.46e-5angstrom,
maximum0.00170angstrom. Geometry and property responses contract strongly;
tighter convergence reduces retained response further. This supports a local
relaxation-map explanation, without asserting crystallographic basin identity.

These are predicted property slopes, not optimization success rates or DFT
accuracy. Raw finite steps can be nonlinear. Si has zero gap gradient; other
input-gradient QC failures remain explicit. No fresh-parent replacement or
post hoc clipping is used. Both all-completed and gradient-QC cohorts are in
the full report. The endpoint property pullback is measured as finite response,
not claimed to be an exact derivative through BFGS.

## Frozen twelve-system score confirmation

Confirmation483 used the same12 fixed probe geometries, new orthonormal random
planes (seed20261055), and the reporting window0.0005/0.001angstrom frozen
before these outcomes. Native inference and grad-enabled paths were measured
separately. Of48 planned/evaluated case-times,36 in9systems pass numerical
controls and corroborate nonzero local curl. Among the eight systems outside
the initial pilot,24/32 points in6systems corroborate it. Native Si and Ti are
precision-sensitive; SiO2 changes neighbor graphs. The12 inconclusive points
are not evidence of conservativity. FP64 independently corroborates40/48 but
is not substituted for native results. No full-matrix QC gate is upgraded.

Same-pose MatterSim plane matrices have FD-scale relative differences
0.00015..0.00244 and normalized antisymmetry below0.00142. Tiny-step energy-force
checks pass9/12 parents; their three failures remain explicit, shown with open
markers. The force-plane comparison is diagnostic at those three points.
Raw confirmation auditing checks48NPZ hashes, both precisions/modes,108 physical
ledger points and the script source hash.

This confirms that the learned score approximation cannot everywhere be
identified with an exact scalar potential. It does not exclude a useful
approximately conservative component or useful generative sampling. Exact
posterior/thermodynamic interpretations require conditions beyond high cosine.

## Connectivity counterfactual and interpretation

Some very small structural differences still produce substantial predictor
changes: for Al2O3 a strict0.02angstrom branch differs from its reference by
3.25e-5angstrom ordered RMS yet its predicted gap differs by0.0403eV, with
changed connectivity. Fixed-geometry counterfactual484 completed for all384
endpoints, rechecking384 cached endpoint predictions and22 references within
1e-8. At the exact same endpoint coordinates, replacing the freshly selected
graph with the reference graph changes that Al2O3 gap from4.056656 to4.016354eV;
the reference is4.016361eV. Its0.040295eV total change decomposes into
-0.00000728eV fixed-graph geometry response and0.040302eV connectivity response.
Thus this particular apparent gap change is dominated by graph selection,
without any geometry update in the counterfactual. It is not physical-property
improvement established by DFT.

Among374 endpoints within0.001angstrom ordered RMS of their references,60 have
fresh-graph gap changes above0.001eV. On those same endpoints, median absolute
change falls from0.01254 to0.000189eV with reference connectivity. For formation
energy,24 endpoints above0.001eV/atom fall in median absolute change from0.00427
to0.0000070eV/atom. These are correlated branch-level descriptive counts, not
independent structures or a held-out accuracy benchmark. Freezing connectivity
here diagnoses a local numerical mechanism; it is not proposed as a globally
valid graph representation for large structural changes.

The evidence motivates two distinct forward targets: a property conditioned on
a noisy structure's clean posterior, and a property of the relaxed output.
They are not interchangeable. Exact Gaussian guidance additionally uses
posterior property variance; a point prediction alone can be overconfident.
The independent joint-Gaussian test in test_likelihood_guidance.py verifies that
derivation. Sources include [classifier guidance](https://arxiv.org/abs/2105.05233),
[DPS](https://arxiv.org/abs/2209.14687), and the
[generalized h-transform](https://arxiv.org/abs/2406.01781).

## Evidence and next stage

- Remote `results/mattergen/forward-inverse-property-20261030-v2` and sibling
  `-report`; local `results/mattergen/evidence/forward-inverse-property-20261030`.
- Remote `results/mattergen/forward-inverse-confirm-20261030-v2`; local
  `results/mattergen/evidence/forward-inverse-confirm-20261030/reviewed-v2`.
- Property archiveSHA256
  `9526d842073f9e23a3a490f26aca1b339a2bd65d8219a155996efed78181d1f0`;
  property report archive
  `ef540d6d8167ba9c48ee02a267e47a4b492d6b4992824119d62985f49aa2427b`.
- Confirmation archiveSHA256
  `8a2f57d4c283a393dcf14d63369badce6009c2d3435f07972fe94510e292bd0c`.

Graph evidence: remote `results/mattergen/forward-inverse-graph-20261030`, local
`results/mattergen/evidence/forward-inverse-graph-20261030`. ArchiveSHA256
`c006b10b097c8dd2766745ec21e415f9de85d9854029e03da777264595222ff5`;
source/script hash and all384 response decompositions independently audited.

Completed: broader frozen numerical confirmation, actual forward-property
response through relaxation, and connectivity-only counterfactuals, with
failures and uncertainty. The subsequent frozen forward readout and its
held-out-label controls are now reported in [REPORT.md](REPORT.md): generative
features improve on composition/cell features, while additional noisy training
has no resolved advantage over clean training in this study. MP20 checkpoint structure
exposure remains unknown, so frozen-backbone comparisons must not be advertised
as unseen-structure CSP generalization. No DFT/SUN or old paused job was resumed.
