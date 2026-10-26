# CF gradients through a fixed-species MatterGen denoising map

This module first tests the feasibility of a new directed TTO proposal. It does
not claim that a learned denoising map produces physically stable structures or
that its gradients survive full MatterSim relaxation. Those are separate outcome
tests. Existing Adam, independent48 and guided-ES results remain unchanged.

`mapping.py` defines a symmetric-cell latent geometry and converts the official
score outputs to a one-step clean estimate. Positions use the local wrapped-VE
estimate `wrap(y + std**2 * score)`; on a torus this is a local lifted estimate,
not a globally unique posterior mean. Cells use the affine VP inversion
`(y + std**2 * score - (1-a)*limit_mean)/a`. Cell noise is symmetrized with the
official variance-preserving rule. Ordered atomic numbers are always fixed.

`adapter.py` loads the existing hash-pinned conditional MatterGen checkpoint,
checks cached source against the pinned original archive, and evaluates the
official `DiffusionModule.score_fn` with explicit gap conditioning2.5 and CFG2.
It does not call a sampler decorated with no_grad. Its score Jacobian remains in
the computation graph. CF uses the existing corrected Gaussian input derivative
adapter, with unchanged pretrained weights and98-element vocabulary.

`native.py` tests the first four unchanged development parents at two fixed noise
times (.1/.3), with seeded forward-corruption noise. It checks periodic-distance
preservation under the symmetric-cell rotation, CF predictions on saved original
poses, explicit score conversion, score-free VP roundtrip, fixed species, CUDA,
finite values and derivatives. Whole gradients and tiny finite differences are
tested in same-weight float64 arithmetic, with float32 forward/gradient controls.
Each failure is saved and prevents scientific proposal trials. No search or
post-MatterSim efficacy is inferred from a successful native test.

`precision.py` supplies scoped dtype-only clones for the official OCP graph
helpers during float64 diagnostics: periodic integer offsets and the distance
sorting buffer use the input dtype. Exact source-expression counts are checked,
original/adapted hashes recorded, and every imported alias restored on exit.
Official files and the float32 path remain unchanged. This addresses a native
Float/Double error; it does not change graph cutoffs, neighbor limits or tests.

Submit `native.sbatch` with `DENOISED_SOURCE` pointing to the completed original
directed16 run and `DENOISED_ROOT` to a fresh results directory. The existing
`.venv-mattergen` is reused. All model work runs under Slurm on GPU.

Future efficacy tests must include the identical denoised seed without CF,
direct CF gradients, reverse and block-norm-matched random directions, the same
post-relaxation selection budget, full MatterSim relaxation, original energy/
volume/formation guards, all original parents and held-out ALIGNN. Denoising
itself, extra samples or favorable CF self-scoring cannot establish the value
of the gradient. No new efficacy trial is authorized by this native module.

`diagnostic.py` and `diagnostic.sbatch` investigate a failed native derivative
test without changing it. They trace both ordered and canonical periodic-edge
identities at each finite-difference point, decoded pose changes and CF scores.
A second scan holds the MatterGen score tensors fixed, isolating the remaining
affine decode/CF path. All four original proof parents and both noise times are
kept, with eps1e-4/1e-5/1e-6/1e-7. A completed diagnostic is not a passed native
gate and cannot authorize scientific search. Set `DENOISED_FAILED` to the failed
native run in addition to the source and fresh output variables.

The separate `stable_native.sbatch` explicitly tests a numerical variant using
`(pos1-pos2)-periodic_offset` for graph selection, instead of subtracting an
already-offset position. This removes translation-dependent roundoff in self
images and makes +image/-image squared distances exactly equal. It is a declared
arithmetic change in BOTH precisions, not a changed cutoff, checkpoint or gate.
The original native/diagnostic records stay unchanged. The same four parents,
two times, 1e-5/1e-6 finite differences and thresholds still have to pass.

`double_native.py`/`double_native.sbatch` define a distinct execution choice:
the proposal denoiser and CF gradient run in float64, while official final CF
scoring stays float32. They require all48 original finite differences and
<=1e-5 agreement with the official evaluator on EACH identical saved clean pose.
This separates different decoded geometries from different score arithmetic.
The failed original float32-proxy gates are never relabeled. A pass proves the
actual double proposal derivative, not equivalence to a float32 denoising map
and not post-MatterSim efficacy. Seed-corruption arithmetic is still float32,
fixed and recorded; the subsequent differentiable map is evaluated in float64.

## Full development16 comparison

`denoised_policy.py` freezes two noise times(.1/.3), seed20261032 and four ray
scales(.05/.2/.5/.8). Each operator computes one gradient of half-squared CF gap
error through the denoiser. Five equal-ray-budget arms are full descent, exact
reverse, atomic/cell-block-norm-matched random, stopped-score-Jacobian descent,
and direct CF descent at the identical clean seed. Ordered species stay fixed;
formation is a post-quench screening condition. The direct CF comparator uses
its original RMS/symmetric-strain metric, while decoded arms use symmetric-cell
latent variables; actual decoded displacements/FLOPs need not be equal.

There are32 shared seed quenches and640 rays,672 total attempts. All16 parents
remain, including failed seeds, geometric rejections and force failures. Every
arm shares BOTH seed-B outcomes and originalB before adding its own eight rays;
ALIGNN never ranks. A seed-only output isolates denoising gains. Proposal guards
use minimum distance.7A, maximum translation-corrected atomic displacement2A,
singular-value cell strain.15 and original volume bounds.65..1.4. Final B keeps
the existing .01/raw.01/FixSymmetry/ExpCellFilter/BFGS1600/maxstep.05 profile,
original energy+0.05eV/atom, volume/separation and CF formation<-0.5 guards.

`denoised_search.py` requires the complete actual-double native proof and its
unchanged core hashes. Before any new relaxation, all16 old frozenCF/same-pose
MatterSim controls and all valid decoded seeds' same-pose CF controls are checked. It records
gradient vectors, update directions, saved inputs, terminal hashes, all rejected
attempts and actual NFE. Invalid noisy/decoded seed cells consume all their
attempts rather than removing their parents. `denoised_validate.py` rebuilds
directions,672-query coverage, input geometry/physical guards, shared fallback
choices and NFE. Seven outputs use the original frozen CF/ALIGNN pipeline.

`denoised_report.py` keeps all16 denominators and parent-paired confidence
intervals. Directional evidence requires>=4 material gains>=.1eV beyond the
already-queried seed pool and>=.05eV mean/negative95%CI against seed-only,
reverse AND random. Jacobian value additionally requires>=.05eV/negativeCI
against stopped-score and direct-CF controls. Even a positive development study
requires new-parent confirmation and matched tighter physical retention.

Submit `search.sbatch` with `DENOISED_SOURCE`=the original directed16 run,
`DENOISED_NATIVE`=the completed actual-double native run and a fresh
`DENOISED_ROOT`. `--export-only` can finish a complete validated search without
repeating optimization. Reporter changes may be synchronized separately; all
scientific core hashes stay frozen during a run. No SUN/DFT is added.

`denoised_shared.py` loads legacy helpers under unique names and restores the
process's prior `policy` module and import path. This prevents the alternating
and basin-search policy files from silently changing these comparison rules.
