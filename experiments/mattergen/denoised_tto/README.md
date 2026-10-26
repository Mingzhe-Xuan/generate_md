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
