# PCGrad on MatterGen CF/MatterSim force blending

Two tasks are the original total MatterSim energy and lambda times the original
CF bandgap/formation-energy loss, lambda0.1/0.2/0.3. For each structure, concatenate
the gradients of scaled lengths, scaled angles and fractional coordinates.
If their dot product is negative, project EACH task against the other ORIGINAL
task. Sum the projected tasks and pass this gradient to unchanged Adam200 cosine.
For two tasks there is only one other task, so projection order is immaterial.
Projection coefficients use F64; returned optimizer gradients retain native F32.
No per-atom/force/RMS normalization is introduced. Fixed element and oxidation
placeholder gradients must stay zero. Structures never share projection factors.

This implements symmetric two-task [PCGrad](https://arxiv.org/abs/2001.06782),
with sum reduction to preserve the exact old update on nonconflicting pairs.
The metric is explicitly the inherited scaled optimization coordinates, including
the physical stress pullback. It is not solely a Cartesian atom-force projection,
not an always-orthogonal update, and does not promise energy-neutral Adam moves.
The old weighted scalar remains a logged reference, not a PCGrad objective.

`compose.py` provides pure `project_pair` and the per-structure logging adapter.
The optional hook in `force_mixing/provider.py` defaults to the previous linear
combination. Run with `force_mixing/run.py --composition pcgrad`, or set
`FORCE_MIX_COMPOSITION=pcgrad` for the existing pipeline. All existing native
CF/MatterSim controls, fixed-species guards and final B settings remain active.
The provider hash changed, so obtain a fresh native physics FD proof in the pilot;
do not reuse the old365 report under its previous hash.

`pilot.sbatch` runs the same four original raw parents0000/0001/0008/0087, all
three weights,200 GPU updates and the unchanged final B/frozen CF/ALIGNN scoring.
It reuses readonly linear pilot365, pure CF200, direct B and raw controls.
`report.py` retains every original ID, failed B/unknown scores, each-property and
joint counts, MAE/RMSE and force/projection/timing diagnostics. All inference runs
inside Slurm, reusing existing environments and weights. No SUN/DFT/MD.

Every step logs original/projected geometry-vector dot products and norms to
`pcgrad-trace.jsonl`; steps1/25/.../200 save original/projected F64 vectors for
independent replay. The existing force snapshots also save the actual F32 mixed
gradient supplied to Adam. `test_pcgrad.py` covers analytic projection, zero and
antiparallel limits, per-structure isolation, inactivity, and immutable species.
