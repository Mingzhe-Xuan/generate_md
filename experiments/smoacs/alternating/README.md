# Alternating Crystalformer / MatterSim optimization

Start from exactly the original128 templates, masks and seed42, not the archived
200-update outputs. `run.py` intercepts the pinned author source immediately
before optimization to obtain its original inputs/models, then requires initial
predictions to match the archived CPU206 first history record within1e-3.
Pilot selection is deterministic by atom count(2,4,8,10), never property success.

Each outer iteration performs one Adam update on all five original variables:
scaled lengths,scaled angles,fractional coordinates,element logits and oxidation
logits. The same three Adam parameter groups and momentum states persist across
iterations. Decode the updated elemental distribution by argmax for MatterSim.
Persistent subprocesses in .venv-smoacs-postprocess perform GPU BFGS using the
pinned5M model, FixSymmetry and ExpCellFilter until fmax<=0.01. The1600-step safety
limit is not convergence; an unconverged/failed B stops that candidate explicitly.
CPU A uses the unchanged .venv-smoacs-paper; all processes run in one Slurm job.

B returns the actual cell and ordered fractional positions. Update existing A
parameters in place, retaining Adam state and elemental/oxidation logits. B is
never silently projected back into the old2–10angstrom/60–120degree box. If B
leaves it, expand that candidate's bounds only enough to include B and record the
extension. Reconstructed cell Gram matrices and atom identities are checked.
This is a disclosed new-method boundary rule, not the original paper protocol.

The first200 A updates follow the original temperature and cosine LR schedule.
Afterwards temperature is fixed at0.0001 and learning rates are held at10% of
their initial values, so a zero learning rate cannot simulate convergence.
At a B-relaxed state, true joint stopping requires non-batch-averaged maximum A
gradient<=1e-3, previous A update<=1e-3, B converged, and five consecutive fixed-
temperature rounds with identical composition, positional RMS<=1e-3angstrom,
relative cell change<=1e-4 and gap/Ef changes<=1e-3. Gradient/update tolerances
refer to scaled optimization variables, not physical forces.
The A proposal itself must also move positions/cell by no more than the geometric
tolerances; B returning to the same point alone cannot meet joint convergence.

Ten stable physical rounds with negligible elemental/oxidation gradients but
remaining A gradients are `nonjoint_physical_cycle`, not joint convergence.
This additionally requires elemental/oxidation updates<=1e-5, so ongoing
composition search is not stopped merely because hard assignments stay fixed.
`outer_limit`, `B_unconverged` and `B_failed` remain distinct. Default300 rounds
is a guard, not evidence that convergence is guaranteed. The A and B objectives
can conflict; a common stationary state may not exist.

Outputs under a fresh ignored run directory: protocol.json, results.json,
history.jsonl, per-round A.vasp/B results, B step timings and checkpoint.pt.
Atomic checkpoints distinguish A_ready from B_pending so resuming does not
repeat an already completed A update. Resume uses --resume with the original
run path. Original experimental outputs are never overwritten; no DFT is run.

Submit from the repository root:

```bash
SMOACS_AB_PILOT=1 sbatch experiments/smoacs/alternating/run.sbatch
sbatch experiments/smoacs/alternating/run.sbatch
```

The pilot uses shortened2-round annealing/16-round guard solely to validate
integration. It is not a numerical comparison with the200-update baseline.
Full-run evaluation must retain all128 IDs and report both dual convergence
and final-iterate properties separately. Recompute composition neutrality and
reference coverage because A can change elements; old reference coverage must
not be assumed. Use the same Crystalformer/ALIGNN targets and MP/MatterSim SUN
criteria as earlier, with any missing competitors explicitly unknown.

`evaluate.sbatch` exports final A/B inputs, recomputes common-oxidation neutrality,
independently checks Crystalformer predictions and runs frozen ALIGNN inference.
`references.sbatch` reselects phases for changed final compositions from the same
official snapshot, reuses hash-matched existing5M phases, benchmarks the same
parallel physical relaxer, computes SUN and writes a baseline comparison report.
Reuse of the HTTP range cache does not bypass object ETag/size verification;
the reference job explicitly verifies its direct network route in its allocation.
DFT is not part of either evaluation job despite using the existing compatibility
environment for MP2020 corrections.

Tests: `python -m unittest discover -s experiments/smoacs/alternating`.

Pilot288 exposed large-strain FixSymmetry guards under the original BFGS step.
The production numerical profile therefore uses maxstep0.05angstrom and a1600-
update safety cap, keeping fmax0.01 and all physical constraints unchanged.
Compare against an additional A200+B control with this same B profile; keep the
old maxstep/default400-step baseline visible separately. These are optimizer
numerical changes and are not silently attributed to alternating A/B itself.
