# Full128 paired Best-of-N

Input: immutable MatterGen full128 inventory, raw structures, canonical direct
MatterSim relaxation and frozen scores. All128 parents participate with fixed
ordered elements, including63 that fail the legacy neutral-mask criterion.

`plan.py` partitions original inventory order into16 groups of8. `pipeline.sh`
runs three paired RMS noise amplitudes (0.10/0.20/0.30 A), ten perturbations and
one zero control per parent, with and without20-step CUDA CF optimization.
Each allocation runs the three amplitudes concurrently; array concurrency is1.
All8448 MatterSim attempts are submitted through Slurm. Zero controls consume
compute but are excluded from Best-of-10. Proposal cap0.60 A is unchanged from
the amplitude pilot. Seeds use original global parent index, never shard index.

Submit `gate.sbatch`, then `shard.sbatch --array=1-15%1` dependent on gate success,
then CPU `report.sbatch` dependent on array success. Export FULL_BESTN_ROOT to a
new absolute results/mattergen directory for all three jobs. Gate output is also
the first production shard. Failure preserves artifacts and blocks dependents.

`verify_shard.py` checks actual step counts, CUDA use, species preservation,
attempt coverage and frozen predictor controls. `selection.py` ranks CF gap
error only after force+CF formation-energy eligibility; ALIGNN is held out.
Report both legacy validity and a force+formation-energy+separation gate, both
geometry tolerances and both selection orders (best then classify novelty;
best among novel). These are local force/predicted-energy checks, not SUN.

`report.py` verifies complete128-parent coverage and saved geometry hashes,
compares actual relaxed terminals to canonical direct-B, and writes
comparison.json, comparison.md, selection-records.json. Existing full CF200
results are a separate-budget reference. Each selected parent has weight1.
The canonical failed relaxation remains geometry-unknown, not silently novel.

Checks: `python -m pytest experiments/mattergen/full_best_n/test_bestn_full.py`;
native first-shard gate is required before the remaining15 shards can run.
