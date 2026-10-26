# Conditional physical retention of annealed CF gains

This module checks every full16 CF-annealed endpoint with at least 0.1 eV
post-quench MAE improvement beyond its complete shared zero-CF candidate pool.
It does not filter by ALIGNN agreement or the source population success flag.
The first-parent integration run cannot supply this cohort. Source policies,
all five completed evidence files, original native proof and scientific source
hashes must still match. Nothing here changes the source search or its criteria.

`anneal_retention_policy.py` defines selection, complete trial inventories,
physical eligibility and paired CF/ALIGNN summaries. `anneal_retention.py`
validates source artifacts and runs CPU physics or frozen GPU evaluation.

For each material parent, the seven endpoints are original B, seed B, shared
zero-bias best, reverse, random, fixed-bias and CF-annealed best. Each receives
four matched trials: unchanged trial0 and three perturbations with 0.02 A atomic
RMS and 0.003 strain, using the existing seed20261028 mutation implementation.
All endpoints use the same parent/trial random seed. Feasibility resampling can
accept different draws at different endpoints; actual inputs and seeds are saved.
All endpoints use unconstrained-cell BFGS, maxstep0.05, cap1600, raw and filtered
force thresholds0.001 eV/A, the original energy+0.05 eV/atom ceiling and original
volume bounds0.65..1.4. Elements remain fixed. CF formation<-0.5 is required in
paired retention summaries. All same-pose potential controls precede the trials;
saved-pose CF controls precede the unchanged frozen CF/ALIGNN evaluation.

Failed trials and missing predictions stay in the four-trial denominator.
Robust material retention requires all four paired gains>=0.1 eV against
original, seed and zero-bias endpoints. Direction and constant-bias comparisons
are reported separately. Trials from one parent are correlated; conditional
case findings cannot rescue a failed full16 population test or prove SUN/DFT,
phonon or MD stability. New-parent confirmation remains necessary.

Use `physics.sbatch` with `ANNEAL_RETENTION_SOURCE` pointing to the completed
full16 source and a fresh `ANNEAL_RETENTION_ROOT`. Model inference and all
processing require Slurm. CPU physics uses the existing postprocess environment;
`evaluate.sbatch` uses the existing GPU blackwell environment and that same root.
Submit evaluation only after CPU physics completes and material cases exist.
With zero cases the CPU phase records zero work and writes the negative report
and summary archive without allocating a GPU.
