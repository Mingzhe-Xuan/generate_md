# Paired fixed-species random-start Best-of-N pilot

Four immutable original MatterGen templates are selected by nearest4/8/12/20
atom counts and ID, without property/outcome selection. For each original
parent, generate ten coordinate perturbations plus one zero-perturbation
control. The RMS sequence is0.10,0.10,0.05,0.10,0.20,0.10,0.10,0.05,0.10,0.20A:
two0.05/six0.10/two0.20. Gaussian vectors have their mean removed, are scaled
to exact per-atom RMS and redrawn if any site moves more than2RMS. Initial
periodic separations must be at least0.5A. All branch seeds/draw counts/input
hashes are stored. No initial cell perturbations are used in this first pilot.

Both arms use exactly those44 inputs: direct MatterSim B, and CF20 then B.
Every atom keeps its original element. CF20 retains the existing loss
`relu(abs(gap-2.5)-.04)+CF_eform`, original Adam base rates and temperature0.01.
Each branch starts fresh Adam state and uses a full20-update cosine schedule.
The vectorized implementation has separate per-case parameters/moments and
gradient normalization1 to avoid changes in branch optimization when N changes.
Original parents are independently checked against their native frozen-CF
reference before initializing branches; branch geometry/onehot/CPU-GPU and
microbatch controls follow, then independent frozen CF/ALIGNN pose predictions.

During CF proposals, total per-site displacement from the unperturbed parent is
capped0.4A; lengths stay within2% and angles within2degrees, intersected with
the original framework ranges. Atom collisions below0.5A cause rollback and
explicit candidate failure. The final physical B is allowed to explore beyond
these proposal bounds. Both B arms use the same pinned MatterSim5M,
FixSymmetry/ExpCellFilter/BFGS,maxstep0.05,fmax0.01,1600-step safety cap. Space
group detection remains the existing per-input policy in both arms; this is
not a controlled fixed-space-group comparison. No relaxation is recomputed
solely to change reported outcomes.

The user's eligibility rule is local force convergence plus predicted formation
energy: actual optimizer convergence, filtered AND raw atomic maximum force
<=0.01, valid structure, and frozen CF formation energy<-0.5eV/atom. It does
not establish thermodynamic/phonon/MD stability. No hull, SUN, MD or DFT is run.
Among eligible perturbed terminals, choose smallest CF absolute gap error to
2.5eV; ties prefer lower MatterSim energy, then branch index. Selection never
uses ALIGNN. ALIGNN predicts the already-selected structure independently.

Best-of-10 excludes the zero control. Each parent contributes one selection or
an explicit no-eligible outcome; branch failures retain their attempt and parent
coverage is always shown. Report selected-parent MAE, RMSE, success/unknowns,
paired same-parent differences, prefixN=1/3/5/10 and distinct eligible terminals.
Dedup uses StructureMatcherltol.02/stol.05/angle1degree/primitiveFalse/scaleFalse
as a diagnostic. Different selected structures from the two arms are expected;
the original parent and starting perturbations are paired. Four parents support
feasibility observations only. MatterSim energy relative to the unperturbed-B
baseline is diagnostic and is not an additional eligibility threshold.

Submit `sbatch experiments/mattergen/best_of_n/pilot.sbatch` from the repository
root. Existing blackwell and postprocess environments and model caches are
reused. `prepare.py`, `run_raw.py`, evaluations and `report.py` run through Slurm.
Results and the final evidence archive are under `results/mattergen/bestN-<job>`.
Completed CF/B/prediction phases can be reused for report-only continuation.

## Equal-budget amplitude sweep

Submit `sbatch experiments/mattergen/best_of_n/sweep.sbatch` for the same four
parents with ten branches at each constant RMS0.10/0.20/0.30A, plus four zero
controls per amplitude. Both noCF and CF20 use each exact starting pose. Seed
directions are shared across amplitudes unless collision rejection requires a
different draw; manifests and the sweep report disclose these redraws. Each
run uses the same0.60A maximum CF displacement from the original parent, so
legal0.30A RMS inputs (per-site cap0.60A) are not clipped by the old0.40A bound.
All other CF/B/predictor/selection settings remain unchanged. The0.10A arm is
a fresh equal-N control; the previous mixed-amplitude pilot is not its substitute.

`prepare.py --amplitude 0.20 --proposal-cap 0.60` configures a constant-amplitude
run. Omitting these flags preserves the original mixed-amplitude protocol and
0.40A bound. Resume rejects an amplitude/bound configuration mismatch.
`pilot.sbatch` accepts `MATTERGEN_BESTN_AMPLITUDE` and
`MATTERGEN_BESTN_PROPOSAL_CAP`; the sweep sets both explicitly and runs three
arms sequentially inside its single Slurm GPU allocation. Results are under
`results/mattergen/amplitude-sweep-<job>/rms-<amplitude>`; `sweep_report.py`
produces the parent-level comparison, eligibility and geometry counts, and
cross-amplitude common-parent metrics. Failed attempts retain their budget.
This remains a four-parent feasibility study with no SUN/DFT/MD.
