# SUN for the frozen CF-joint subset

Input: final full128 Best-of-10 selection-records.json and plan.json. Freeze
noise0.30 A + CF20 + MatterSim final selected structures, then retain the12
CF-joint passes. Both legacy and force/Ef selection gates must give the same12.
SUN never changes branch selection; zero branches and alternate candidates are
excluded. No A/B proposal rerun, downloads, new training or DFT is performed.

`policy.py` defines cohort validation and chemical-subsystem inclusion.
`prepare.py` checks original A/B inputs, terminal hashes and forces, copies
existing endpoints, and prunes the full128 selected-322 MP inventory only to
chemical subsystems needed by these12 fixed-species parents. All competitors
and conversion failures for each system must remain. The full raw-MP novelty
stream is reused unchanged. Existing reference caches follow the established
316/322/261 priority, then343, preferring converged records without energy ranking.
No missing or failed reference is silently certified stable.

Submit `BESTN_SUN_SOURCE=/absolute/fullrun sbatch --export=ALL
experiments/mattergen/sun_subset/run.sbatch`. All batch preparation, reference
cache validation, energy correction, hull and structure matching run in Slurm.
Only CPU is required because MatterSim endpoints and references already exist.
Existing environments are reused unchanged.

The unchanged references/assess.py defines S as MP2020-corrected MatterSim5M
hull <0.1 eV/atom, U within these12 in original parent order, and N against
MatterSim-relaxed MP reference phases. Its author StructureMatcher uses
ltol0.2/stol0.3/angle5, primitive/scale/supercell enabled. Unknown coverage stays
unknown. This novelty differs from merely differing from the direct-B endpoint.

`report.py` joins the12 immutable CF/ALIGNN scores with S/U/N and reports both
conditional /12 rates and CF-joint+SUN yield /128. It writes report.json/md and
the batch script archives summary/provenance. Tests:
`python -m pytest experiments/mattergen/sun_subset/test_sun_subset.py`.
