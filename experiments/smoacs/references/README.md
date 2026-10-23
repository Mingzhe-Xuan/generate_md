# Public MP reference acquisition

Download source: official materialsproject-build S3 Open Data, release2026-04-13.
This date falls within the paper's stated access interval but does not prove an
identical historical API snapshot. No API key is required. All derived selection,
decompression and structure processing runs in Slurm; source bytes are cached in
ignored results/smoacs/mp-public-reference. No API credentials are printed.

prepare.py reads selected Parquet columns through cached byte-range requests,
pins each object's ETag/size, and records hashes for every cached range. It collects
nested entry material IDs from the three paper thermo collections, then summary
structures/hull values. It explicitly audits differences from parent material IDs.
Low-level ParquetFile.read supports map-leaf projection; dataset read_table does
not, so the former retrieves entry IDs without downloading whole energy entries.
Summary selection follows the published Zenodo20149017 function: <=0.3eV/atom,
excluding missing hull values. A later GitHub change adding missing-hull elemental
<=10site structures is deliberately excluded from this published-protocol selection.
All candidate chemical subsystems and all selected phases are retained without
random truncation. Unfiltered structures are kept for a secondary raw-MP novelty
diagnostic; primary novelty follows the archived MP-only Stage4 and compares
against MatterSim-relaxed selected reference phases of the same reduced formula.

Install requirements.txt in a new Python3.10 venv .venv-smoacs-reference.
prepare.sbatch tests the proxy path inside its allocation; login-node loopback
access is never assumed. If unavailable, the job fails before network reads.
Reference structures subsequently require the same MatterSim5M relaxation and
MP2020Compatibility protocol as proposals before a hull can be evaluated.
Raw MP hull values select references; they are not the final MatterSim hull.

Optional parallel.sbatch keeps independent per-structure BFGS optimizers and
the same5M calculator in separate processes within one GPU allocation. Switching
requires a16-reference serial-vs4-process benchmark with converged force thresholds,
energy agreement<=1meV/atom, relative volume agreement<=0.1%, and at least10%
wall-time improvement. Cached completed primary records are reused by input hash.
Work partitions are disjoint/exhaustive, and cells>128atoms occupy only one worker
to limit simultaneous large GPU allocations. Each worker has a separate manifest,
log and environment record. A failed benchmark leaves the serial method in use.

convert.sbatch writes hash-identified POSCARs and a separate reference inventory;
only subsystems needed by converged hash-matched proposals undergo expensive
reference relaxation. Raw selections remain intact, and excluded reference IDs
and the115 eligible proposal IDs are recorded. This does not remove any competing
phase for a proposal that can pass screening; all128 proposals remain in results.
relax.sbatch uses the validated proposal MatterSim relaxer with identical physical
settings and reference-specific metadata. Convert/relax failures remain explicit.
assess.sbatch uses the paper-version DFT venv for MP2020Compatibility, with run
parameters from MPGGARelaxSetGenerator as in the author script. A composition-
balanced linear program queries the same convex envelope at each candidate,
avoiding expensive enumeration of every facet for high-component systems. Binary
and ternary tests match pymatgen PhaseDiagram, including below-hull clamping;
a nine-element fixture verifies mass balance and the analytic hull value.
Per-candidate hull processes have a120second limit and are terminated only by
their own handle. Energy corrections and the stable threshold are unchanged.
Missing reference competitors cannot certify stability: a low incomplete hull
remains unknown. Stability requires E_hull<0.1eV/atom. Uniqueness and novelty use
the author StructureMatcher settings. OpenMX BG&SUN remains unknown unless the
electronic-structure calculation and its numerical checks have completed.
Actual optimizer convergence is an additional validation requirement: the
published relax_with_mattersim wrapper ignores the returned convergence flag.
Finite400-step but unconverged geometries are recorded separately, never silently
certified as stable. This distinction must remain explicit in reproduction claims.
This covers the primary MatterSim/MP benchmark screening; the paper's additional
representative examples use MP+Alexandria and DFT-derived hulls, which require
additional reference acquisition and consistent alternative DFT calculations.

Source: https://docs.materialsproject.org/downloading-data/aws-opendata
