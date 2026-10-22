# CPU206 relaxation and DFT validation

Preserve all 128 generated structures and original identities. This module adds
postprocessing to an earlier Section 2.1 Crystalformer CPU run; it cannot turn
that generation protocol into the distinct Section 2.3 ALIGNN benchmark.

Inputs: immutable CPU206 CSV and 128 POSCARs; sources.json pins the CSV and
external model/source identities. Outputs live only in results/smoacs/postprocess-206.
Submit inventory.sbatch from the parent repository root. It checks candidate
identity/denominator and inventories installed potential datasets that pymatgen
recognizes as PBE_54. It never publishes potential bytes or silently substitutes
old/incompatible potentials. Missing datasets remain explicit blockers to DFT.

MatterSim settings follow the author postprocessor: 5M model, BFGS,
ExpCellFilter, symmetry constraint, 400 steps, default fmax0.01 eV/angstrom.
The README-pinned research fork commit is no longer available from origin;
sources.json records both that unavailable revision and the acquired revision.
This discrepancy must remain in any reproducibility claim.

Independent .venv-smoacs-postprocess and .venv-smoacs-dft environments preserve
the generation environments. requirements-mattersim.txt reflects the current
fork's package metadata with Blackwell Torch and ASE3.27; requirements-dft.txt
starts with the author paper environment's scientific package versions. A
dependency-resolution failure must be investigated, not bypassed by ignoring it.

The DFT target is VASP5.4.4 with atomate2 MPGGADoubleRelaxStaticMaker,
PBE/PBE+U MP settings and a final NEDOS2001 static for bandgap evaluation.
It is not the low-cutoff two-Si capability test. Full stability screening needs
consistent MatterSim-relaxed competing phases and MaterialsProject2020Compatibility,
not just a negative candidate formation energy. Reference unavailability must
remain unknown, never be treated as passing stability or novelty.

Installation is performed by install.sbatch with SMOACS_STAGE=mattersim/dft.
Each job validates its own direct PyPI route, installs into an already-created
isolated venv and runs pip check/freeze. Native extension builds stay in Slurm.
prepare_model.sbatch extracts the exact 5M file from the pinned official source
archive and verifies its known SHA; a partial transfer is never loadable input.

After inventory and environment/model preparation, relax.sbatch uses
SMOACS_INVENTORY to select the audited128 manifest. SMOACS_RELAX_IDS selects an
explicit pilot; SMOACS_REQUIRE_CONVERGED=1 makes that pilot fail unless every
selected candidate converges. Omit these variables for the full batch. Existing
completed records are reused only with matching input hashes. Per-candidate
failures/unconverged states remain records in the128 denominator. The explicit
ASE BFGS/ExpCellFilter/FixSymmetry sequence follows the author's Relaxer
implementation, while recording the real optimizer convergence return value.
The exact verified checkpoint uses the existing scoped legacy-load adapter;
no global Torch loading defaults are modified.

dft_prepare.sbatch builds a private PBE54 cache only from matching installed
datasets, then checks actual POTCAR construction and MP flow creation in the
paper-version DFT venv. Cache potential contents remain server-only and ignored.

DFT execution: set SMOACS_DFT_ID to one eligible original candidate ID and
SMOACS_INVENTORY to the audited inventory JSON, then submit dft_run.sbatch.
It requires a converged, hash-matched MatterSim result and recognized installed
PBE54 datasets. paper_dft.py centralizes the exact author maker for preflight
and execution. The workflow must contain exactly two relaxations and a static;
each parsed stage must converge with finite energy. The final static must have
NEDOS2001. Record bandgap from its eigenvalues, not from the ML predictor.
All settings/input hashes and stage-level convergence are retained. E_hull is
explicitly unknown until consistent reference phases are available. Per-job
attempt directories preserve failed runs. Do not confuse successful workflow
execution with meeting bandgap or S.U.N. criteria.
