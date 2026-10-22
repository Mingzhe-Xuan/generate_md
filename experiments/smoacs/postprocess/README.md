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
