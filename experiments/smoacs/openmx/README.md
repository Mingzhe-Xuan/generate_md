# OpenMX alternative validation

This module validates the existing CPU206 proposals using OpenMX3.9/PBE19.
It is an alternative electronic-structure method, not a VASP/PBE54 reproduction.
Generation remains the original Section2.1 Crystalformer128/200-step experiment.
All input identities, unsuccessful candidates and the128 denominator are retained.

Modules: resources.py audits matching VPS/PAO valence and hashes; formats.py
writes explicit inputs and validates SCF, variable-cell convergence and sampled
band edges; run.py executes two RFC5 variable-cell relaxations and a static via
native Slurm PMI2. submit.sbatch supplies the installed oneAPI runtime, validates
rank affinity, and calls the runner. All computation and audits run in Slurm.

Inputs: paper-version inventory249 and hash-matched converged MatterSim structures.
Outputs: ignored results/smoacs/openmx-206, with unique per-job attempts. Never
edit installed data. File existence does not certify pseudopotential accuracy.
Use installed DFT_DATA19, matching valence counts, explicit s3p3d2f1 basis (or
available lower angular channels), and minimum bulk orbital cutoff radius >=6au.
Special H/S potential variants require matching H/S orbital files.

Pilot: SMOACS_OPENMX_ID=mp-2730.vasp sbatch experiments/smoacs/openmx/submit.sbatch
The initial basis-level0 pilot showed a0.14eV gap shift on basis expansion and
is NOT certified for production. SMOACS_OPENMX_BASIS_LEVEL=1 increases every
radial multiplicity, then checks the next larger basis at the original tolerances.
SMOACS_OPENMX_MESH_INCREMENT=2 similarly refines the base/check mesh together.
Changed physical profiles require fresh relaxations; original attempts are kept.
Default profile: GGA-PBE, collinear spin, 300K,200Ry real-space grid,
SCF1e-8Ha/300 iterations, RFC5 generalized gradient1e-4Ha/Bohr/200 iterations.
The real-space grid cutoff is not equivalent to VASP ENCUT. Odd Gamma-containing
k meshes are derived from MP's503-density mesh and rounded upward to odd values;
OpenMX's even mesh differs from VASP Gamma-centered meshes. Static validation
checks250Ry, a larger basis, and a denser k mesh before accepting convergence.
These are alternative-method convergence checks, not paper-specified parameters.

All SCF steps must explicitly report achieved convergence. A final optimization
gradient at/above the threshold fails. Final geometry is read from the restart
input, not ASE's OpenMX reader that may select original input coordinates.
Sampled gaps use per-spin band occupations relative to the chemical potential;
changing occupied-band counts across k points or near-Fermi states mark metal.
Sparse meshes cannot exclude unsampled band crossings; convergence checks matter.

The calibration profile is collinear plain PBE, initially1muB/atom. Production
physics.sbatch derives compound-specific Ueff=U-J, angular channels and initial
MAGMOM from the paper MP generator; run.py requires a hash-matched manifest.
OpenMX uses Dudarev Type1/dual and applies Ueff to the first radial orbital of
the specified angular channel, with zero U on other radial functions, following
the official example. The actual FeO+U SCF probe must pass and have a nonzero
Hubbard energy before production dispatch. PBE19 PAO projectors differ from VASP
PAW projectors; matching U values does not prove numerical equivalence.

OpenMX total energies cannot be mixed with VASP energies for a convex hull.
The paper's stability protocol instead uses consistently MatterSim-relaxed MP
reference structures; reference snapshot/version/coverage must be audited.

Production dispatch.sbatch requires SMOACS_REFERENCE_SELECTION,
SMOACS_REFERENCE_ASSESSMENT and SMOACS_OPENMX_CALIBRATION directories. Submit it
with SMOACS_OPENMX_PHYSICS manifest and SMOACS_OPENMX_U_PROBE directory, and
with afterok dependencies on the higher-basis calibration and reference assessment.
It refuses uncertified calibration, keeps all128 candidate statuses, and queues
only exactTrue SUN candidates with audited resources. Arrays run at most4 jobs
concurrently. Every candidate still performs its own numerical checks. A final
assessment uses afterany on the array so failed calculations remain visible.
No certified calibration or no complete reference screen means no production DFT.

References:
https://openmx-square.org/openmx_man3.9/node54.html
https://openmx-square.org/vps_pao2019/
