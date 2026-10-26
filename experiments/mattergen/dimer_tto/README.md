# Property-gradient-seeded saddle search: real-physics feasibility

This is a separate fixed-species candidate method. It does not modify the
running annealed-bias study. Its first native run uses exactly the first four
original parents:0000/0010/0015/0025. The immutable441 proof provides gradients
computed with the frozen CF model on RTX5090 and the identical saved seed pose.
No CF model inference occurs on CPU. Only the atomic gradient is used here;
cell gradients are omitted during fixed-cell activation. Cartesian gradients
are the cached atomic q gradients divided by sqrt(N), followed by centroid
projection and direction normalization. This is initial
direction guidance, not repeated CF property descent along the activation path.

`dimer_policy.py` freezes controls, parameters and future-quench inventory.
`dimer_engine.py` implements a geometric-centroid gauge, counted guarded
calculator calls, ASE dimer activation and projected finite-force Hessians.
`dimer_native.py` verifies source/model provenance and runs CPU MatterSim physics
through Slurm. The existing postprocess environment and5M checkpoint are reused.

CF descent, reverse and seeded random directions all start from the same seed,
with0.10A RMS initial displacement. ASE dimer has120 translation steps, up to5
rotations/step, maxtranslation0.05A and separation0.002*sqrt(N)A. Raw saddle force
must be<=0.02eV/A and curvature<-0.01eV/A^2. Translation is removed using a
centroid constraint without changing atomic masses. Proposals preserve species
and cell, offsets<=2A, minimum distance0.7A, and intermediate energy<=seed+1eV/atom.
Each activation has at most2000 actual calculator calls including certification.

All four CPU versus cached-GPU energy/force/stress controls must pass1e-4 before
any activation. Directional curvatures at0.004/0.002A RMS must agree. A candidate
then needs two projected Hessians with relative asymmetry<=0.02, declared
two-scale agreement, and exactly one resolved negative eigenvalue below-0.01.
This is a numerical resolved-mode check, not a phonon/global stability proof.
Both0.05A RMS exits are attempted; each feasible exit is saved even if the other
fails its geometry guard. Failed exits still consume their future quench slots.
Actual NFE, full center traces and initial/final mode alignment
are recorded. Dimer rotation can erase the initial CF information; this requires
measurement rather than assuming gradient importance.

Use `native.sbatch` with original directed16 `DIMER_SOURCE`, completed441
`DIMER_CACHE` and a fresh `DIMER_ROOT`. This produces12 activation attempts and
24 future quench slots. It does not yet run canonical GPU MatterSim B or frozen
CF/ALIGNN scoring. Those remain necessary before any crystal property claim;
small native feasibility cannot establish population efficacy or goal completion.

The ASE interface and saddle search are described in the
[official documentation](https://docs.ase-lib.org/ase/dimer.html) and
[Henkelman/Jonsson1999 paper](https://henkelmanlab.org/pubs/henkelman99_7010.pdf).
Using a CF gradient to seed these searches is our proposed hybrid; neither
reference establishes retained CF property improvement.
