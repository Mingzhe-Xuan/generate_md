# Direct-B versus CF200-B structure and property paths

This is a B-only observational replay, not a new A optimization. Use the four
fixed raw parent IDs0000/0001/0008/0087; compare original MatterGen inputs with
the saved FULL fixed-species CF200 inputs under alignn-full-361/cf200/neutral.
Their full-run terminal scores are controls; do not substitute the older
four-case CF pilot, whose optimization cohort differs.

`replay.py` prepares eight immutable tasks for the existing MatterSim GPU worker,
enabling its optional trajectory observer. The unchanged5M/FixSymmetry/
ExpCellFilter/BFGS profile uses fmax0.01/maxstep0.05/1600-step cap. Save input
before FixSymmetry, optimizer step0 after symmetry refinement, and every step.
Every frame gets its source pose SHA, E, force and actual optimizer-step index.
Source structures and original results remain readonly. No A/SUN/DFT/MD.

`score.py` reuses the hash-pinned author CF models/inference on GPU, scoring
bandgap and formation energy for every saved pose. Check original input/final
predictions against archived frozen scores at unchanged1e-3 tolerance and
terminal energies against archived B. Scoring occurs after B, so it cannot
modify the relaxation. This path question uses CF; prior independent ALIGNN
endpoint results remain available but no per-frame ALIGNN claim is made.

`report.py` reports per-case errors, actual/normalized steps, symmetry-initialization
jumps and all-frame nonmonotonicity. Fixed-denominator aggregate MAE curves carry
already-converged structures forward at their last score, avoiding survivor bias.
Four-case replay curves do not establish full128 trajectory behavior.
`pilot.sbatch` runs all inference/processing through Slurm using existing envs.

Full128 same-ID endpoint similarity is a separate archived-data analysis using
StructureMatcher with strict(ltol.02,stol.05,angle1) and looser(.1,.2,5) settings,
species matching, scaleFalse/primitiveFalse/no supercells. These tolerance-based
matches are not proof of mathematically identical stationary points or basins.
