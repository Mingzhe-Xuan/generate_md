# MatterGen fixed-species, one Adam update then force-converged relaxation

This four-template pilot starts from the same immutable original MatterGen
structures selected by nearest4/8/12/20 atom counts and ID, without selecting
on properties or relaxation outcome. Every atom keeps its original element.
Only lattice lengths, angles and fractional coordinates are optimized on GPU.

`pilot.sbatch` runs two arms from the same raw poses: fixed-element A200 then
one B, and200 repetitions of one A then B. Both use base Adam rates
0.002/0.007/0.005, one global200-update cosine decay, temperature0.01, the
same CF gap/formation-energy loss, and frozen CF/ALIGNN evaluators. Adam
moments persist through each B; only geometry targets are replaced with the
force-converged B structure. Each subsequent A input is saved and checked
against the preceding B, including periodic fractional coordinates and the
lattice Gram matrix. No initial B is inserted before the first A.

B retains the existing MatterSim5M/FixSymmetry/ExpCellFilter/BFGS profile,
maxstep0.05, fmax0.01 and1600-step safety cap. A failed/unconverged B stops
that candidate; later endpoints remain explicit unknowns, not prior successes.
The budget is200 CF updates per surviving case, not a joint-convergence claim.
Final states and checkpoints are saved at every A/B boundary. No SUN, DFT,
new MatterGen generation, or training is performed.

`evaluate.sh` exports and cross-predicts the final endpoints without rerunning
completed A/B stages. `report.py` compares exact same IDs against original
MatterGen, direct B, prior variable-element A200/B and A600/B, prior fixed
A600/B, and the matched new fixed A200/B control. Old full-cohort arms used
their own65/63 mean losses; the two new four-case arms provide the matched
batch/budget comparison. Report gap and formation-energy passes, structural
validity, joint passes, gap MAE/RMSE, unknowns, paired errors and the200-cycle
CF trajectory. Four cases support feasibility observations only.

Submit from the repository root:

```bash
sbatch experiments/mattergen/stepwise/pilot.sbatch
```

Outputs are under `results/mattergen/stepwise-<job>`. The existing Blackwell
CF/ALIGNN and MatterSim postprocess environments are reused. All numerical
work executes within Slurm; source changes are made locally and synced by Git.
