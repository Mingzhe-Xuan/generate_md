# Full128 GPU comparison

Three fresh runs use the same128 original Struct-AtomicMask templates, seed42,
frozen predictors, five optimized variables, temperature0.01 and base Adam rates.

| Arm | Property updates | MatterSim B calls per eligible template |
|---|---|---|
| single | One600-step cosine | 1 |
| alternating | Three200-step cosines, persistent Adam moments | 3 |
| restart200 | 600 updates with200-step cosine restarts | 1 |

`run.py` selects the shared CUDA Adam controller. Its16-structure gradient
microbatches accumulate the original loss divided by128, then perform one global
Adam update. Startup compares full and microbatch forward values, and gradients
under8/16 partitions. GPU placement is audited during actual optimization.
`submit.py` persists job IDs and submits three optimization/evaluation pairs and
one dependent shared-reference job. All computation runs through Slurm.

`report.py` validates cohort IDs, initial structure hashes, protocol invariants,
CUDA audits and evaluated B hashes. It constructs a union of successful endpoint
chemical systems for one MP reference selection. `references.sbatch` reuses
hash-verified references, validates parallel MatterSim against serial controls,
then assesses all three methods against that pool. DFT remains paused.

Inputs: existing original archive,128-template baseline, frozen CF/ALIGNN weights,
MatterSim5M checkpoint and task virtual environments. Outputs under an ignored
`results/smoacs/<experiment>`: per-arm optimization/evaluation files, shared
reference inputs and energies, `comparison.json` and `comparison.md`.

```bash
python3 experiments/smoacs/full_comparison/submit.py \
 --output results/smoacs/full128-three-methods
```

Reports show gap[2.3,2.7] eV, formation energy< -0.5 eV/atom, structural validity,
joint validity for both frozen predictors, and separate S, U, N, SUN. S means
MatterSim/MP2020 hull<0.1 eV/atom; U is within each method; N compares to the same
relaxed MP phases. SUN plus property joint is also reported. All128 templates
stay in the main denominator, with unknown counts and possible-rate bounds.
A secondary common completed cohort supports paired comparisons. Fixed600-update
budgets and force-converged B endpoints do not demonstrate joint A/B convergence.
