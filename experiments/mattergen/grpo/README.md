# Targeted MatterGen GRPO

User objective: start from pretrained MatterGen, optimize generation towards
2.5 +/-0.1eV and compare with the existing official bandgap-conditioned
checkpoint using identical frozen ALIGNN evaluation.

User clarified pretrained-checkpoint GRPO directly, without supervised warmup.
GRPO uses online generated groups; it does not turn a static set of correctly
labeled crystals into policy rollouts. Real target-range training structures
define the training atom-count/context distribution. Generation starts from
random noise; species, positions and lattice remain generated. Their DFT labels
are not reused as the truth of newly generated crystals. This does not establish
DFT accuracy of the generated candidates.

Module interfaces/planned stages:

- `prepare.py`: cached MEGNet data + exact ALIGNN ID splits -> target-range,
  <=20atom cohorts with grouped train/validation/test roles, rejection counts,
  atom-count distributions and input hashes. Remote processing is Slurm only.
- `policy.py`: explicit stochastic generation transition densities and tests;
  exact coordinate periodicity, symmetric-lattice support and categorical
  species handling. No reward-weighted denoising loss is called GRPO.
- `rollout.py` / `train.py`: sampled groups, frozen ALIGNN rewards, clipped
  group-relative policy updates with reference regularization, resumable state.
- `evaluate.py` / `report.py`: independent rollout seeds, same atom-count and
  sampling budgets for base, GRPO and official conditioned checkpoints; raw and
  common MatterSim-relaxed property errors, success/failure denominators,
  structure validity and diversity. DFT/SUN remain paused.

The source checkpoint/config and sampler must be audited before training. The
existing official conditional checkpoint is a distinct baseline, not a stand-in
for the unconditional base weights. Record changes to sampling kernels and
compare checkpoints under matched kernels; report an official native-sampler
baseline separately if the GRPO sampler differs.

Outputs live in ignored `results/mattergen/grpo-gap25-20261031/`. Never overwrite
earlier experiments. The common ALIGNN evaluator is also the proposed reward
model, so its test results measure surrogate adherence; independent validation
and reward exploitation checks are reported separately.

```bash
export GRPO_ROOT=results/mattergen/grpo-gap25-20261031
sbatch --export=ALL experiments/mattergen/grpo/prepare.sbatch
```

Local contracts: `python -m unittest discover -s experiments/mattergen/grpo -p 'test_*.py'`.

The training budget, actual selected cohort and usable GPU throughput are
reported from artifacts after their checks, not inferred from planned settings.
