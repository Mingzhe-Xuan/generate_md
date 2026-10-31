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

## Explicit experimental policy

The native PC corrector chooses step sizes using realized Gaussian noise norms
and batch-averaged score norms. Treating its transition as an ordinary Gaussian
is incorrect. This implementation instead defines a per-crystal corrector with
the expected noise squared norm and retains native ancestral predictor
coefficients and native1000-level D3PM posteriors. Gaussian standard deviations
have a0.001 floor (fractional positions; Angstrom lattice entries), including
the otherwise deterministic final step. Six independent symmetric-lattice
entries are the cell action. Position wrapping and cell absolute-eigenvalue
projection (minimum0.01Angstrom) are parameter-independent environment maps.
The unwrapped, preprojection actions are stored and scored; marginal densities
of the projected geometries are not substituted for those action densities.

LoRA rank4 adapts linear maps with zero-initialized output factors; pretrained
weights stay frozen. GRPO uses per-step joint action probabilities, group-wise
standardized terminal reward, PPO clipping0.2 and dimension-normalized analytic
KL to the frozen base policy. Reward is `exp(-0.5*((ALIGNN_gap-2.5)/0.5)^2)`;
unsupported/invalid structures receive-1 and remain in the group/denominator.
No true labels are assigned to generated structures by copying reference gaps.

The smoke must complete all1000noise levels, replay saved actions under the
unchanged policy (maximum absolute per-crystal logprob error0.01), obtain real
within-group reward variation, backpropagate a finite nonzero gradient, change
LoRA weights and preserve frozen parameter hashes. Passing this smoke does not
establish a property improvement.

## Exact base recovery audit

The complete official base checkpoint was downloaded locally and verified against
SHA81668ee12afc1ee1b037f362420730de3460bfd2d36e547585fdcb911a3dfdef.
An ignored manifest records all281state-tensor hashes, dtypes and shapes plus
the canonical config hash. `recover_base.py` can select those tensors from the
cached conditional checkpoint only if every entry is bitwise identical. It
excludes all adapter-only weights, writes a separately hashed base state and
strictly loads the official base architecture. Any mismatch fails the stage;
the conditional/unconditional branch is never silently relabeled as base.

The first audit (Slurm519) found only105/281matching tensors; its failed report
is retained. The remaining176official tensors are supplied as a lossless NumPy
delta, checked against the same original-base manifest after assembly. Partial
transfers use1MiBchunks with individual and whole-payload hashes. Recovery
success still requires all281exact matches; partial matches are not accepted.

## Planned online training budget

`train.py` is gated on the completed full-length GPU smoke, matching policy-code
hashes and the unchanged audited cohort. It proposes200rollout iterations,
fourgroups of eightcandidates (6400new candidates), twoinnerupdates per rollout,
rank4LoRA AdamW2e-6 with cosine decay to2e-7, and eightuniformly sampled noise
levels per rollout for likelihood updates. Microbatches contain at most four
crystals/64atoms; advantages are computed over complete groups before slicing.
The reference is the frozen official base. No raw geometry gradient is taken
through ALIGNN. Invalid rewards remain in groups and reported denominators.

Validation uses64independently generated structures at fixed validation seeds
every25iterations. Checkpoint selection uses target-window success rate,
validity, then valid-only MAE, in that order. Final test rollouts use separate
seeds and never select checkpoints. State/optimizer/context-RNG checkpoints and
saved old-policy actions support a verified resume without silently replacing
partially completed work. These are implementation/proposed budgets; actual
training is not claimed until the GPU gate and Slurm artifacts prove it.

## Independent checkpoint comparison

`compare.py`/`compare.sbatch` require completed GRPO and evaluate256new structures
per arm under the same heldout atom-count contexts and test seeds: base,
validation-selected GRPO, final GRPO, conditionalcheckpoint under the matched
explicit kernel, and conditionalcheckpoint under official native PC/CFG2 with
its terminalmean. The extra native arm prevents a changed sampler from being
misreported as a trained-model gain. Checkpoints are never selected on these
final samples. Real heldout geometries/compositions are not generation inputs.

All five arms then receive the same pinned canonical MatterSim5M B profile,
with raw atomic force convergence reported separately from constrained-filter
convergence. DFT and SUN remain paused. Invalid and unsupported candidates remain
in full requested denominators; valid-only MAE must be read alongside coverage.
Same-seed comparisons resample sampling batches (native corrector is coupled
within a batch). Degenerate empirical bootstrap intervals are marked uninformative.
The generated report records raw and relaxed gap success/MAE, force passes and
joint gap/force passes. CUDA/OOM infrastructure errors abort the stage instead
of being disguised as poor material quality.
