# Frozen ALIGNN reliability classifier

Predict `P(abs(current ALIGNN gap - reference DFT gap) <= tau)` for
tau=0.1/0.2/0.5 eV. The property predictor remains frozen. Features are the
256-dimensional crystal vector immediately before the scalar `fc` layer,
not the scalar gap. Gap outputs use the same nonnegative clamp as prior work.

`prepare.py` selects original predictor validation5000/test4239 structures from
the cached Jarvis MEGNet data. Original predictor training60000 IDs are excluded.
Validation structures are deterministically grouped by reduced composition into
meta-train/meta-validation/calibration at approximately70/15/15. Original test
IDs never fit normalization, model selection, head weights or calibration.
The original validation may have influenced base-model selection; this is not
an untouched external dataset for the whole predictor-development process.

`features.py` uses the pinned existing CUDA/FP64 ALIGNN pipeline and captures
the actual same-forward pre-fc vector. It verifies scalar reconstruction, model
identity, unchanged frozen state and prespecified native architecture controls.
The cache contains raw/clipped predictions, DFT labels, feature vectors and IDs.
Published test prediction differences are diagnostic; all DFT targets must match.

`train.py` trains prediction-only logistic, feature logistic and a tiny
256->64->1 MLP (16513 parameters). Hyperparameters and model choice use only
meta-validation AP for the bad class, followed by BCE. Separate calibration fits
a positive-slope Platt map. Each threshold is trained and reported separately.
`head.py` stores only NumPy arrays and supports pickle-free NumPy inference.
`core.py` defines labels, splits, scaling, calibration and selective metrics.
`report.py` independently checks original ID exclusion, head/scaler/probability
serialization and rank-statistic AUC, then exports AUROC, AP_bad, Brier and coverage/error curves,
including unseen-meta-composition and nonmetal/predicted-semiconductor subsets.

This estimates empirical error risk on reference equilibrium crystals. It does
not certify arbitrary noisy geometries, novel structures or an adaptively optimized
RL distribution. Perturbed states require their own true property labels to train
correctness; reusing the parent DFT gap would define a different task. A confidence
head is another learned surrogate and may itself be exploited in optimization.
No new DFT or GRPO training is included.

Generated data, features, heads and reports live in an ignored run directory,
e.g. `results/mattergen/alignn-reliability-20261031`; no weights enter Git.
Reuse `.venv-mattergen` (Torch2.7.1cu128, NumPy1.26.4, SciPy1.15.3,
scikit-learn1.7.2, DGL1.1.3cu121, jarvis-tools2026.6.12), with no package changes.

```bash
export RELIABILITY_ROOT=results/mattergen/alignn-reliability-20261031
prep=$(sbatch --parsable --export=ALL experiments/mattergen/alignn_reliability/prepare.sbatch)
feat=$(sbatch --parsable --dependency=afterok:$prep --export=ALL experiments/mattergen/alignn_reliability/features.sbatch)
sbatch --dependency=afterok:$feat --export=ALL experiments/mattergen/alignn_reliability/heads.sbatch
```

All remote preparation, model inference, head training and reporting use Slurm.
Tests: `python -m unittest discover -s experiments/mattergen/alignn_reliability -p 'test_*.py'`.

`diagnose.py`/`diagnose.sbatch` optionally check posthoc current-versus-published
prediction discrepancies using native architecture/original graph and single
versus batched evaluation. They do not retune heads. Three independently trained
tolerance heads are not a jointly monotonic error CDF; report this limitation.
Interpret confidence within the intended nonzero-gap domain separately from
overall results, which can be dominated by easily identified zero-gap materials.

See [RESULTS.md](RESULTS.md) for the first completed run, held-out metrics,
deployment package location and coordinate/graph sensitivity caveat.
