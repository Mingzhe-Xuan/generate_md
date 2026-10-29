# Force-available score orthogonalization: paired generation, B and SUN

The user confirmed s_new=s-proj_F(s), then explicitly removed the cosine0.7 gate:
apply whenever current discrete species are all known/supported and MatterSim
successfully supplies a nonstationary force. s is the Cartesian deterministic
score-driven positional displacement s_frac L, not the density covector. Remove
the global per-structure centered-force component and convert back with L^{-1}.
Negative and small cosines are also projected. Preserve original score COM;
zero force/score, masked/unsupported atoms, invalid poses or failed predictions
retain the original score with a recorded reason. No final-type backfill or
arbitrary density/distance/energy/property eligibility filter is applied.

`projection.py` defines the operation. `run_experiment.py` runs16 paired fresh
unconditional trajectories from official priors using seed20261050, the same
frozen dft_band_gap checkpoint/CFG0, two batches8 and1000official time levels.
Baseline and orthogonal branches reset the same seed and verify identical prior
arrays and CPU/CUDA RNG checkpoints. Prior random-probe16finals are compared as
a replay diagnostic, not substituted for the freshly sampled primary control.

Modify positional scores in BOTH corrector and predictor. Per-call cell/type
outputs remain unchanged; future coupled predictions may differ. No norm
restoration is applied. The original corrector uses a batch-mean score norm to
adapt step/noise amplitude: projection can increase noise amplitude and affect
other structures in that batch. This experiment tests the requested score
operation under that native rule, not an isolated direction-only change. Record
before/after vectors, physical forces, reasons, actual F32 injected score,
per-state norms/cosines and nominal unclipped corrector-step multipliers.

All force/model compute uses Slurm and existing .venv-mattergen / persistent
.venv-smoacs-postprocess CUDA workers. Parent RNG cannot change during projection
RPC/transform. Both raw outputs then receive identical existing canonicalB:
MatterSim5M BFGS/ExpCellFilter/FixSymmetry, fmax0.01, maxstep0.05, cap1600. Preserve
failures and all16denominators. No CF/ALIGNN optimization or DFT is run.

`offline_references.py` reuses immutable verified MP2026-04-13 range-cache bytes,
reconstructs thermo IDs and scans ALL summary rows through the established
published-protocol selector for the chemical subsystems of both arms. It does
not prune an old scope that might omit new competitors. New immutable phase
inputs are converted with the unchanged reference converter. Validated existing
316/322/261/343 caches are reused with fixed priority; missing phases use the
unchanged canonical serial400-step reference relaxation. Missing or failed
competitors remain explicit unknown, never certified stable. No downloads or
environment changes are planned; incomplete local cache halts selection.

Both arms use the SAME reference pool and unchanged references/assess.py:
MatterSim energies + MP2020 corrections, hull<0.1eV/atom, within-arm U and N
against relaxed MP phases (raw-MP novelty secondary). `report.py` reports
S/U/N/SUN individually /16 and unknown coverage; pairs raw and relaxed structures
with strict and author StructureMatchers, composition/order changes, fixed-site
translation-removed RMS, species-pair distances, volume and energy. A mismatch
alone is not a new energy basin. This16pair study is exploratory.

Four sequential stages (submit with matching ORTHOGONAL_ROOT and afterok deps):

```bash
python -m unittest discover -s experiments/mattergen/orthogonal_score -p 'test_*.py' -v
export ORTHOGONAL_ROOT=results/mattergen/orthogonal-score-20261029-seed50
sbatch experiments/mattergen/orthogonal_score/generate.sbatch
sbatch --dependency=afterok:GEN_ID experiments/mattergen/orthogonal_score/select.sbatch
sbatch --dependency=afterok:SELECT_ID experiments/mattergen/orthogonal_score/references.sbatch
sbatch --dependency=afterok:REF_ID experiments/mattergen/orthogonal_score/assess.sbatch
```

The final job archives own evidence and SHA/512KiB pieces; the external novelty
symlink is excluded while its source hash/provenance remains in references/summary.json.
