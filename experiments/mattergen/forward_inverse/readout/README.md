# Frozen generative representations for forward property prediction

This final mechanism test asks whether a fixed-composition generative backbone
contains useful forward-property information, and whether noisy-input training
helps recover clean properties. It does not update the generator or demonstrate
successful property-guided generation.

Protocol frozen before extraction: hash-select256/64/128 structures from the
original MP20 train/val/test CSVs, seed20261056; eligibility ordered2..20 atoms,
Z<=94, finite geometry/labels. Preserve full-cell species and the author's labels
`band_gap` and `formation_energy_per_atom`. The current CSP backbone's Alex-MP20
training exposure is unknown; this is held-out readout-label evaluation, not
proof of unseen-structure generalization or causal pretraining benefit.

At times0.001/0.05/0.15/0.30, extract mean native-FP32 GemNet node embeddings from
both clean coordinates and one shared Gaussian-corrupted coordinate realization.
Noise uses the official positional marginal standard deviation, fixed cells and
species, and a translation-free gauge. No labels enter the backbone. Append
element fractions, log(atom count), log(volume/atom). Each training head sees
four examples for each of the same256 unique property labels. The clean head
sees clean structures at all four times; the noise-trained head sees noisy
structures at those same times, avoiding a time-embedding exposure confound.

Controls: composition+cell ridge; whole-parent shuffled-label noise head; frozen
ALIGNN evaluated at identical clean/noisy geometries. No extra optimization,
MatterSim relaxation, model download or environment change is needed.

Interfaces:

- `prepare.py`: pinned CSV/hash/split validation; `selection.json` and examined
  input-domain records, CPU Slurm only.
- `extract.py`: frozen native embeddings, noise arrays and ALIGNN predictions;
  per-parent raw NPZ and `features.npz`, GPU Slurm only.
- `regression.py`: train-only scaling/SVD ridge, validation-only per-property
  regularization, paired whole-parent bootstrap; no model fitting on test labels.
- `evaluate.py`: frozen predictions, per-time/clean MAE and RMSE, uncertainty,
  fitted coefficients and plots, CPU Slurm only.

Regularization grid1e-6..1e6 (13 values), validation MSE selects each property's
alpha. Gap predictions are clipped at zero for every ridge model; formation
energy is unrestricted. Test labels are consumed only after coefficients and
predictions are frozen. Paired bootstrap differences resample whole test
structures, retaining all four correlated noise levels. Constant per-time
residuals are not presented as calibrated posterior uncertainty.

An initial score-replay gate failed before extraction. Same-input diagnostic488
showed that unobserved native FP32 score repeats themselves exceeded1e-5 at the
first clean parent, while independent hook/wrapper pooled features differed by
about2e-7. That score gate remains a failed diagnostic. Readout validation now
compares both independent feature-capture mechanisms at every clean/noisy state
with the unchanged1e-5 relative feature threshold and identical graph hashes.
Both feature vectors and raw score repeats are saved; no score-precision or
score-Jacobian validation is inferred from feature success. The448 selected
structures, labels and corruption protocol are unchanged.

Outputs stay under ignored `results/mattergen/forward-inverse-readout-20261030`.
Use READOUT_ROOT for the three Slurm wrappers. This module reuses pinned CSP and
ALIGNN adapters in its parent; all source changes originate locally.

```bash
python -m unittest discover -s experiments/mattergen/forward_inverse/readout -p 'test_*.py'
```
