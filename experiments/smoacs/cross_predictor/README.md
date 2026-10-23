# Crystalformer to ALIGNN cross prediction

Frozen official MEGNet ALIGNN band-gap/formation-energy checkpoints independently
predict the same128 immutable CPU206 generated POSCARs. A second stage evaluates
the115 converged MatterSim geometries. No parameters, composition or geometry
are optimized; no DFT is run. Both architectures share a training-data lineage,
so model agreement is not independent ground-truth validation.

`predict.py` verifies both official ZIP SHA256 values and trusts legacy payload
loading only for the checked ZIP members. Loader temporary files are isolated
inside the new result directory. Fresh graphs use cgcnn node features,
k-nearest neighbors,8angstrom cutoff,12neighbors, canonical periodic edges and
line-graph cosines. Checkpoint graph settings must agree when specified. Band
gap is clamped at0 as in the author predictor; raw outputs are also saved.

Before completing inference, two real graphs must pass a strict comparison
between ALIGNN4inverse explicit-feature forwarding and the native ALIGNN graph
forward with identical state/config; maximum absolute error<=1e-5. All graph
features and outputs must be finite and onCUDA. Model weights are frozen.
Original/relaxed hashes and the prior same-Crystalformer predictor control are
checked. Same geometry validity and targets apply to both predictors: gap in
[2.3,2.7]eV, formation energy<-0.5eV/atom; joint success also requires archived
composition neutrality and no interatomic separation<0.5angstrom.

Interfaces:

- `metrics.py`: numerical differences, correlations and classification agreement
  with explicit unknowns. MAE/RMSE are BETWEEN PREDICTORS, not against truth.
- `predict.sbatch`: Slurm GPU inference in existing .venv-smoacs-blackwell.
  Outputs predictions.json and provenance/native-control metadata under ignored
  results/smoacs/cross-predictor-206/prediction-<job>. No online downloads needed.
- `summarize.py --input <prediction-dir> --output <fresh-dir>`: local result
  analysis, CSV/JSON/Markdown, scatter and difference histograms (PNG/SVG).
  Full128 denominators and the115 converged paired cohort are both preserved.
  Difference histograms use the same complete paired IDs and common bin edges
  before/after; scatter panels retain their explicitly labeled full-stage counts.
  If run remotely, submit this batch analysis through Slurm too.

Primary interpretation: among Crystalformer-positive generated structures, how
many remain positive in ALIGNN? Report both-pass, CF-only, ALIGNN-only, neither,
unknown and confirmed retention. Also report numerical bias,MAE,RMSE,Pearson,
Spearman and absolute-difference quantiles. Constant series yield undefined
correlation, not fabricated0/1. Inference failures remain unknown rather than
being removed from denominators or falsely classified as negative predictions.

Tests: `python -m unittest discover -s experiments/smoacs/cross_predictor`.
