# A600 then B versus interleaved A200/B

Same four original templates,seed42,frozen models, corrected Gaussian derivatives,
CUDA float32,T0.01,three Adam groups and B profile as completed Adam-cycles302.
Primary arm:600 consecutive A updates with one600-step cosine decay, then B.
Auxiliary arm:600 consecutive A updates with200-step cosine restarts, then B.
The latter matches the alternating arm's LR sequence to help distinguish B
interleaving from LR-restart effects. All five optimization variables remain free.
Both arms start fresh; neither starts from the optimized three-cycle outputs.

`run.py` invokes the shared GPU runner, `run.sbatch` allocates RTX5090,
`evaluate.sbatch` reuses immutable export and frozen CUDA CF/ALIGNN evaluators.
`compare.py` pairs actual relaxed outputs with cycle3 from the previous run,
reporting target2.5eV absolute error perID, mean/median/RMSE andpassrate with
missing outcomes retained. A-endpoint errors are also reported separately.
No SUN or DFT rerun is part of this band-gap comparison.

Use `submit.py --output results/smoacs/A600-GPU-RUN` to launch both arms and
their dependent evaluations/final comparison; a sibling registry preserves IDs.
All inference/optimization/relaxation/processing runs through Slurm. All generated
artifacts are ignored under task results. Tests: `python -m unittest discover -s
experiments/smoacs/a600`.
