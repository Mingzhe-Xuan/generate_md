# CPU206 convergence diagnostics

`analyze.py` consumes the immutable original result.npz, run_config.json and CSV.
It validates128 IDs and final predictions, reconstructs the exact nonperovskite
objective `max(abs(gap-2.5)-0.04,0)+Ef` (coefficient1, noEfclip), and plots the
200x128 property trajectories, late-window changes and annealing schedules.
History row i is the forward pass BEFORE update i+1, i.e. completed_updates=i
for i=0..199. Final dist/onehot predictions use a different1e-8temperature and
must not be silently appended to the training-temperature curve.

Late-window endpoint changes and peak-to-peak spans use an explicit0.01 diagnostic
tolerance, not a paper-defined convergence criterion. Property-only success at
intermediate steps omits neutrality/distance checks, since geometric histories
were not saved. Final original CSV validity is retained separately.

`stationarity.py` restores the archived scaled lengths/angles, periodic fractional
coordinates, raw elemental and oxidation variables/masks. It loads the identical
archived author source and checkpoints in the originalCPU environment. Require
restored final continuous predictions to match the archive within1e-3 before
probing. The endpoint temperature1e-8 and last training temperature0.0001495 are
tested separately; each probe holds temperature and elemental/oxidation weights
fixed. Differentiate ONLY geometry, normalize each block's negative gradient,
scale by the original length/angle/coordinate learning rates, and try fixed
multipliers .001,.01,.1,1 from the same unmodified endpoint. Normalization has
positive block scales, preserving a descent direction when gradients are finite.
The original preprocessing enforces the same cell bounds and periodicity.

Output gradient norms are with respect to scaled lengths, scaled angles and
fractional coordinates, not physical forces. A successful finite decrease shows
that the endpoint still admits improvement under that fixed objective. It does
not estimate an exact number of additional Adam steps, guarantee a better
discrete-material success rate, or claim to resume missing Adam momentum states.
No model is retrained, no existing result is overwritten, no DFT is run.

Run local history analysis using downloaded NPZ/CSV/config and a fresh ignored
results directory. Submit all predictor/gradient inference with
`sbatch experiments/smoacs/convergence/stationarity.sbatch`; fixed4CPU/24G
allocation uses .venv-smoacs-paper, never the login node.

Tests: `python -m unittest discover -s experiments/smoacs/convergence`.
Outputs: history.csv, candidates.csv, summary.json, convergence.png/svg and
tail_and_schedule.png/svg; stationarity-<job>/summary.json holds control checks,
per-candidate gradients, finite trial changes and source/model/input hashes.
