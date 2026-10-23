# Adam200 / MatterSim: one versus three cycles

Same four preselected original templates andseed42 as the block pilot. Each A
phase performs exactly200 Adam updates on lengths/angles, coordinates and
elemental/oxidation weights. Three parameter groups use original base rates
0.002/0.007/0.005. A200 cosine schedule restarts from its base rates each cycle;
Adam moments persist through B feedback. Temperature stays0.01, fixed for both
comparands. This is a fixed-budget experiment, not a joint-convergence claim.

`run.py` reuses hash-verified original initialization and the controlled float64
Gaussian derivative correction from block_alternating. The two compared endpoints
are cycle1B and cycle3B of the SAME trajectory, sharing the entire first cycle.
This avoids different random starts or a separately tuned one-cycle baseline.
B uses the same pinned5M checkpoint, FixSymmetry/ExpCellFilter/BFGS maxstep0.05,
fmax0.01 and1600-step safety cap. Failed B candidates are not advanced from an
unconverged geometry. All four IDs remain in every cycle's denominator; missing
later cycles retain explicit noneligibility status rather than prior successes.

Crystalformer optimization runs on the RTX5090 using the existing Blackwell
Torch2.7.1+cu128 environment and native PyTorch CUDA encoder. After verified CPU
initialization, move the frozen models, all five targets, inputs and captured
scaler/bound tensor aliases to CUDA float32. Require CPU/GPU initial predictions
within1e-3 and observe actual CUDA encoder forwards/backwards plus CUDA input
gradients. `gpu-audit.json` records this evidence; CPU fallback is an error.
Both cycle evaluations also run frozen Crystalformer and ALIGNN on CUDA.

`adam_policy.py` defines the schedule, inactive-state protection and reporting
math. `run.py` stores each immutable A/B endpoint, trace, hashes, per-step timings
and atomic checkpoints. `export.py` creates compatible evaluation inventories for
cycles1 and3. `evaluate.sbatch` independently evaluates frozen Crystalformer and
ALIGNN. `references.sbatch` selects the union of both endpoints' chemical systems
from the same publicMP snapshot, reuses verified reference phases, and computes
S/U/N/SUN against one shared reference pool. `report.py` compares full-four rates
and identical completed pairs, including unknown bounds and property+SUN overlap.
The existing MP2020 compatibility environment does not run DFT.

Launch optimization via `sbatch experiments/smoacs/adam_cycles/run.sbatch`.
Export `SMOACS_ADAM_RUN` to that run directory for dependent evaluation/reference
jobs. All inference, relaxation and batch processing runs through Slurm. Generated
artifacts stay under ignored `results/smoacs/adam-cycles-<jobid>`.
Alternatively run the lightweight login-node manager `python3
experiments/smoacs/adam_cycles/submit.py --output results/smoacs/adam-GPU-RUN`.
It records job IDs in a sibling `*-jobs.json` and submits an afterok chain for
optimization, properties, and shared-pool SUN. Repeating it preserves recorded jobs.

The shared runner also exposes a600-update/one-B mode used by `../a600/`;
the default command and200-update/three-cycle numerical schedule remain unchanged.

Tests: `python -m unittest discover -s experiments/smoacs/adam_cycles`.
