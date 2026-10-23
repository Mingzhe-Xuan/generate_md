# Block alternating feasibility pilot

This is a separate experiment: solve A to stationarity, relax B to force
convergence, and repeat. Use the same four atom-count-selected original templates
as the previous pilot, retaining all five A variable groups and pinned models.
The original single-update alternating implementation is unchanged by default.
`alternating.run.main(driver=...)` supplies its verified initialization and
forward callbacks; the driver owns subsequent optimization and result files.

`solver.py` implements bounded L-BFGS-B with explicit projected-gradient checks.
A uses a fixed temperature0.01 throughout, so every A phase solves the same
objective `max(abs(gap-2.5)-0.04,0)+Ef`. This is a new feasibility method, not
the original paper's Adam/annealing protocol. Element/oxidation logits and
fractional coordinates remain variable. Cell bounds initially match the
original2..10angstrom/60..120degree box, expanding only to include actual B
geometry before the next A phase. Report both raw and constrained gradients;
a bounded stationary point need not have a zero raw outward gradient.

Each A solve allows1200 accepted iterations/3000 evaluations. A line-search or
loss-tolerance termination alone is never called convergence; the independently
reevaluated maximum projected gradient must be<=1e-3 for that candidate.
Unconverged candidates retain their denominator and do not enter B prematurely.
Pilot293 used float32 and all four A phases stalled before meeting the gradient
criterion. Subsequent pilots promote A parameters, model weights and floating
inputs to float64, verifying initial predictions against the unchanged float32
models within1e-3. This precision change is recorded in the protocol; it does
not change MatterSim precision or convergence thresholds. Record the continuous
gap's distance to the hinge boundary to diagnose nonsmooth stopping separately.
B reuses the pinned MatterSim worker, symmetry, fmax0.01, maxstep0.05 and1600-step
safety cap. Numerical/B failures are recorded individually.

After B, reconstruct lengths/angles in A and independently check the cell Gram
matrix with scale-aware float32 tolerances (rtol2e-6,atol1e-5); preserve ordered
fractional positions. This replaces the old absolute-only guard for this pilot,
not the force threshold. Absolute and relative errors remain in the evidence.
Track volume expansion because low forces alone do not establish stability.

Joint stopping requires a converged A phase, a converged B phase, and an A
projected residual<=1e-3 reevaluated at the B endpoint. Require three consecutive
rounds of unchanged composition, positional RMS<=0.001angstrom, relative cell
change<=1e-4, gap/Ef change<=0.001 and B force<=0.01. A and B geometrical movement
within the latest round must also satisfy those tolerances. Report raw-gradient
joint stationarity separately from constrained stationarity. A30-round safety
cap is not convergence. There is no guarantee a common stationary state exists.

Run from repository root using `sbatch experiments/smoacs/block_alternating/run.sbatch`.
All prediction and relaxation runs inside Slurm. Outputs: protocol/results,
per-round A diagnostics/B artifacts, history and complete marker in a fresh
ignored results directory. No full128 restart, reference downloads or DFT.
Tests: `python -m unittest discover -s experiments/smoacs/block_alternating`.
