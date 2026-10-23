# Archived optimization cost statistics

Analyze the same128 CPU206 proposals, not the separate5874 MP reference phases.
`measurements.py` parses complete Crystalformer tqdm cumulative timestamps and
ASE BFGS clocks; `analyze.py` verifies candidate hashes, writes descriptive
JSON/CSV tables and PNG/SVG histograms. Inputs and outputs stay under ignored
results/smoacs; no inference, optimization or DFT is run.

Crystalformer uses one128-candidate CPU batch and a fixed200-update schedule.
The per-candidate step count is constant by design; this does not demonstrate
convergence at200. Recover the200 batch-update durations by differencing the
logged cumulative seconds, not tqdm's smoothed s/it. MatterSim uses individual
GPU BFGS optimizers. Exclude the initial step0 from update counts; require each
completed log to match its result.json optimizer_steps. Converged cases are the
primary cohort; report400-step unconverged cases separately as censored, and
failed cases' last logged indices as partial counts, not successful convergence.

Both logs have integer-second timestamps. Raw zero intervals are quantization,
not zero actual work. For a structure with N>0 updates, average seconds/update
is (last logged time - initial logged time)/N, with a clock-quantization error
of less than1/N seconds. These within-optimizer spans exclude model loading,
the initial logged evaluation and final export. Midnight rollover is supported;
missing/duplicate BFGS indices and other backward time jumps are rejected.
CPU batch steps and GPU single-structure steps are not a hardware speed benchmark.

Statistics use linear quantile interpolation at(n-1)*q and report N, minimum,
P5, P25, median, mean, P75, P95 and maximum. The main four-panel histogram shows
step counts and primary timing; a second image displays the raw timing
quantization and total BFGS spans. CSV files retain per-step and per-structure
values for auditing.

Run locally on the archived logs:

```bash
python experiments/smoacs/timing/analyze.py \
 --crystalformer results/smoacs/evidence-206 \
 --mattersim results/smoacs/postprocess-206-evidence/relaxation \
 --inventory results/smoacs/postprocess-206-evidence/inventory-249/inventory.json \
 --output results/smoacs/timing-206
```

This is lightweight local log analysis. If run on Guqq2, submit through Slurm.
Tests: `python -m unittest discover -s experiments/smoacs/timing`.
