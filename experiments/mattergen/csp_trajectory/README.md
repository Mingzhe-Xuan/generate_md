# Pure CSP denoising / force trajectories

Completed experiment and provenance: [RESULTS.md](RESULTS.md).

Observational mechanism study on the 12 GSS fixed-composition systems, two paths
per system. This is not a held-out CSP structure-recovery benchmark: checkpoint
training overlap is unknown. `prepare.py` reuses property-blind hash selection of
one ordered 2–12-atom GSS reference per system, then exports only integer species
counts to the sampler. Reference positions and lattices are never sampler inputs.

`run.py` uses the verified pure-CSP checkpoint, random native priors, native
1000-step predictor/corrector schedule, batch size eight, and zero guidance.
The RNG-neutral observer saves the prior and all 1000 predictor-input states.
All species are fixed; coordinates and lattice evolve. Paths within a native
adaptive-corrector batch are coupled, so 24 paths in three batches are not 24
independent experimental units. No trajectory bootstrap is used.

For each actual state, the isolated pinned CUDA MatterSim worker evaluates a
single point without relaxation. The prior joint-model geometry guards are kept:
positive nonsingular cell, minimum singular value 0.3 A, volume/atom >= 2 A^3,
periodic pair distance >= 0.5 A, supported species. Unsupported/failed/near-zero
vectors remain unknown, never zero cosine. Cell-force/stress alignment is outside
this coordinate-direction study.

`metrics.py` measures true adjacent predictor steps only (missing steps break
sequences), downward transitions at tolerances 0, 0.01, 0.05, maximum drawdown
inside contiguous segments, and nonoverlapping complete 25-step block means.
`report.py` independently audits every force-ledger geometry and cosine against
the saved arrays, then reports changing coverage and fixed complete cohorts for
the last 200/100/50 steps. Score covectors use `s_frac @ inv(cell).T`; coordinate
update directions use `s_frac @ cell`. Both vectors and forces are centered.

Run from repository root under Slurm:

```bash
export CSP_TRAJECTORY_ROOT=results/mattergen/csp-trajectory-20261030
sbatch --export=ALL experiments/mattergen/csp_trajectory/prepare.sbatch
# Submit run.sbatch afterok:prepare, controls.sbatch afterok:run,
# then report.sbatch afterok:controls.
```

Modules: `prepare.py` writes `selection/selection.json` and
`composition_plan.json`; `run.py` writes dense per-path NPZ arrays, terminal
states, force ledger, progress and `complete.json`; `metrics.py` is pure numeric
analysis; `report.py` writes audited `report.json`, `REPORT.md`, PNG/SVG and
`evidence.tar.gz`. Local tests: `python -m unittest discover -s
experiments/mattergen/csp_trajectory -p 'test_*.py'`.

`controls.py` checks four prespecified last predictor inputs (path IDs 0,7,15,23)
against two-scale central energy differences (0.002/0.001 A), using the existing
force-check tolerance 0.005 + 0.02*max(abs(FD),abs(expected)). Unknown endpoints
are reported as such. These GPU controls run after the observational sampling
job and cannot change its random trajectory. Their separate ledger is audited.

`energy_trajectory.py` audits the existing raw single-point energies and plots
one15-panel figure: pooled energy change from each path's last observed input,
a fixed-cohort late-window view, energy coverage, and absolute energy/atom for
all12systems with two paths each. It retains missing-state gaps and uses25-step
medians only when at least20states are present. It writes CSV, JSON, Markdown
and PNG/SVG/PDF with raw input and source hashes. No new inference is needed.
The plotted original dense run is distinct from the later800/1000 replay.

```bash
python experiments/mattergen/csp_trajectory/energy_trajectory.py \
  --source results/mattergen/evidence/csp-trajectory-20261030 \
  --output results/mattergen/evidence/csp-energy-trajectory-20261030
```

Cached local analysis is lightweight. If running this batch analysis remotely,
submit it through Slurm using an existing postprocess environment.

`energy_diagnostic.py` explains gaps and apparent midpoint changes using the
same immutable data: linear relative-energy view, closest-pair distances,
SHA-verified smooth forward-noise schedule, and omitted-block fractions. It
keeps the distinction between forward marginal noise and actual adaptive sampler
step size. Its50-step median-drop locations are descriptive window statistics,
not phase-transition or barrier estimates. Example:

```bash
python experiments/mattergen/csp_trajectory/energy_diagnostic.py \
  --source results/mattergen/evidence/csp-trajectory-20261030 \
  --native-source results/mattergen/evidence/mattergen-source-review/source \
  --output results/mattergen/evidence/csp-energy-diagnostic-20261030-final
```
