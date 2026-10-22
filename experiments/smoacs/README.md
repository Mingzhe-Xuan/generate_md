# SMOACS reproduction tools

The upstream implementation is a pinned submodule at `../../third_party/SMOACS`.
Use `git submodule update --init --recursive` after checking out the parent repository.
Provenance and licensing are described in [../../third_party/README.md](../../third_party/README.md).

## Interfaces and outputs

- `preflight.py`: standard-library source, provenance and asset inventory. It does not import machine-learning packages or run optimization. Exit 0 means the listed assets are present and match; exit 2 means missing or mismatched assets. This is not an environment or paper-results certificate.
- `env_probe.py`: Torch/DGL CPU and requested CUDA checks, gradients, and upstream CLI imports. It requires a Slurm job ID and exits 1 on any failed check. It never silently switches devices.
- `probe.sbatch`: Slurm entry point, submitted from the parent repository root. `SMOACS_PYTHON` selects the dedicated environment interpreter; `SMOACS_DEVICE` selects cpu or cuda, default cuda. The default interpreter is `.venv-smoacs-paper/bin/python`.
- Generated inventories, job logs, environment snapshots, weights, templates and optimization results belong under ignored runtime directories. Do not commit them as source.

## Usage

From the parent repository root:

```bash
python experiments/smoacs/preflight.py --output results/smoacs/preflight.json
mkdir -p results/smoacs/logs results/smoacs/env
sbatch experiments/smoacs/probe.sbatch
```

Inspect both the Slurm exit state and `results/smoacs/env/probe-<job_id>.json`.
The environment probe does not reproduce any paper success rate. Full optimization
must additionally load the fixed templates and corresponding pretrained weights,
complete the optimizer schedule, discretize candidates and export per-candidate results.

## External assets and fidelity

The upstream README references a separate reproducibility archive, which must be
checked against the actual checkout: the pinned Git tree does not include the
`reproducibility/` scripts. Do not assume these commands exist merely because
the README links to them.

The paper's Crystalformer property predictor is
[omron-sinicx/crystalformer](https://github.com/omron-sinicx/crystalformer), associated
with Taniai et al., ICLR 2024. The similarly named deepmodeling/CrystalFormer
generative model is a different project. Match the predictor configuration and
MEGNet band-gap/formation-energy checkpoints to the correct release.

Preserve the paper's evaluation definitions, sample count, seeds and schedules.
Sections 2.1/2.2 use 200 steps; Section 2.3 uses 600 iterations and Section 2.4 uses
650 iterations, each totaling 1000 updates when SA is counted. Predictor success,
relaxed structure quality, S.U.N. screening and DFT validation are distinct
measurements. Report differences rather than treating an execution smoke check
as numerical reproduction.

Follow the root AGENTS.md for environment records, source synchronization and
Slurm-only computation. Edit and commit source locally; do not patch tracked
source on the server.
