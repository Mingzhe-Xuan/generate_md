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

## Verified asset preparation

assets.json pins the public archive sizes and SHA-256 digests. Copy the two archives
and crystalformer_default.json into results/smoacs/downloads, then submit
`sbatch experiments/smoacs/prepare.sbatch` from the parent root. The job writes a new
results/smoacs/assets-<job_id> directory, verifies archives, extracts only template
data and the two layer-4 checkpoints, and writes prepared.json. It excludes source
scripts, rejects unsafe paths and refuses to overwrite an existing output.

requirements-paper.txt describes the initial paper runtime; obtain Torch/DGL CUDA
wheels from the official indexes separately. Treat installed-version snapshots and
Slurm runtime checks as evidence; installation alone does not certify GPU compatibility.
Run the tiny preparation tests with python -m pytest -q experiments/smoacs/test_prepare_assets.py.

## Original Section 2.1 execution

run_section21.py verifies the archive and original author script SHA-256, then executes
those unchanged source bytes in a Slurm job. The wrapper supplies paths and CLI
arguments, adds the pinned checkout to the Python import path, and records provenance.
It does not write or patch upstream source on the server. Set SMOACS_ASSETS to the
prepared assets directory before submitting section21.sbatch. The defaults are 256
candidates in a complete batch, 200 steps, seed 42, template geometry, oxidation mask
and gradient optimization. Select other methods with SMOACS_METHOD and initialization
with SMOACS_INITIALIZATION. For a smoke check, explicitly set SMOACS_CANDIDATES=2,
SMOACS_BATCH_SIZE=2, SMOACS_STEPS=10; mutation methods additionally require
SMOACS_MUTATION_STEPS="5 7". Every job uses a fresh experiment directory.

The archived author script uses a 0.04 eV training window in its saved summary.
Recompute the paper 0.2 eV assessment from all per-candidate rows separately.
Report candidate count, actual saved configuration, device implementation, and any
difference from the paper; keep CPU smoke results separate from CUDA reproduction.

When the full predictor ZIP is expensive to transfer, extract its two layer-4 files
locally without changing their bytes and transfer them separately. checkpoints.json
pins their individual hashes and their original archive provenance. Set
SMOACS_DIRECT_CHECKPOINTS=1 when submitting prepare.sbatch to verify these files
instead of requiring the complete four-model ZIP. prepared.json records exactly
which inputs were verified; both modes produce the same two model files.

## Durable job status

Every batch entry sources job_status.sh and writes results/smoacs/status/job-<id>.json
with its real exit code and UTC finish time. This supplements controller observations
on clusters without an accounting database. status_check.sbatch validates only this
recorder, not models or optimization. Local tests need Bash and pytest; set
SMOACS_TEST_BASH when Bash is not on PATH. A killed job may lack a final file, so
a missing record must never be interpreted as successful completion.

## Bounded transfer over the existing SOCKS route

range_server.py serves only public .whl files from an explicitly selected local
directory. It binds 127.0.0.1 on an ephemeral port and reports that port/PID.
Each request is limited to 256 KiB and includes a SHA-256; HTTP/1.1 permits
connection reuse. Keep it private and stop the task-owned helper after use.

range_fetch.py runs as a lightweight download command with an explicit socks5h
proxy, verifies every range before appending, and checkpoints atomically. It
truncates uncommitted tails and resumes verified offsets after interruption.
Use --only <wheel-name> for a transport test; its report is marked subset.
Without --only it fetches the full source manifest. Files remain resources
under ignored results/smoacs, and still need installation/runtime validation.

Run both sides following the root network rules: one dedicated SSH proxy, an
end-to-end health check, and ordinary work connections with forwarding disabled.
This route does not require public HTTP hosting or a direct-download exception.

## Offline environment installation

After downloading complete Linux/Python-compatible binary wheels, submit
SMOACS_WHEELS=results/smoacs/direct-wheels sbatch experiments/smoacs/install_env.sbatch
from the repository root. This installs only into SMOACS_PYTHON (default the task
venv), uses no network, requires the pinned Torch/DGL builds, runs pip check and
saves freeze-<job_id>.txt. A successful install is followed by probe.sbatch; it
does not establish GPU compatibility or paper reproduction by itself.

The archived script executes in a registered __main__ module with its own compile
flags. Launcher future imports must not change author annotations; this matters
for Pydantic models and dataclass type resolution. Author bytes remain unchanged.

## Paper Section 2.1 assessment

Set SMOACS_RESULT_CSV to the author's result.csv and SMOACS_EXPECTED_CANDIDATES
to the full submitted candidate count, then submit summarize.sbatch (optionally
with an afterok dependency). summarize_section21.py independently applies the
paper bandgap [2.3,2.7] eV, formation energy < -0.5 eV/atom, neutrality and
minimum-bond flags. Missing rows remain failures in the declared denominator.
The original narrow-window success column is preserved and is not substituted
for this assessment. Output JSON includes the source CSV SHA-256.

## RTX 5090 / Blackwell profile

Keep the paper environment intact. Create .venv-smoacs-blackwell using Python
venv --without-pip before installation, and download the requirements-blackwell.txt
binary wheels into ignored runtime directories. install_blackwell.sbatch installs
offline using the paper interpreter only as a pip bootstrap, targeting the new venv.
This profile changes Torch to 2.7.1+cu128 and records all resolved versions.

Set SMOACS_PYTHON=$PWD/.venv-smoacs-blackwell/bin/python and
SMOACS_CUDA_WHEEL_RUNTIME=1 for probe.sbatch and section21.sbatch.
blackwell_runtime.sh selects CUDA headers and libraries from that venv rather than
an older system toolkit. The probe includes a real CuPy RawKernel compilation.

For GPU smoke, also set SMOACS_REQUIRE_CUDA_KERNELS=1, SMOACS_DEVICE=cuda,
SMOACS_CANDIDATES=2, SMOACS_BATCH_SIZE=2, SMOACS_STEPS=10 and the prepared assets.
cuda_audit.py observes the real CrystalformerEncoderCUDA forward calls and finite
CUDA backward gradients without changing tensors or optimizer updates. A fallback
encoder, missing gradients or non-finite outputs fails the job; the audit JSON
accompanies the launch record. This establishes the Crystalformer path only,
not ALIGNN's full GPU operation support or paper-scale reproduction.

The Blackwell profile uses the official DGL 1.1.3+cu121 wheel: DGL 2.4 wheels
pin torch<=2.4.0 and cannot resolve with Blackwell Torch. Dependency checking
remains enabled; the runtime probe and Crystalformer smoke determine the actual
supported scope. Do not assume this proves ALIGNN compatibility.
