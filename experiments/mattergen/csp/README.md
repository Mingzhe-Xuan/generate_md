# Fixed-composition MatterGen CSP

Uses the MatterGen-derived CSP checkpoint published by the authors of
[GSS](https://github.com/Yifang-Qin/Generative-Structure-Search), with the existing
unmodified Microsoft MatterGen source pinned by `../sources.json`. This is a
different trained checkpoint from the previous `dft_band_gap` model. No GSS
energy guidance or band-gap conditioning is enabled in the baseline.

`sources.json` pins the release, byte length, publisher MD5 and config SHA256.
`model_config.yaml` is the author's unchanged MIT-licensed configuration.
Only coordinates and lattice are corrupted/trained; the retained atom-type
output head does not imply atom-type diffusion. `prepare.py` verifies the
download under Slurm and records its SHA256 before trusted legacy loading.
The only configuration override resolves the GemNet scale JSON to the existing
absolute packaged path; checkpoint state loading remains strict.

Assets go in ignored `results/mattergen/csp-assets/`, with checkpoint at
`checkpoints/last.ckpt.partial` before verification. Download from the pinned
Zenodo URL through the documented healthy SSH proxy on the login node.
Never replace the old source, checkpoint or environments.

For a slow endpoint, `download.py --direct` (only with direct-download permission)
or `download.py --proxy socks5h://127.0.0.1:1080` fetches eight exact byte ranges
with at most four concurrent curl downloads. This is network I/O only; assembly
and publisher checksum verification stay in `prepare.py` under Slurm. Interrupted
monolithic files remain intact. A same-sized chunk alone is never trusted for
model loading: the complete checkpoint must match the publisher MD5 first.

```bash
sbatch --export=ALL,CSP_ROOT=results/mattergen/csp-smoke-20261029 \
  experiments/mattergen/csp/smoke.sbatch
```

The smoke uses Si8 and SrTiO3, two samples each, batch one, seed20261051.
The native 1000-step predictor/corrector schedule gives2000 score calls per
structure. Atomic numbers are audited at every score input and both terminal
states. Model parameters and score geometry must be CUDA and finite. Positions
and lattice start from their native random priors, without reference geometry.
Six input states per trajectory are saved for independent chemistry auditing.

`run.py --output <new-directory> --compositions '[{"Si":8}]'
--samples-per-composition 2` selects explicit atom counts per cell. Reduced
formula ratios alone do not define the number of formula units. Fractional
counts are rejected rather than silently truncated by the upstream loader.
The initial scope is at most20 atoms and MatterSim-supported elements.

After generation, unchanged MatterSim5M RPC workers perform physical single
points and canonical BFGS/ExpCellFilter/FixSymmetry relaxation with fmax0.01,
maxstep0.05 and1600-step cap. Geometry files, audits, hashes, initial forces,
relaxation records and timing are retained. `complete.json.passed` requires
all requested samples and all relaxations to converge; failures remain visible.
This smoke establishes runtime integration, not property accuracy or SUN.
Early noisy-state force reliability is not established by final relaxation.

Existing `.venv-mattergen` (Torch2.7.1cu128) and `.venv-smoacs-postprocess` are
reused without package changes. Reported parent CUDA peak excludes RPC workers.
DFT and old broad TTO/dimer work remain paused.

```bash
python -m unittest discover -s experiments/mattergen/csp -p 'test_*.py'
```

Validated on2026-10-29: Slurm467 completed with exit0; all four native
1000-step trajectories passed8000 total score/denoiser-call checks. All four
MatterSim relaxations converged (7,21,14,86 steps), including raw atomic
fmax below0.01eV/angstrom. Sampling took407.61s and the core sampling/physics
workflow434.92s, excluding downloads. Slurm468 archived the evidence; local
independent checks verified source blobs, all24 saved states, ordered species,
geometry hashes, energies, force thresholds and full physical NFE accounting.
Ignored artifacts are under `results/mattergen/evidence/csp-smoke-20261029`.
This is runtime validation, with no property-throughput or SUN claim.

## Public bandgap-conditioned CSP checkpoint availability

Checked on 2026-10-30 following a request to download a fixed-composition,
bandgap-conditioned checkpoint. No verified matching MatterGen release was found
in the official catalog, the GSS release, or the public Hugging Face model search
for `mattergen`. This is a scoped search result, not proof that no such model exists.

The [Microsoft checkpoint catalog](https://huggingface.co/microsoft/mattergen/tree/main/checkpoints)
contains `dft_band_gap`, but its
[configuration](https://huggingface.co/microsoft/mattergen/raw/main/checkpoints/dft_band_gap/config.yaml)
includes atomic-number corruption and atomic-number loss: it jointly generates
species and geometry. The already cached GSS CSP configuration has no species
corruption/loss and has empty property embeddings, so it has no trained bandgap
condition. Downloading either again does not supply the combined capability.

The Hugging Face API returned 12 repositories matching `mattergen` during this
check. Public alternate configurations did not establish a matching release:
OneScience's gap model also corrupts species; the electride-named CSP config has
empty property embeddings; XPANCEO condition keys describe other objectives;
MP40 is described as unconditional. Opaque weight-only releases remain unverified.
Search metadata and small candidate configs are retained under ignored `tmp/`.
No new large weights, server connection, training or inference were needed.

A trained property adapter on a CSP backbone is a possible subsequent project.
Applying the joint bandgap model while clamping species would be a distinct
sampling intervention requiring validation, not a downloaded CSP checkpoint.

See [the proposed training protocol](TRAINING_PLAN.md) for an adapter-first
bandgap-conditioned CSP design, including polymorph-data checks, species-fixed
denoising supervision, sampling-condition forwarding and relaxed evaluation.
It is a proposal; no training run has been launched for it.

[Template-local editing](TEMPLATE_EDITING.md) describes a separate warm-start
route for near-final input structures: matched low-noise geometry corruption,
native suffix denoising and post-relaxation candidate selection. Existing joint
bandgap weights are a proposed low-noise editing candidate; this does not assert
that they are a trained fixed-composition CSP checkpoint. No editing run yet.
