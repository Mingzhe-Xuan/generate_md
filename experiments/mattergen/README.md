# MatterGen on the current property/SUN task

Use the official pinned MatterGen dft_band_gap model, target2.5eV,
classifier-free guidance2.0, seed42 and its unchanged1000-step sampler.
Generate128 de novo structures. These are independent generated samples,
not paired reconstructions of the128 SMOACS starting templates.
The public checkpoint conditions on band gap; formation energy is an evaluation
constraint rather than an additional learned conditioning input.

`sources.json` pins the upstream commit, source archive, config and512MB weight
hash. Generated assets belong in ignored `results/mattergen`. `setup.py` creates
an isolated inference venv with read-only access to the existing Blackwell
Torch2.7.1+cu128/scientific packages. New dependencies are installed only in the
new venv. This explicitly replaces upstream's Torch2.2.1 pin for RTX5090 support;
the upstream model and sampler source remain unchanged. Freeze and pip-check
results are saved. Dataset downloads and training from scratch are unnecessary
for checkpoint reproduction. Official released checkpoints differ from the
original publication's training runs, as documented upstream.

All setup, native model checks, generation, inference, relaxation and reference
assessment run through Slurm. Common frozen Crystalformer/ALIGNN evaluators,
MatterSim5M B(fmax0.01, maxstep0.05,1600 cap), MP2020 hull<0.1eV/atom, within-set U
and relaxed-reference N preserve the existing task definitions. Failures and
unknowns stay in the requested denominator. DFT stays paused.

Before MatterGen, preserve the full128 experiment's matching JSON/PT checkpoint
and hold its pending chain. Requeue the paused B-pending job in held state to
release the GPU. Its phase-resume path restores all variables and Adam moments
and skips the completed A phase. After successful MatterGen end-to-end
verification, release exactly that chain and verify the saved trajectory resumes.
