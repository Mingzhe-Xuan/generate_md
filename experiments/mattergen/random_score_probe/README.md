# Random-prior unconditional MatterGen / MatterSim force probe

`run_random.py` draws16 independent structures from official MatterGen priors,
using ALEX_MP_20 atom counts, random coordinates/cells and masked atom types.
The existing frozen dft_band_gap checkpoint is run with properties={} and CFG=0:
this is its unconditional branch, not a separately downloaded mattergen_base.
No original crystal templates, composition constraints, fixed cells, property
gradients, force guidance, relaxation, SUN or DFT are used.

The verified official1000-step predictor/corrector sampler remains unchanged.
`observer.py` wraps its score function only to clone sparse states/scores toCPU.
There is one corrector then one predictor score call per time level. Observe the
exact random prior before the first corrector and15 predefined predictor inputs,
denser near the end (see `trajectory_math.STEPS`), giving256 snapshots.
The observer tests that CPU/CUDA RNG states do not change during capture. All
4000 batched model calls (two batches of8) must execute onCUDA, have finite
outputs, and contain no bandgap condition. Captures also verify the official
raw-output/noise-std score conversion using that same forward call.

After sampling, two persistent pinned CUDA MatterSim workers evaluate supported
snapshot states. MaskZ101 has no physical energy/force: any masked atom makes the
whole structure unknown. Unsupported real elements, nearly singular/nonpositive
cells, volume/atom<2A3 and pair distance<0.5A also remain explicit unknowns. No
types are replaced by predicted or final elements, no states are redrawn, and
no unknown is treated as zero cosine. The sampling trajectory itself is never
altered by these eligibility rules. Final sample structures (no masks) are saved
separately; comparisons refer to the sampled predictor input at the named time.

`trajectory_math.py` removes uniform translation and compares MatterSim atomic
force to two representations: Cartesian density score covector s_f L^{-T}, and
Cartesian pushforward of the sampler positional update s_f L. The scalar
diffusion variance does not alter the latter direction. These differ under the
Cartesian diffusion metric; both must be labeled. Lattice scores/stress and
discrete element logits are outside this comparison. Global, per-atom and
energy-directional diagnostics are recorded. Near-zero vectors have no cosine.

`trajectory_report.py` shows similarity AND eligibility counts across time.
Bootstrap intervals resample independent trajectories within each stage; repeated
timepoints are correlated. Early/late eligible cohorts can differ and early
known-atom counts are small. Positive cosine is local energy descent for the
learned potential, not a guarantee of stability or correct finite-step dynamics.
The first four predefined trajectory last-input states receive two-scale
energy/force finite differences if supported; unavailable cases stay recorded.

Input/config/source hashes, sampler call/condition/RNG audit,256NPZ snapshots,
16finalNPZs, full force records, controls and stage coverage are retained in a
fresh results directory. Failed roots must be preserved. No environment changes
or downloads are required: `.venv-mattergen` and `.venv-smoacs-postprocess` reuse
cached model/source manifests.

```bash
python -m unittest discover -s experiments/mattergen/random_score_probe -p 'test_*.py' -v
export RANDOM_SCORE_ROOT=results/mattergen/random-score-probe-20261029-seed50
sbatch experiments/mattergen/random_score_probe/run.sbatch
```
