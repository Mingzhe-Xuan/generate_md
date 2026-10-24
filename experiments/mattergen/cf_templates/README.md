# MatterGen structures as CF initial templates

This module tests whether immutable generated poses can seed the existing
five-variable GPU Adam controller. `context.py` checks all128 inputs and builds
new neutral masks; `template_policy.py` finds at most10 exact neutral paths
without enumerating exponentially many combinations. The same SUPER_COMMON
four-way oxidation-state intersection is retained. Nonneutral inputs are
reported as ineligible for this constrained route, not replaced or relabelled.

Four eligible templates nearest4/8/12/20 atoms are selected by size and ID,
without selecting on predictor score or prior B success. Cells, fractional
coordinates and initial discrete elements must match the generated structures;
initial bounds expand only as needed to contain the input. More than10 atoms
is permitted. New masks use the actual ordered input atoms, never old frozen
template masks. Atom count and site ordering remain fixed during optimization;
element identities are subsequently learnable.

Element weights start at1/98, with0.001 extra on the original element. This
preserves hard identity while keeping a soft distribution for gradients.
`initialization.json` separately reports hard and soft initial predictions.
Frozen hard predictions must match recorded MatterGen predictions within1e-3.
The existing corrected Gaussian derivative, Adam rates,600-step cosine and
MatterSim B profile are reused. Native CUDA gradients for all five parameter
blocks and actual CUDA forward/backward calls are audited. This is an external
template pilot, not original-template or original-paper reproduction.

Run `sbatch experiments/mattergen/cf_templates/run.sbatch`. It uses existing
task environments/assets, writes only a fresh ignored
`results/mattergen/cf-template-pilot-<job>[-step<step>]` directory, and compares raw generation,
direct B, CF600 and CF600+B on the same four IDs with frozen CF/ALIGNN. All four
remain in the denominator. Pilot SUN is not computed; no full128 improvement
claim follows from this feasibility experiment. Existing production jobs
are not cancelled or modified.

`pipeline.sh` also supports a dedicated `srun --jobid=<owned-allocation>
--overlap --exact --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=32G`
step when available GPU/CPU/memory headroom has been checked. Step logs and
output directories must be separate. Only the batch wrapper sources the
job-level status recorder: a shared step must not overwrite its parent job's
exit status. The step's `srun` exit and complete/report files establish its
own outcome.

When attaching an external `srun` to an existing allocation, explicitly pass
`SLURM_SUBMIT_DIR` as the verified repository root. To finish evaluation after
a failed evaluator, set `MATTERGEN_CF_RUN` to the existing completed pilot:
the pipeline preserves its A/B trajectory and resumes only missing stages.
The frozen CF evaluator names its output with `SLURM_JOB_ID`, independently
of the step tag used for the pilot's containing directory.

Local policy tests: `python -S -m unittest discover -s
experiments/mattergen/cf_templates -p 'test_*.py'`.
