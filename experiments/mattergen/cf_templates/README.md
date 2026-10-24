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

## Full128, two concurrent initialization strategies

`sbatch experiments/mattergen/cf_templates/full.sbatch` processes all128 raw
MatterGen structures. The65 charge-compatible cases use the validated neutral
masks. The63 remaining cases use the author's genuinely unmasked element
preprocessing (`oxidation_target=None`); an inactive zero-gradient oxidation
tensor exists only to retain the controller interface. Unmasked optimization
has four active variable blocks. Initial ordered discrete species and poses
remain checked in both strategies, while soft initial predictions are distinct.

Two child pipelines inside one Slurm batch allocation share its GPU. Separate
GPU-requesting steps queued behind each other on the target cluster even with
`--overlap`, so the batch directly launches both children under its own resource
allocation, with separate strategy labels, logs and directories. CPU, CUDA
forward controls, hard predictions and gradients all stream microbatches of4;
gradient partitions2/4 must agree within1e-3. Each strategy performs600 global
Adam updates with its own mean loss, then uses two MatterSim workers. Frozen
CF/ALIGNN evaluation follows. No full retained128-structure graph is required.

`monitor.py` samples device memory and sums GPU processes by the current Slurm
job and strategy. It records only those environment labels, never other
environment values. Sampled NVIDIA MiB peaks and exact per-process Torch
allocator peaks are distinguished. Batch cleanup preserves its actual exit
code and stops only its own monitor; substeps do not overwrite job status.

`full_report.py` rejects overlapping/missing IDs, merges128 endpoints and
reports overall128 and separate65/63 strata, failures/unknowns and paired
bandgap errors. It schedules `references.sbatch`, which uses a union of direct
MatterGen-B and hybrid-B chemical systems, hash-validated reference caches and
unchanged serial reference relaxation. Both SUN evaluations use that same
pool. U is computed within each full128 method before stratification. DFT
remains paused. Output is under `results/mattergen/cf-full-<job>`.

### CUDA error isolation and retained CF98 vocabulary

The full experiment keeps the user's CF98 vocabulary. The loaded MatterSim
encoder's actual maximum atomic number is checked on CPU before evaluation;
unsupported output elements are explicit model-domain failures, never silently
replaced. A device assertion poisons its worker, so later records from that
context are retryable infrastructure artifacts.

`repair_relaxation.py` preserves A traces, checkpoints, protocols, successful B
records and genuine step-limit/symmetry outcomes. Only retryable GPU records
are archived and checked in fresh per-case worker processes; unsupported
elements are rejected without a GPU call. Repaired physical B endpoints are
marked `B_converged_posthoc_repair`: no further A phase or optimizer feedback
is claimed. Old evaluator output is archived before re-export. The completed
optimizer tensors remain unchanged and must not be treated as repaired B
feedback for another A phase.

`repair.sbatch` finishes this recovery/evaluation for both strategies, refreshes
the merged128 outputs, archives stale SUN/scope reports, and schedules a new
common-reference assessment. Prior phase caches remain available with strict
input/model/pose hashes; poisoned GPU cache records are excluded.

Local policy tests: `python -S -m unittest discover -s
experiments/mattergen/cf_templates -p test_template_policy.py`.
All local module tests use regular `python -m unittest discover -s
experiments/mattergen/cf_templates` with the existing Torch test environment.

### Shorter single-phase budget

Use `sbatch --export=ALL,MATTERGEN_CF_STEPS=200 experiments/mattergen/cf_templates/full.sbatch`
from the repository root. The default remains600. Both
strategies use the same selected budget, original inputs and base Adam rates;
cosine decay spans the selected200 or600 updates, so200 is a complete shorter
schedule rather than the prefix of a600 schedule. Variable blocks, CF98
vocabulary, microbatch size, loss and B profile are unchanged. Reports read
the actual saved protocol and reject mixed budgets or incorrect per-case
update counts;200 endpoints are labelled CF200 throughout.

Optionally set `MATTERGEN_CF_REFERENCE_CACHE` to a prior selection directory
under task results. Reference reuse still requires matching input/model and
validated converged-geometry hashes. Each run builds a fresh common reference
pool for its direct/hybrid comparison; different runs can select different
pools, which must be disclosed when comparing SUN across budgets.
