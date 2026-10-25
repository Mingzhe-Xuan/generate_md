# Post-relaxation basin search on16 frozen MatterGen parents

This module explores the active user goal: improve CF joint yield or CF bandgap
MAE after MatterSim, with substantial same-parent changes from direct relaxation.
`plan.py` freezes16 uniformly sampled canonical-converged parents using seed
20261026, without filtering property scores, neutrality or previous outcomes.
All16 remain in the denominator. Fixed site elements are preserved.

`policy.py` fixes16 quench attempts per parent per method and explicit acceptance:
at least4 parents with author-matcher failure, species-pair distance RMS>=0.1 A
and CF absolute-error gain>=0.1 eV; additionally netCFjoint gain>=3 or pairedCF
MAE bootstrap95% interval entirely negative. Successful discovery requires a
second independent search seed before declaring the goal achieved. These
operational thresholds are fixed before seeing new results, not replacements
for the user's two requirements. ALIGNN is reported but never ranks candidates.

Initial methods: IID perturbations; a diverse archive/beam with collective atomic
and cell moves; short MD escapes along low-curvature directions followed by the
same canonical quench. They borrow ideas from basin hopping, minima hopping and
particle search, but are not literal reproductions of variable-cell minima MD
or diffusion-time Feynman-Kac sampling. MD uses fixed cell after a strain kick.
The extra MD/Hessian force calls are counted explicitly; equal quench count
does not establish equal computational cost.

`proposals.py` implements bounded collision-free moves and mass-weighted soft
directions with translational modes projected out. Mechanics and CF are separate
persistent CUDA workers in their existing postprocess/Blackwell environments.
CF scores every quenched pose; failed attempts consume budget; baselines remain
as final fallback. Physical eligibility requires both forces<=0.01 eV/A, no
distance<0.5 A, CF Ef<-0.5 eV/atom and MS energy<=baseline+0.05 eV/atom.
An unconstrained direct-relaxation control separates symmetry locking from search.
All candidate final volumes must also remain0.65..1.4 times the direct baseline.

`probe.sbatch` first validates frozenCUDA CF scores on all16 parents, replays
the canonical directB to check16 energies/structures, and runs four predetermined
600K MD numerical controls. `run.sbatch` can depend on its success, sharing an
absolute BASIN_RUN_ROOT. Completed requests have immutable task/input records and
are reused; infrastructure failures are restarted in fresh processes while
scientific rejections consume the attempt. The quench success of probe escapes
is reported separately from MD numerical correctness.

`confirm.sbatch` may depend on the discovery job. It reads the complete frozen
report and proceeds only if beam or soft_md passes discovery, then repeats all
three methods with seed20261027. `--frozen-plan` copies and validates the exact
original16 IDs and baseline hashes; it never resamples from a changed pool.
IID or symmetry-release control success alone does not trigger confirmation.
Skipped confirmation is not goal completion. Both seeds need independent final
evaluation and materiality auditing before a successful method is claimed.

`interim.sbatch` can inspect completed methods while later ones are running.
It reads an immutable snapshot of search-results.json, validates saved pose
hashes, and measures all16 same-parent structural changes without model calls.
It writes a separate interim-JOB.json with provisional=true and no final proof
flag. Confirmation reads only report.json from the complete independent
evaluation, so interim scores never trigger confirmation or certify success.

`scaling.sbatch` audits completed reports and reconstructs N=0,1,4,8,12,16
prefixes by attempted ID, so rejected proposals still consume budget. It checks
pose hashes, the frozen16-parent cohort and controls, recomputes structural
changes, and requires prefix16 to reproduce the independent final report.
Optional BASIN_CONFIRM_ROOT adds the second search seed on the same cohort.
The output compares CF MAE, substantial improvements and actual MatterSim force
evaluations. Prefix curves are descriptive and do not replace endpoint criteria
or establish a causal advantage of beam selection over different move sizes.

`endpoints.sbatch` packages all16 direct/selected pairs for each completed method
from BASIN_RUN_ROOT. It verifies independently scored geometry hashes and ordered
species, copies original POSCAR bytes, checks exact two-frame ASE trajectory
round trips, and renders every materially improved beam pair. The archive and
manifest retain mechanics records, neutral-mask provenance and both predictors.
These are saved endpoint visualizations, not new relaxations or phase proofs.

`attractors.sbatch` runs an optional CPU MatterSim diagnostic alongside GPU search.
For every material-improved beam endpoint and its directB comparator it first
checks saved GPU energy and maximum raw force against CPU (1e-4 tolerances).
Only then does it run three predetermined .02A/.003strain perturbations per
endpoint and unconstrained same-fmax re-quenches. All outcomes and return-to-own
versus other matches are recorded, without CF ranking or changed acceptance.
This tests sensitivity to small perturbations; it does not prove global basin
membership, DFT stability or positive phonon frequencies.

All native mechanics, inference, batch geometry and evaluation must run in Slurm.
No source edits occur on the server. Existing weights/environments are reused.
Generated poses, trajectories and reports are ignored task results.

Primary sources:
- https://arxiv.org/abs/cond-mat/9803344
- https://arxiv.org/abs/1007.2003
- https://arxiv.org/abs/2501.06848
