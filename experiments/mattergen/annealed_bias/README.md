# Annealed CF bias in the MatterSim energy landscape

This is a new fixed-species directed TTO method. The full development study is
planned on the unchanged16 source parents, using their common relaxed baseline
and the same seeded small perturbation for every arm. Native numerical proofs
precede any post-quench efficacy trial. Previous negative studies stay unchanged.

The conservative objective in the existing Cartesian-RMS/symmetric-strain
coordinates q is `Phi=(E_MS(q)-E_reference)/N + lambda*S(q)`. Energy is eV/atom;
`L_CF=.5*(CF_gap-2.5)**2` has units eV^2. Lambda therefore has units
eV/atom per eV^2. Full uses `S=L_CF`, reverse uses `S=-L_CF`, random uses a fixed
linear potential with initial atomic/cell-block gradient norms matched to CF,
and zero uses no bias. A fixed-bias arm uses the true CF objective at constant
peak strength. Formation is an evaluation guard, not a gradient objective.

`anneal_policy.py` freezes peaks(.1/.3/1), eight bias fractions
(.1/.25/.5/1/.5/.25/.1/0), max200FIRE updates per stage and the original2A/.15
strain/.7A separation proposal bounds. Only q carries between stages: velocity
and adaptation restart at each strength. A stage gradient tolerance of.002 is
an optimizer criterion, not a raw MatterSim force/stability claim. Failed or
stalled stages remain recorded. Intermediate states may exceed the final energy
ceiling; final quenches retain the original physical/formation guards.

`anneal_math.py` defines the conservative potentials and bounds.
`anneal_fire.py` supplies FIRE vector updates with explicit feasibility
backtracking. Pure tests check both a double-well case where bias can retain a
new minimum after release and a single-well case where all gains disappear.
These are optimizer tests, not crystal evidence.

`anneal_objective.py` connects unchanged CUDA CF derivatives and counted
MatterSim energy/force/stress workers, caching only identical q values.
`anneal_native.py` proves components
at their appropriate precision: tiny same-weight F64 differences for CF and
coarser F32 potential differences for MatterSim. Different finite-difference
scales must not be combined into a misleading joint derivative test. It checks
all16 source baselines and same-saved-pose official CF values, then the first4
parents'48 component differences and60 mixture identities. Two micro-updates at
two schedule slots per arm test the actual integration; they do not evaluate
post-quench efficacy. Use `native.sbatch` with `ANNEALED_SOURCE` pointing to the
original directed16 run and a fresh `ANNEALED_ROOT`. Existing blackwell CF and
postprocess MatterSim environments are reused; all model work requires Slurm.

The intended study forks full canonical MatterSim quenches at stages3/4/7 for
each of three peak strengths. Five arms give240steering trajectories and720fork
quenches, plus16shared seed quenches. Every arm gets the same9-quench budget. All nine
zero-CF FIRE endpoints, shared seed-B and originalB form the common fallback
pool. Thus better physical optimization alone cannot establish gradient value.
Frozen CF selects; ALIGNN only evaluates. Original energy+0.05eV/atom, original
volume.65..1.4, force.01/raw.01 and CF formation<-0.5 remain final guards. Any
positive development finding still needs new-parent and matched tighter-force
confirmation. No SUN/DFT certification is implied.

`anneal_experiment.py` runs either the first predefined parent (`--count 1`) as
an integration check or all16 (`--count 16`) with identical schedules/controls.
It requires the unchanged complete native proof, repeats baseline/seed controls,
records each stage's solver and physical/bias telemetry, and validates all
query/trajectory/pose/NFE/shared-pool links. `anneal_report.py` evaluates ten
selected sets with the original frozen CF/ALIGNN pipeline. Integration1 cannot
set population-efficacy flags. Full16 requires>=4 material gains>=.1eV beyond
the complete zero-CF pool and>=.05eV/negative95%CI against reverse/random/zero;
an additional fixed-bias contrast tests the value of annealing. The three full
stage subsets have3newforks rather than9 and are diagnostic readouts.

Use `run.sbatch` with `ANNEALED_SOURCE`, `ANNEALED_NATIVE`, a fresh `ANNEALED_ROOT`
and `ANNEALED_COUNT=1` or16. No hyperparameters or parent selection may be tuned
to pilot outcomes. The integration run proves the complete pipeline, not the
scientific claim; the unchanged16-parent run remains required.
