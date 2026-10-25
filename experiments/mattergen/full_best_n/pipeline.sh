#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${FULL_BESTN_ROOT:?}" "${FULL_BESTN_SHARD:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
printf -v task_tag '%02d' "$FULL_BESTN_SHARD"
task_shard="$FULL_BESTN_ROOT/shards/$task_tag"
task_source="$PWD/results/mattergen/gap2p5-task128-20261024/full128"
test -f "$task_shard/parents.json"
task_monitor=""
finish_monitor() {
 local task_exit=$?
 trap - EXIT
 touch "$task_shard/memory.stop"
 if [[ -n "$task_monitor" ]]; then wait "$task_monitor" || true; fi
 exit "$task_exit"
}
trap finish_monitor EXIT
python3 experiments/mattergen/cf_templates/monitor.py --output "$task_shard" --stop-file "$task_shard/memory.stop" &
task_monitor=$!
run_amplitude() {
 local task_amplitude="$1"
 local task_run="$task_shard/rms-$task_amplitude"
 export MATTERGEN_CF_COHORT="bestN_full_${task_tag}_${task_amplitude}"
 .venv-smoacs-postprocess/bin/python experiments/mattergen/best_of_n/prepare.py \
  --source "$task_source" --output "$task_run" --amplitude "$task_amplitude" --proposal-cap .60 --parents "$task_shard/parents.json"
 if [[ ! -f "$task_run/cf20/complete.json" ]]; then
  .venv-smoacs-blackwell/bin/python experiments/mattergen/cf_templates/run.py \
   --source "$task_source" --output "$task_run/cf20" --cohort pilot --fixed-species --steps 20 \
   --parent-ids "$task_shard/parents.json" --branch-manifest "$task_run/manifest.json"
 fi
 if [[ ! -f "$task_run/cf20/evaluation-cycle1/inventory-249/inventory.json" ]]; then
  python3 experiments/smoacs/adam_cycles/export.py --run "$task_run/cf20" --cycles 1
 fi
 .venv-smoacs-postprocess/bin/python experiments/mattergen/best_of_n/run_raw.py --run "$task_run"
 env BESTN_EVALUATION="$task_run/raw-evaluation" bash experiments/mattergen/best_of_n/evaluate.sh
 env BESTN_EVALUATION="$task_run/cf20/evaluation-cycle1" bash experiments/mattergen/best_of_n/evaluate.sh
 .venv-smoacs-postprocess/bin/python experiments/mattergen/best_of_n/report.py --run "$task_run"
}
task_pids=()
for task_amplitude in 0.10 0.20 0.30; do
 run_amplitude "$task_amplitude" > "$task_shard/rms-$task_amplitude.out" 2> "$task_shard/rms-$task_amplitude.err" &
 task_pids+=("$!")
done
task_failed=0
for task_pid in "${task_pids[@]}"; do wait "$task_pid" || task_failed=1; done
if [[ "$task_failed" != 0 ]]; then exit 1; fi
python3 experiments/mattergen/full_best_n/verify_shard.py --run "$task_shard"
touch "$task_shard/memory.stop"
wait "$task_monitor"
task_monitor=""
git rev-parse HEAD > "$task_shard/source-commit.txt"
