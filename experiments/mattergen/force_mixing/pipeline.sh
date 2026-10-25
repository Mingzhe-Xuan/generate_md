#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Use Slurm}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_run="${FORCE_MIX_RUN:?}"
task_source="${MATTERGEN_TEMPLATE_SOURCE:-$PWD/results/mattergen/gap2p5-task128-20261024/full128}"
task_args=()
if [[ -n "${FORCE_MIX_PROBE:-}" ]]; then
 task_args+=(--physics-probe "$FORCE_MIX_PROBE")
fi
if [[ ! -f "$task_run/complete.json" ]]; then
 .venv-smoacs-blackwell/bin/python experiments/mattergen/force_mixing/run.py \
  --source "$task_source" --output "$task_run" --weight "${FORCE_MIX_WEIGHT:?}" \
  --cohort "${FORCE_MIX_COHORT:-pilot}" --composition "${FORCE_MIX_COMPOSITION:-linear}" "${task_args[@]}"
fi
if [[ ! -f "$task_run/evaluation-cycle1/inventory-249/inventory.json" ]]; then
 python3 experiments/smoacs/adam_cycles/export.py --run "$task_run" --cycles 1
fi
env BESTN_EVALUATION="$task_run/evaluation-cycle1" bash experiments/mattergen/best_of_n/evaluate.sh
