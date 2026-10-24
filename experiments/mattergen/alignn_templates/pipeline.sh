#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Use Slurm}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_source="${MATTERGEN_TEMPLATE_SOURCE:-$PWD/results/mattergen/gap2p5-task128-20261024/full128}"
task_run="${ALIGNN_TEMPLATE_RUN:?}"
task_args=()
if [[ -n "${ALIGNN_GRADIENT_PROBE:-}" ]]; then
 task_args+=(--probe-report "$ALIGNN_GRADIENT_PROBE")
fi
if [[ ! -f "$task_run/complete.json" ]]; then
 .venv-smoacs-blackwell/bin/python experiments/mattergen/alignn_templates/run.py \
  --source "$task_source" --output "$task_run" --cohort "${ALIGNN_TEMPLATE_COHORT:-pilot}" "${task_args[@]}"
fi
if [[ ! -f "$task_run/evaluation-cycle1/inventory-249/inventory.json" ]]; then
 python3 experiments/smoacs/adam_cycles/export.py --run "$task_run" --cycles 1
fi
env BESTN_EVALUATION="$task_run/evaluation-cycle1" bash experiments/mattergen/best_of_n/evaluate.sh
