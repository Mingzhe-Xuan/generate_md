#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Run through Slurm}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_source="${MATTERGEN_TEMPLATE_SOURCE:-$PWD/results/mattergen/gap2p5-task128-20261024/full128}"
task_tag="$SLURM_JOB_ID"
if [[ -n "${SLURM_STEP_ID:-}" && "$SLURM_STEP_ID" != batch && "$SLURM_STEP_ID" != extern ]]; then
 task_tag="$task_tag-step$SLURM_STEP_ID"
fi
task_run="${MATTERGEN_CF_RUN:-$PWD/results/mattergen/cf-template-pilot-$task_tag}"
export MATTERGEN_CF_COHORT="${MATTERGEN_CF_COHORT:-pilot}"
if [[ ! -f "$task_run/complete.json" ]]; then
 .venv-smoacs-blackwell/bin/python experiments/mattergen/cf_templates/run.py --source "$task_source" --output "$task_run" \
  --cohort "$MATTERGEN_CF_COHORT" --microbatch-size "${MATTERGEN_CF_MICROBATCH:-4}" --steps "${MATTERGEN_CF_STEPS:-600}"
fi
python3 experiments/mattergen/cf_templates/repair_relaxation.py --run "$task_run" --workers 2
task_eval="$task_run/evaluation-cycle1"
if [[ ! -f "$task_eval/inventory-249/inventory.json" ]]; then
 python3 experiments/smoacs/adam_cycles/export.py --run "$task_run" --cycles 1
fi
task_predictor="$task_eval/predictor-$SLURM_JOB_ID"
if [[ ! -f "$task_predictor/summary.json" ]]; then
 .venv-smoacs-blackwell/bin/python experiments/smoacs/postprocess/predict_relaxed.py \
 --inventory "$task_eval/inventory-249/inventory.json" --assets results/smoacs/assets-199 --root "$task_eval" --device cuda
fi
if [[ ! -f "$task_eval/cross-prediction.complete" ]]; then
 if [[ -d "$task_eval/cross-prediction" ]]; then
  mv "$task_eval/cross-prediction" "$task_eval/cross-prediction-incomplete-$task_tag"
 fi
 .venv-smoacs-blackwell/bin/python experiments/smoacs/cross_predictor/predict.py \
 --inventory "$task_eval/inventory-249/inventory.json" --predictor "$task_predictor" \
 --relaxation "$task_eval/relaxation" --author-csv "$task_eval/result.csv" \
 --weights results/smoacs/alignn-weights --output "$task_eval/cross-prediction"
 touch "$task_eval/cross-prediction.complete"
fi
python3 experiments/mattergen/cf_templates/report.py --run "$task_run" --source "$task_source"
