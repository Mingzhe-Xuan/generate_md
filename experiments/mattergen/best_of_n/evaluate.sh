#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Use Slurm}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_eval="${BESTN_EVALUATION:?}"
if [[ -f "$task_eval/cross-prediction.complete" ]]; then exit 0; fi
task_predictor="$task_eval/predictor-$SLURM_JOB_ID"
if [[ ! -f "$task_predictor/summary.json" ]]; then
 .venv-smoacs-blackwell/bin/python experiments/smoacs/postprocess/predict_relaxed.py \
  --inventory "$task_eval/inventory-249/inventory.json" --assets results/smoacs/assets-199 --root "$task_eval" --device cuda
fi
if [[ -d "$task_eval/cross-prediction" ]]; then
 mv "$task_eval/cross-prediction" "$task_eval/cross-prediction-incomplete-$SLURM_JOB_ID"
fi
.venv-smoacs-blackwell/bin/python experiments/smoacs/cross_predictor/predict.py \
 --inventory "$task_eval/inventory-249/inventory.json" --predictor "$task_predictor" \
 --relaxation "$task_eval/relaxation" --author-csv "$task_eval/result.csv" \
 --weights results/smoacs/alignn-weights --output "$task_eval/cross-prediction"
touch "$task_eval/cross-prediction.complete"
