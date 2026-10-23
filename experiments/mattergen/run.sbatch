#!/bin/bash
#SBATCH --job-name=MatterGen-task
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mattergen-task-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-task-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache" HF_HUB_OFFLINE=1
task_run="${MATTERGEN_RUN:?}"
mkdir -p "$task_run"
if [[ ! -f "$task_run/generated/complete.json" ]]; then
 .venv-mattergen/bin/python experiments/mattergen/generate.py --output "$task_run/generated" \
  --count "${MATTERGEN_COUNT:-128}" --batch-size "${MATTERGEN_BATCH:-16}"
fi
if [[ ! -f "$task_run/inventory-249/inventory.json" ]]; then
 .venv-smoacs-blackwell/bin/python experiments/mattergen/export.py --run "$task_run"
fi
if [[ ! -f "$task_run/reference-scope.json" ]]; then
 .venv-smoacs-blackwell/bin/python experiments/mattergen/relax.py --run "$task_run" --workers 4
fi
if [[ ! -f "$task_run/cf-predictor-path.txt" ]]; then
 .venv-smoacs-blackwell/bin/python experiments/smoacs/postprocess/predict_relaxed.py \
  --inventory "$task_run/inventory-249/inventory.json" --assets results/smoacs/assets-199 --root "$task_run" --device cuda
 printf '%s\n' "$task_run/predictor-$SLURM_JOB_ID" > "$task_run/cf-predictor-path.txt"
fi
if [[ ! -f "$task_run/cross-prediction.complete" ]]; then
 if [[ -d "$task_run/cross-prediction" ]]; then mv "$task_run/cross-prediction" "$task_run/cross-prediction-incomplete-$SLURM_JOB_ID"; fi
 task_predictor="$(cat "$task_run/cf-predictor-path.txt")"
 .venv-smoacs-blackwell/bin/python experiments/smoacs/cross_predictor/predict.py \
  --inventory "$task_run/inventory-249/inventory.json" --predictor "$task_predictor" \
  --relaxation "$task_run/relaxation" --author-csv "$task_run/result.csv" \
  --weights results/smoacs/alignn-weights --output "$task_run/cross-prediction"
 touch "$task_run/cross-prediction.complete"
fi
