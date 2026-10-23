#!/bin/bash
#SBATCH --job-name=AdamAB-properties
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/adam-eval-%j.out
#SBATCH --error=results/smoacs/logs/adam-eval-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
.venv-smoacs-paper/bin/python experiments/smoacs/adam_cycles/export.py --run "${SMOACS_ADAM_RUN:?}"
for task_cycle in 1 3; do
 task_root="$SMOACS_ADAM_RUN/evaluation-cycle$task_cycle"
 .venv-smoacs-blackwell/bin/python experiments/smoacs/postprocess/predict_relaxed.py \
  --inventory "$task_root/inventory-249/inventory.json" --assets results/smoacs/assets-199 --root "$task_root" --device cuda
 .venv-smoacs-blackwell/bin/python experiments/smoacs/cross_predictor/predict.py \
  --inventory "$task_root/inventory-249/inventory.json" --predictor "$task_root/predictor-$SLURM_JOB_ID" \
  --relaxation "$task_root/relaxation" --author-csv "$task_root/result.csv" \
  --weights results/smoacs/alignn-weights --output "$task_root/cross-prediction"
 .venv-smoacs-paper/bin/python experiments/smoacs/cross_predictor/summarize.py \
  --input "$task_root/cross-prediction" --output "$task_root/cross-analysis"
done
