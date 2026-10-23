#!/bin/bash
#SBATCH --job-name=A600-B-eval
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/a600-eval-%j.out
#SBATCH --error=results/smoacs/logs/a600-eval-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_root="$SMOACS_A600_RUN/evaluation-cycle1"
python3 experiments/smoacs/adam_cycles/export.py --run "$SMOACS_A600_RUN" --cycles 1
.venv-smoacs-blackwell/bin/python experiments/smoacs/postprocess/predict_relaxed.py \
 --inventory "$task_root/inventory-249/inventory.json" --assets results/smoacs/assets-199 --root "$task_root" --device cuda
.venv-smoacs-blackwell/bin/python experiments/smoacs/cross_predictor/predict.py \
 --inventory "$task_root/inventory-249/inventory.json" --predictor "$task_root/predictor-$SLURM_JOB_ID" \
 --relaxation "$task_root/relaxation" --author-csv "$task_root/result.csv" \
 --weights results/smoacs/alignn-weights --output "$task_root/cross-prediction"
