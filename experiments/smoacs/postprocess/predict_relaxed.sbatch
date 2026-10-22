#!/bin/bash
#SBATCH --job-name=cpu206-drift
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/post-predict-%j.out
#SBATCH --error=results/smoacs/logs/post-predict-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4
export DGLBACKEND=pytorch
task_extra=()
if [[ -n "${SMOACS_PREDICT_LIMIT:-}" ]]; then task_extra+=(--limit "$SMOACS_PREDICT_LIMIT"); fi
.venv-smoacs-paper/bin/python experiments/smoacs/postprocess/predict_relaxed.py \
 --inventory "${SMOACS_INVENTORY:?}" --assets results/smoacs/assets-199 \
 --root results/smoacs/postprocess-206 "${task_extra[@]}"
