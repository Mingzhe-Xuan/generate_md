#!/bin/bash
#SBATCH --job-name=mattersim-before
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/compare-singlepoint-%j.out
#SBATCH --error=results/smoacs/logs/compare-singlepoint-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/smoacs/comparison/singlepoint.py \
 --inventory "${SMOACS_COMPARE_INVENTORY:?}" \
 --checkpoint results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth \
 --output "${SMOACS_COMPARE_ROOT:?}/singlepoint-${SLURM_JOB_ID}"
