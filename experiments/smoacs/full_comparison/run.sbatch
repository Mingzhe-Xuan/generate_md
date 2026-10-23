#!/bin/bash
#SBATCH --job-name=full128-AB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/smoacs/logs/full128-A-%j.out
#SBATCH --error=results/smoacs/logs/full128-A-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
.venv-smoacs-blackwell/bin/python experiments/smoacs/full_comparison/run.py \
 --output "${SMOACS_FULL_RUN:?}" --workers 4 --max-rounds "${SMOACS_FULL_CYCLE:?}"
