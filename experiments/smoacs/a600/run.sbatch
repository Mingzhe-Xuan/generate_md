#!/bin/bash
#SBATCH --job-name=A600-B-GPU
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/a600-%j.out
#SBATCH --error=results/smoacs/logs/a600-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
.venv-smoacs-blackwell/bin/python experiments/smoacs/a600/run.py \
 --output "${SMOACS_A600_RUN:?}" --pilot --workers 1 --max-rounds 1
