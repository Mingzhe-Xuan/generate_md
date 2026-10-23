#!/bin/bash
#SBATCH --job-name=block-AB-pilot
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/smoacs/logs/block-AB-%j.out
#SBATCH --error=results/smoacs/logs/block-AB-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
.venv-smoacs-paper/bin/python experiments/smoacs/block_alternating/run.py \
 --output "results/smoacs/block-AB-${SLURM_JOB_ID}" --pilot --workers 1 --max-rounds 30
