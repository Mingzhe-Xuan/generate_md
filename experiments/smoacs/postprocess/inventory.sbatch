#!/bin/bash
#SBATCH --job-name=cpu206-inventory
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=results/smoacs/logs/post-inventory-%j.out
#SBATCH --error=results/smoacs/logs/post-inventory-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2
.venv-smoacs-blackwell/bin/python experiments/smoacs/postprocess/inventory.py \
 --input results/smoacs/optimization/section21-Struct-AtomicMask-grad-206 \
 --output "results/smoacs/postprocess-206/inventory-${SLURM_JOB_ID}"
