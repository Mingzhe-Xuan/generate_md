#!/bin/bash
#SBATCH --job-name=block-grad-probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/block-grad-%j.out
#SBATCH --error=results/smoacs/logs/block-grad-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch SMOACS_BLOCK_PROBE=1
.venv-smoacs-paper/bin/python experiments/smoacs/block_alternating/run.py \
 --output "results/smoacs/block-gradient-probe-${SLURM_JOB_ID}" --pilot --workers 0 --max-rounds 1
