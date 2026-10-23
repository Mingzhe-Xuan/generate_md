#!/bin/bash
#SBATCH --job-name=block-AB-report
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=results/smoacs/logs/block-report-%j.out
#SBATCH --error=results/smoacs/logs/block-report-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
python3 experiments/smoacs/block_alternating/summarize.py \
 --run "${SMOACS_BLOCK_RUN:?}" --gradient-probe "${SMOACS_BLOCK_GRADIENT_PROBE:?}"
