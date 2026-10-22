#!/bin/bash
#SBATCH --job-name=smoacs-status-check
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --time=00:01:00
#SBATCH --output=results/smoacs/logs/status-check-%j.out
#SBATCH --error=results/smoacs/logs/status-check-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the parent repository root}"
source experiments/smoacs/job_status.sh
true
