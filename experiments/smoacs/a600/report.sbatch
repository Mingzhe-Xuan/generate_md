#!/bin/bash
#SBATCH --job-name=A600-AB-compare
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=results/smoacs/logs/a600-compare-%j.out
#SBATCH --error=results/smoacs/logs/a600-compare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
python3 experiments/smoacs/a600/compare.py --root "${SMOACS_A600_ROOT:?}"
