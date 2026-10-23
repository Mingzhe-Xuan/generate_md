#!/bin/bash
#SBATCH --job-name=MatterGen-cache
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/mattergen-cache-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-cache-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
python3 experiments/mattergen/cache_wheels.py
