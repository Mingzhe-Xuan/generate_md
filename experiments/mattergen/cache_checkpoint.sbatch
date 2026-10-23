#!/bin/bash
#SBATCH --job-name=MatterGen-weights
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/mattergen-weights-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-weights-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
python3 experiments/mattergen/cache_wheels.py --checkpoint
