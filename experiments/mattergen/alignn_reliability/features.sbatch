#!/bin/bash
#SBATCH --job-name=ReliabilityFeat
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=results/smoacs/logs/reliability-features-%j.out
#SBATCH --error=results/smoacs/logs/reliability-features-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-mattergen/bin/python experiments/mattergen/alignn_reliability/features.py --run "${RELIABILITY_ROOT:?}"
