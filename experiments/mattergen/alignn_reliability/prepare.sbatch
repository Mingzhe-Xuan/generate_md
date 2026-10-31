#!/bin/bash
#SBATCH --job-name=ReliabilityData
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/reliability-data-%j.out
#SBATCH --error=results/smoacs/logs/reliability-data-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mattergen/bin/python experiments/mattergen/alignn_reliability/prepare.py --run "${RELIABILITY_ROOT:?}"
