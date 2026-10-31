#!/bin/bash
#SBATCH --job-name=ReliabilityHeads
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=results/smoacs/logs/reliability-heads-%j.out
#SBATCH --error=results/smoacs/logs/reliability-heads-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-mattergen/bin/python experiments/mattergen/alignn_reliability/train.py --run "${RELIABILITY_ROOT:?}"
.venv-mattergen/bin/python experiments/mattergen/alignn_reliability/report.py --run "${RELIABILITY_ROOT:?}"
tar -czf "${RELIABILITY_ROOT}-evidence.tar.gz" -C "${RELIABILITY_ROOT}" .
sha256sum "${RELIABILITY_ROOT}-evidence.tar.gz"
