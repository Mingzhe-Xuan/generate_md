#!/bin/bash
#SBATCH --job-name=ALIGNNgapMode
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/gap-mode-%j.out
#SBATCH --error=results/smoacs/logs/gap-mode-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mattergen/bin/python experiments/mattergen/alignn_reliability/distribution.py --run "${GAPMODE_ROOT:?}"
