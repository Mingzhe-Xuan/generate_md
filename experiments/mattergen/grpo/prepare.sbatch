#!/bin/bash
#SBATCH --job-name=GRPOcohort
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/grpo-cohort-%j.out
#SBATCH --error=results/smoacs/logs/grpo-cohort-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mattergen/bin/python experiments/mattergen/grpo/prepare.py --run "${GRPO_ROOT:?}/data"
