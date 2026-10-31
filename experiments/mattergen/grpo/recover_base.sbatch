#!/bin/bash
#SBATCH --job-name=GRPObaseAudit
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/grpo-base-%j.out
#SBATCH --error=results/smoacs/logs/grpo-base-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mattergen/bin/python experiments/mattergen/grpo/recover_base.py --run "${GRPO_ROOT:?}"
