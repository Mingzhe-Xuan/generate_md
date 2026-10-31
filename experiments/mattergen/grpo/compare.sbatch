#!/bin/bash
#SBATCH --job-name=GRPOcompare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/smoacs/logs/grpo-compare-%j.out
#SBATCH --error=results/smoacs/logs/grpo-compare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-mattergen/bin/python experiments/mattergen/grpo/compare.py --run "${GRPO_ROOT:?}"
