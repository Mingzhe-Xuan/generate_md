#!/bin/bash
#SBATCH --job-name=GapTrajectory
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/gap-trajectory-%j.out
#SBATCH --error=results/smoacs/logs/gap-trajectory-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-mattergen/bin/python experiments/mattergen/conditional_trajectory/run.py --run "${GAP_TRAJECTORY_ROOT:?}"
.venv-mattergen/bin/python experiments/mattergen/conditional_trajectory/evaluate.py --run "${GAP_TRAJECTORY_ROOT:?}"
