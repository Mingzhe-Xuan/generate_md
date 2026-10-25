#!/bin/bash
#SBATCH --job-name=BestN128-gate
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/bestN128-gate-%j.out
#SBATCH --error=results/smoacs/logs/bestN128-gate-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
: "${FULL_BESTN_ROOT:?}"
python3 experiments/mattergen/full_best_n/plan.py --run "$FULL_BESTN_ROOT"
env FULL_BESTN_SHARD=0 bash experiments/mattergen/full_best_n/pipeline.sh
