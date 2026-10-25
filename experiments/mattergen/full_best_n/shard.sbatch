#!/bin/bash
#SBATCH --job-name=BestN128-shards
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/bestN128-%A_%a.out
#SBATCH --error=results/smoacs/logs/bestN128-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
env FULL_BESTN_SHARD="${SLURM_ARRAY_TASK_ID:?}" bash experiments/mattergen/full_best_n/pipeline.sh
