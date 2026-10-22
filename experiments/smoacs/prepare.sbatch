#!/bin/bash
#SBATCH --job-name=smoacs-assets
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=results/smoacs/logs/prepare-%j.out
#SBATCH --error=results/smoacs/logs/prepare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the parent repository root}"
source experiments/smoacs/job_status.sh
task_extra=()
if [[ "${SMOACS_DIRECT_CHECKPOINTS:-0}" == 1 ]]; then
  task_extra+=(--direct-checkpoints)
fi
python3 experiments/smoacs/prepare_assets.py \
  --downloads results/smoacs/downloads \
  --output "results/smoacs/assets-${SLURM_JOB_ID}" "${task_extra[@]}"
