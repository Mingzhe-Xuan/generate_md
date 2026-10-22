#!/bin/bash
#SBATCH --job-name=smoacs-assets
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=results/smoacs/logs/prepare-%j.out
#SBATCH --error=results/smoacs/logs/prepare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the parent repository root}"
python3 experiments/smoacs/prepare_assets.py \
  --downloads results/smoacs/downloads \
  --output "results/smoacs/assets-${SLURM_JOB_ID}"
