#!/bin/bash
#SBATCH --job-name=MatterGen-CF-templates
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/mattergen-CF-templates-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-CF-templates-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
bash experiments/mattergen/cf_templates/pipeline.sh
