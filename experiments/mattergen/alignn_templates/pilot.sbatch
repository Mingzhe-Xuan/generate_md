#!/bin/bash
#SBATCH --job-name=MatterGen-ALIGNN200-pilot
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/mattergen-alignn-pilot-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-alignn-pilot-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
task_root="$PWD/results/mattergen/alignn-pilot-$SLURM_JOB_ID"
mkdir -p "$task_root"
touch "$task_root/SKIP_SUN"
env ALIGNN_TEMPLATE_RUN="$task_root/alignn200" ALIGNN_TEMPLATE_COHORT=pilot \
 bash experiments/mattergen/alignn_templates/pipeline.sh
.venv-smoacs-postprocess/bin/python experiments/mattergen/alignn_templates/report.py \
 --run "$task_root" --source results/mattergen/gap2p5-task128-20261024/full128 --pilot
git rev-parse HEAD > "$task_root/source-commit.txt"
tar -czf "${task_root}-evidence.tar.gz" --exclude=loader-tmp --exclude='capture-*' \
 --exclude='*.pth' --exclude='*.zip' --exclude='*.npz' --exclude='*.npy' -C "$task_root" .
sha256sum "${task_root}-evidence.tar.gz" > "${task_root}-evidence.tar.gz.sha256"
