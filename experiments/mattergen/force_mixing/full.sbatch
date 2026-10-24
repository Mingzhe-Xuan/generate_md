#!/bin/bash
#SBATCH --job-name=MatterGen-force-mix-full
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mattergen-force-mix-full-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-force-mix-full-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
: "${FORCE_MIX_PROBE:?Native validated pilot proof required}"
task_root="${FORCE_MIX_FULL_ROOT:-$PWD/results/mattergen/force-mix-full-$SLURM_JOB_ID}"
mkdir -p "$task_root"
touch "$task_root/SKIP_SUN"
for task_weight in 0.1 0.2 0.3; do
 for task_cohort in neutral unmasked; do
  env FORCE_MIX_RUN="$task_root/lambda-$task_weight/$task_cohort" FORCE_MIX_WEIGHT="$task_weight" \
   FORCE_MIX_COHORT="$task_cohort" bash experiments/mattergen/force_mixing/pipeline.sh
 done
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/force_mixing/report.py --run "$task_root" \
 --source results/mattergen/gap2p5-task128-20261024/full128 \
 --baseline results/mattergen/alignn-full-361/cf200
git rev-parse HEAD > "$task_root/source-commit.txt"
tar -czf "${task_root}-evidence.tar.gz" --exclude=loader-tmp --exclude='capture-*' \
 --exclude='*.pth' --exclude='*.zip' --exclude='*.npz' --exclude='*.npy' -C "$task_root" .
sha256sum "${task_root}-evidence.tar.gz" > "${task_root}-evidence.tar.gz.sha256"
