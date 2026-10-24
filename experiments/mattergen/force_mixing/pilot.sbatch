#!/bin/bash
#SBATCH --job-name=MatterGen-force-mix-pilot
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/mattergen-force-mix-pilot-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-force-mix-pilot-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
task_root="${FORCE_MIX_PILOT_ROOT:-$PWD/results/mattergen/force-mix-pilot-$SLURM_JOB_ID}"
mkdir -p "$task_root"
touch "$task_root/SKIP_SUN"
for task_weight in 0.1 0.2 0.3; do
 task_probe=""
 if [[ "$task_weight" != 0.1 ]]; then
  task_probe="$task_root/lambda-0.1/physics-gradient-probe.json"
 fi
 env FORCE_MIX_RUN="$task_root/lambda-$task_weight" FORCE_MIX_WEIGHT="$task_weight" \
  FORCE_MIX_COHORT=pilot FORCE_MIX_PROBE="$task_probe" bash experiments/mattergen/force_mixing/pipeline.sh
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/force_mixing/report.py --run "$task_root" \
 --source results/mattergen/gap2p5-task128-20261024/full128 \
 --baseline results/mattergen/stepwise-354/block200 --pilot
git rev-parse HEAD > "$task_root/source-commit.txt"
tar -czf "${task_root}-evidence.tar.gz" --exclude=loader-tmp --exclude='capture-*' \
 --exclude='*.pth' --exclude='*.zip' --exclude='*.npz' --exclude='*.npy' -C "$task_root" .
sha256sum "${task_root}-evidence.tar.gz" > "${task_root}-evidence.tar.gz.sha256"
