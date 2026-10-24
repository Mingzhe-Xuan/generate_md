#!/bin/bash
#SBATCH --job-name=MatterGen-stepwise-pilot
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mattergen-stepwise-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-stepwise-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export MATTERGEN_CF_COHORT=stepwise_pilot
task_root="${MATTERGEN_STEPWISE_ROOT:-$PWD/results/mattergen/stepwise-$SLURM_JOB_ID}"
task_source="${MATTERGEN_TEMPLATE_SOURCE:-$PWD/results/mattergen/gap2p5-task128-20261024/full128}"
mkdir -p "$task_root"
touch "$task_root/SKIP_SUN"
for task_arm in block200 stepwise200; do
 task_args=()
 task_cycle=1
 if [[ "$task_arm" == stepwise200 ]]; then
  task_args+=(--relax-every-step)
  task_cycle=200
 fi
 if [[ ! -f "$task_root/$task_arm/complete.json" ]]; then
  .venv-smoacs-blackwell/bin/python experiments/mattergen/cf_templates/run.py \
   --source "$task_source" --output "$task_root/$task_arm" --cohort pilot \
   --steps 200 --fixed-species "${task_args[@]}"
 fi
 env MATTERGEN_STEPWISE_RUN="$task_root/$task_arm" MATTERGEN_STEPWISE_CYCLE="$task_cycle" \
  bash experiments/mattergen/stepwise/evaluate.sh
done
python3 experiments/mattergen/stepwise/report.py --run "$task_root" --source "$task_source"
git rev-parse HEAD > "$task_root/source-commit.txt"
sha256sum "$task_root/block200/checkpoint.pt" "$task_root/stepwise200/checkpoint.pt" > "$task_root/checkpoint-sha256.txt"
tar -czf "${task_root}-evidence.tar.gz" --exclude=loader-tmp --exclude='capture-*' \
 --exclude='*.pth' --exclude='*.zip' --exclude='*.npz' --exclude='*.npy' -C "$task_root" .
sha256sum "${task_root}-evidence.tar.gz" > "${task_root}-evidence.tar.gz.sha256"
