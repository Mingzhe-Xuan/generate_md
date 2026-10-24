#!/bin/bash
#SBATCH --job-name=MatterGen-amplitude-sweep
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/mattergen-sweep-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-sweep-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
task_sweep="$PWD/results/mattergen/amplitude-sweep-$SLURM_JOB_ID"
mkdir -p "$task_sweep"
git rev-parse HEAD > "$task_sweep/source-commit.txt"
for task_amplitude in 0.10 0.20 0.30; do
 env MATTERGEN_BESTN_AMPLITUDE="$task_amplitude" MATTERGEN_BESTN_PROPOSAL_CAP=0.60 \
  MATTERGEN_BESTN_RUN="$task_sweep/rms-$task_amplitude" \
  bash experiments/mattergen/best_of_n/pilot.sbatch
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/best_of_n/sweep_report.py --run "$task_sweep"
tar -czf "${task_sweep}-evidence.tar.gz" --exclude='*-evidence.tar.gz' --exclude='*-evidence.tar.gz.sha256' \
 --exclude=loader-tmp --exclude='capture-*' --exclude='*.pth' --exclude='*.zip' --exclude='*.npz' --exclude='*.npy' \
 -C "$task_sweep" .
sha256sum "${task_sweep}-evidence.tar.gz" > "${task_sweep}-evidence.tar.gz.sha256"
