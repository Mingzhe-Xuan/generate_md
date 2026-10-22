#!/bin/bash
#SBATCH --job-name=smoacs-env-probe
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=results/smoacs/logs/probe-%j.out
#SBATCH --error=results/smoacs/logs/probe-%j.err
set -euo pipefail
task_root="${SLURM_SUBMIT_DIR:?Submit from the parent repository root}"
cd "$task_root"
source experiments/smoacs/job_status.sh
task_python="${SMOACS_PYTHON:-$task_root/.venv-smoacs-paper/bin/python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export DGLBACKEND=pytorch
"$task_python" experiments/smoacs/env_probe.py \
  --source third_party/SMOACS --device "${SMOACS_DEVICE:-cuda}" \
  --output "results/smoacs/env/probe-${SLURM_JOB_ID}.json"
