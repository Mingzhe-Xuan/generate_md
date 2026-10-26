#!/bin/bash
#SBATCH --job-name=DirectedGrad-gate
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/directed-gate-%j.out
#SBATCH --error=results/smoacs/logs/directed-gate-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_run="${DIRECTED_RUN_ROOT:?}"
python3 experiments/mattergen/directed_tto/plan.py --output "$task_run"
.venv-smoacs-blackwell/bin/python experiments/mattergen/directed_tto/probe.py --run "$task_run"
