#!/bin/bash
#SBATCH --job-name=alternating-AB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/alternating-%j.out
#SBATCH --error=results/smoacs/logs/alternating-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_args=()
if [[ "${SMOACS_AB_PILOT:-0}" == 1 ]]; then
 task_args+=(--pilot --anneal 2 --max-rounds 16 --workers 2)
fi
.venv-smoacs-paper/bin/python experiments/smoacs/alternating/run.py \
 --output "${SMOACS_AB_OUTPUT:-results/smoacs/alternating-AB-${SLURM_JOB_ID}}" "${task_args[@]}"
