#!/bin/bash
#SBATCH --job-name=Directed-tight-CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/directed-tight-%j.out
#SBATCH --error=results/smoacs/logs/directed-tight-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/tight_relaxation.py --source "${TIGHT_SOURCE_ROOT:?}" --run "${TIGHT_RUN_ROOT:?}"
