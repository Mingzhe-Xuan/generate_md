#!/bin/bash
#SBATCH --job-name=Directed48-retention
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=results/smoacs/logs/directed48-retention-%j.out
#SBATCH --error=results/smoacs/logs/directed48-retention-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/validation_retention.py --mode physics --source "${RETENTION_SOURCE_ROOT:?}" --run "${RETENTION_RUN_ROOT:?}"
