#!/bin/bash
#SBATCH --job-name=Directed48-Bcheck
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/directed48-baseline-%j.out
#SBATCH --error=results/smoacs/logs/directed48-baseline-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/baseline_diagnostic.py --run "${VALIDATION_RUN_ROOT:?}"
