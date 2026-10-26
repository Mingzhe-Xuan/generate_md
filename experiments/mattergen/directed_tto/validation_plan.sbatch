#!/bin/bash
#SBATCH --job-name=Directed48-plan
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=results/smoacs/logs/directed48-plan-%j.out
#SBATCH --error=results/smoacs/logs/directed48-plan-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/validation_plan.py --pilot "${VALIDATION_PILOT_ROOT:?}" --run "${VALIDATION_RUN_ROOT:?}"
