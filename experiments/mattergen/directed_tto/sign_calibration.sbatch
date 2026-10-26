#!/bin/bash
#SBATCH --job-name=Directed-sign-audit
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=results/smoacs/logs/directed-sign-%j.out
#SBATCH --error=results/smoacs/logs/directed-sign-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/relaxation_gradient.py --source "${DIRECTED_SOURCE_ROOT:?}"
