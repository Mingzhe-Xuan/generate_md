#!/bin/bash
#SBATCH --job-name=mp-hull-novelty
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mp-assess-%j.out
#SBATCH --error=results/smoacs/logs/mp-assess-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py \
 --selection "${SMOACS_REFERENCE_SELECTION:?}"
