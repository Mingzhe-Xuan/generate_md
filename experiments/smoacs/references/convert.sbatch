#!/bin/bash
#SBATCH --job-name=mp-ref-convert
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/mp-convert-%j.out
#SBATCH --error=results/smoacs/logs/mp-convert-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2
.venv-smoacs-postprocess/bin/python experiments/smoacs/references/convert.py \
 --selection "${SMOACS_REFERENCE_SELECTION:?}"
