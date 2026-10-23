#!/bin/bash
#SBATCH --job-name=relaxation-compare
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/compare-assess-%j.out
#SBATCH --error=results/smoacs/logs/compare-assess-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-dft/bin/python experiments/smoacs/comparison/evaluate.py \
 --inventory "${SMOACS_COMPARE_INVENTORY:?}" --selection "${SMOACS_REFERENCE_SELECTION:?}" \
 --singlepoint "${SMOACS_COMPARE_SINGLEPOINT:?}" --predictor "${SMOACS_COMPARE_PREDICTOR:?}" \
 --author-csv "${SMOACS_COMPARE_AUTHOR_CSV:?}" \
 --relaxation "${SMOACS_COMPARE_RELAXATION:?}" --benchmark "${SMOACS_COMPARE_BENCHMARK:?}" \
 --output "${SMOACS_COMPARE_ROOT:?}/assessment-${SLURM_JOB_ID}"
