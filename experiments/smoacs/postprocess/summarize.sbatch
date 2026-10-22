#!/bin/bash
#SBATCH --job-name=cpu206-assess
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=results/smoacs/logs/post-assess-%j.out
#SBATCH --error=results/smoacs/logs/post-assess-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2
.venv-smoacs-postprocess/bin/python experiments/smoacs/postprocess/summarize.py \
 --inventory "${SMOACS_INVENTORY:?}" --root results/smoacs/postprocess-206
