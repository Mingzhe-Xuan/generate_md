#!/bin/bash
#SBATCH --job-name=cpu206-dft-ready
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/post-dft-ready-%j.out
#SBATCH --error=results/smoacs/logs/post-dft-ready-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2
.venv-smoacs-dft/bin/python experiments/smoacs/postprocess/dft_prepare.py \
 --inventory "${SMOACS_INVENTORY:?}" --output results/smoacs/postprocess-206/dft-config
