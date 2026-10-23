#!/bin/bash
#SBATCH --job-name=openmx-MP-physics
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/openmx-physics-%j.out
#SBATCH --error=results/smoacs/logs/openmx-physics-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-dft/bin/python experiments/smoacs/openmx/physics.py \
 --output "results/smoacs/openmx-206/physics-${SLURM_JOB_ID}/manifest.json"
