#!/bin/bash
#SBATCH --job-name=OrthogonalMPRelax
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=results/smoacs/logs/orthogonal-refs-%j.out
#SBATCH --error=results/smoacs/logs/orthogonal-refs-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/smoacs/postprocess/relax.py --inventory "${ORTHOGONAL_ROOT:?}/references/mattersim-inventory.json" --output "$ORTHOGONAL_ROOT/references/mattersim-relaxation" --checkpoint results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth
