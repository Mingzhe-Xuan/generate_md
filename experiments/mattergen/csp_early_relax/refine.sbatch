#!/bin/bash
#SBATCH --job-name=CSPearlyTight
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/csp-early-tight-%j.out
#SBATCH --error=results/smoacs/logs/csp-early-tight-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-mattergen/bin/python experiments/mattergen/csp_early_relax/refine.py --source "${CSP_EARLY_COARSE:?}" --run "${CSP_EARLY_ROOT:?}"
