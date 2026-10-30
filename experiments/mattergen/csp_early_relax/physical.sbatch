#!/bin/bash
#SBATCH --job-name=CSPphysical
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/csp-early-physical-%j.out
#SBATCH --error=results/smoacs/logs/csp-early-physical-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/csp_early_relax/physical.py --run "${CSP_EARLY_ROOT:?}"
