#!/bin/bash
#SBATCH --job-name=CSPtrajPrep
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/csp-traj-prepare-%j.out
#SBATCH --error=results/smoacs/logs/csp-traj-prepare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/csp_trajectory/prepare.py --run "${CSP_TRAJECTORY_ROOT:?}"
