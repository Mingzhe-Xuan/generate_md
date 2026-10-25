#!/bin/bash
#SBATCH --job-name=Basin16-interim
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/basin-interim-%j.out
#SBATCH --error=results/smoacs/logs/basin-interim-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/basin_search/interim.py --run "${BASIN_DISCOVERY_ROOT:?}"
