#!/bin/bash
#SBATCH --job-name=cf200-stationarity
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/cf-stationarity-%j.out
#SBATCH --error=results/smoacs/logs/cf-stationarity-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
.venv-smoacs-paper/bin/python experiments/smoacs/convergence/stationarity.py \
 --npz results/smoacs/optimization/section21-Struct-AtomicMask-grad-206/result.npz \
 --config results/smoacs/optimization/section21-Struct-AtomicMask-grad-206/run_config.json \
 --assets results/smoacs/assets-199 \
 --output "results/smoacs/convergence-206/stationarity-${SLURM_JOB_ID}"
