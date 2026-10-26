#!/bin/bash
#SBATCH --job-name=Guided-retention
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=results/smoacs/logs/guided-retention-%j.out
#SBATCH --error=results/smoacs/logs/guided-retention-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/quench_guided/guided_retention.py --mode physics --source "${GUIDED_RETENTION_SOURCE:?}" --run "${GUIDED_RETENTION_ROOT:?}"
