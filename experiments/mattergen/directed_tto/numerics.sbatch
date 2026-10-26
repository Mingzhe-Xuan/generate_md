#!/bin/bash
#SBATCH --job-name=Directed-numerics
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=results/smoacs/logs/directed-numerics-%j.out
#SBATCH --error=results/smoacs/logs/directed-numerics-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
.venv-smoacs-blackwell/bin/python experiments/mattergen/directed_tto/numerics.py --run "${DIRECTED_RUN_ROOT:?}"
