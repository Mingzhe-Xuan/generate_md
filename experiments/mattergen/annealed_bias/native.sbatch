#!/bin/bash
#SBATCH --job-name=AnnealedBiasNative
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=results/smoacs/logs/annealed-native-%j.out
#SBATCH --error=results/smoacs/logs/annealed-native-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
.venv-smoacs-blackwell/bin/python experiments/mattergen/annealed_bias/anneal_native.py --source "${ANNEALED_SOURCE:?}" --run "${ANNEALED_ROOT:?}"
