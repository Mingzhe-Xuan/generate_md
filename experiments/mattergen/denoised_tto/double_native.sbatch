#!/bin/bash
#SBATCH --job-name=DenoisedDoubleNative
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=results/smoacs/logs/denoised-double-native-%j.out
#SBATCH --error=results/smoacs/logs/denoised-double-native-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
.venv-mattergen/bin/python experiments/mattergen/denoised_tto/double_native.py --source "${DENOISED_SOURCE:?}" --run "${DENOISED_ROOT:?}"
