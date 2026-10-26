#!/bin/bash
#SBATCH --job-name=CF-dimer-native
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=results/smoacs/logs/dimer-native-%j.out
#SBATCH --error=results/smoacs/logs/dimer-native-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/dimer_tto/dimer_native.py --source "${DIMER_SOURCE:?}" --cache "${DIMER_CACHE:?}" --run "${DIMER_ROOT:?}"
tar -czf "${DIMER_ROOT:?}-summary.tar.gz" -C "$DIMER_ROOT" plan.json native.json results.json complete.json
sha256sum "$DIMER_ROOT-summary.tar.gz" > "$DIMER_ROOT-summary.tar.gz.sha256"
