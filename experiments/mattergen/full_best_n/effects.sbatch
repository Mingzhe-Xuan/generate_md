#!/bin/bash
#SBATCH --job-name=NoiseCF-effects
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/smoacs/logs/noiseCF-effects-%j.out
#SBATCH --error=results/smoacs/logs/noiseCF-effects-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_run="$PWD/results/mattergen/noiseCF-effects-$SLURM_JOB_ID"
.venv-smoacs-postprocess/bin/python experiments/mattergen/full_best_n/effects.py \
 --source "${BESTN_EFFECT_SOURCE:?}" --output "$task_run" --workers 16
tar -czf "$task_run-summary.tar.gz" -C "$task_run" summary.json summary.md pairs.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
