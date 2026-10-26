#!/bin/bash
#SBATCH --job-name=Anneal-retention
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=results/smoacs/logs/annealed-retention-%j.out
#SBATCH --error=results/smoacs/logs/annealed-retention-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/annealed_bias/retention/anneal_retention.py --mode physics --source "${ANNEAL_RETENTION_SOURCE:?}" --run "${ANNEAL_RETENTION_ROOT:?}"
task_run="${ANNEAL_RETENTION_ROOT:?}"
if [[ -f "$task_run/report.json" ]]; then
 tar -czf "$task_run-summary.tar.gz" -C "$task_run" retention-plan.json physics.json physics.complete.json report.json
 sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
fi
