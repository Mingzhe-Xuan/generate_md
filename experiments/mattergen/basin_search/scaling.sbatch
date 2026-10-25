#!/bin/bash
#SBATCH --job-name=Basin16-scaling
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/basin-scaling-%j.out
#SBATCH --error=results/smoacs/logs/basin-scaling-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_run_args=(--run "${BASIN_DISCOVERY_ROOT:?}")
if [[ -n "${BASIN_CONFIRM_ROOT:-}" ]]; then task_run_args+=(--run "$BASIN_CONFIRM_ROOT"); fi
.venv-smoacs-postprocess/bin/python experiments/mattergen/basin_search/scaling.py "${task_run_args[@]}" --output "$BASIN_DISCOVERY_ROOT/scaling-$SLURM_JOB_ID"
