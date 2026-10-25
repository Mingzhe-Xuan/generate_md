#!/bin/bash
#SBATCH --job-name=Basin16-probe
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/basin-probe-%j.out
#SBATCH --error=results/smoacs/logs/basin-probe-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_run="${BASIN_RUN_ROOT:?}"
if [[ ! -f "$task_run/plan.json" ]]; then python3 experiments/mattergen/basin_search/plan.py --output "$task_run"; fi
.venv-smoacs-blackwell/bin/python experiments/mattergen/basin_search/search.py --run "$task_run" --probe-only
