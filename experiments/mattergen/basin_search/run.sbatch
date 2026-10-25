#!/bin/bash
#SBATCH --job-name=BasinSearch16
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/smoacs/logs/basin-search-%j.out
#SBATCH --error=results/smoacs/logs/basin-search-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_run="${BASIN_RUN_ROOT:-$PWD/results/mattergen/basin-search-$SLURM_JOB_ID}"
if [[ ! -f "$task_run/plan.json" ]]; then
 task_cohort_args=()
 if [[ -n "${BASIN_FROZEN_PLAN:-}" ]]; then task_cohort_args+=(--frozen-plan "$BASIN_FROZEN_PLAN"); fi
 python3 experiments/mattergen/basin_search/plan.py --output "$task_run" --search-seed "${BASIN_SEARCH_SEED:-20261026}" "${task_cohort_args[@]}"
fi
.venv-smoacs-blackwell/bin/python experiments/mattergen/basin_search/search.py --run "$task_run" --seed "${BASIN_SEARCH_SEED:-20261026}"
for task_method in iid beam soft_md symmetry_release; do
 env BESTN_EVALUATION="$task_run/evaluation-$task_method" bash experiments/mattergen/best_of_n/evaluate.sh
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/basin_search/report.py --run "$task_run"
tar -czf "$task_run-summary.tar.gz" -C "$task_run" plan.json search-results.json native-controls.json mechanics-metadata.json report.json report.md
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
