#!/bin/bash
#SBATCH --job-name=BestN128-report
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/bestN128-report-%j.out
#SBATCH --error=results/smoacs/logs/bestN128-report-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-postprocess/bin/python experiments/mattergen/full_best_n/report.py --run "${FULL_BESTN_ROOT:?}" --workers 16
git rev-parse HEAD > "$FULL_BESTN_ROOT/report-source-commit.txt"
tar -czf "$FULL_BESTN_ROOT-summary.tar.gz" -C "$FULL_BESTN_ROOT" plan.json comparison.json comparison.md selection-records.json report-source-commit.txt
sha256sum "$FULL_BESTN_ROOT-summary.tar.gz" > "$FULL_BESTN_ROOT-summary.tar.gz.sha256"
