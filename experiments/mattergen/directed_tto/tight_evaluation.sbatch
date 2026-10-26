#!/bin/bash
#SBATCH --job-name=Directed-tight-score
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/directed-tight-score-%j.out
#SBATCH --error=results/smoacs/logs/directed-tight-score-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_run="${TIGHT_RUN_ROOT:?}"
.venv-smoacs-blackwell/bin/python experiments/mattergen/directed_tto/tight_evaluation.py --run "$task_run" --mode prepare
env BESTN_EVALUATION="$task_run/evaluation-tight" bash experiments/mattergen/best_of_n/evaluate.sh
.venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/tight_evaluation.py --run "$task_run" --mode report
tar -czf "$task_run-summary.tar.gz" -C "$task_run" tight-plan.json tight-controls.json tight-physics.json tight-physics.complete.json tight-CF-controls.json tight-report.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
