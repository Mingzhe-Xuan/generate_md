#!/bin/bash
#SBATCH --job-name=Anneal-ret-score
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=results/smoacs/logs/annealed-ret-score-%j.out
#SBATCH --error=results/smoacs/logs/annealed-ret-score-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_run="${ANNEAL_RETENTION_ROOT:?}"
.venv-smoacs-blackwell/bin/python experiments/mattergen/annealed_bias/retention/anneal_retention.py --mode prepare --run "$task_run"
if [[ -d "$task_run/evaluation-retention" ]]; then
 env BESTN_EVALUATION="$task_run/evaluation-retention" bash experiments/mattergen/best_of_n/evaluate.sh
fi
.venv-smoacs-postprocess/bin/python experiments/mattergen/annealed_bias/retention/anneal_retention.py --mode report --run "$task_run"
tar -czf "$task_run-summary.tar.gz" -C "$task_run" retention-plan.json physics.json physics.complete.json report.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
