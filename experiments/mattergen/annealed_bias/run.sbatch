#!/bin/bash
#SBATCH --job-name=AnnealedBiasTTO
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/smoacs/logs/annealed-run-%j.out
#SBATCH --error=results/smoacs/logs/annealed-run-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_run="${ANNEALED_ROOT:?}"
.venv-smoacs-blackwell/bin/python experiments/mattergen/annealed_bias/anneal_experiment.py --source "${ANNEALED_SOURCE:?}" --native "${ANNEALED_NATIVE:?}" --run "$task_run" --count "${ANNEALED_COUNT:-16}"
for task_method in full reverse random zero fixed seed baseline full_stage3 full_stage4 full_stage7; do
 env BESTN_EVALUATION="$task_run/evaluation-$task_method" bash experiments/mattergen/best_of_n/evaluate.sh
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/annealed_bias/anneal_report.py --run "$task_run"
tar -czf "$task_run-summary.tar.gz" -C "$task_run" plan.json native.json results.json complete.json report.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
