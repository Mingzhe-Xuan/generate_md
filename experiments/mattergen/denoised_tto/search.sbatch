#!/bin/bash
#SBATCH --job-name=DenoisedTTO16
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/smoacs/logs/denoised-search-%j.out
#SBATCH --error=results/smoacs/logs/denoised-search-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_run="${DENOISED_ROOT:?}"
.venv-mattergen/bin/python experiments/mattergen/denoised_tto/denoised_search.py --source "${DENOISED_SOURCE:?}" --native "${DENOISED_NATIVE:?}" --run "$task_run"
for task_method in full reverse random stop direct seeds baseline; do
 env BESTN_EVALUATION="$task_run/evaluation-$task_method" bash experiments/mattergen/best_of_n/evaluate.sh
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/denoised_tto/denoised_report.py --run "$task_run"
tar -czf "$task_run-summary.tar.gz" -C "$task_run" plan.json native.json results.json complete.json report.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
