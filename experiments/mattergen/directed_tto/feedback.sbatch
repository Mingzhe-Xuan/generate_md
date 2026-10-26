#!/bin/bash
#SBATCH --job-name=Directed-feedback16
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/directed-feedback-%j.out
#SBATCH --error=results/smoacs/logs/directed-feedback-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_run="${FEEDBACK_RUN_ROOT:?}"
.venv-smoacs-blackwell/bin/python experiments/mattergen/directed_tto/feedback.py --source "${FEEDBACK_SOURCE_ROOT:?}" --run "$task_run"
for task_arm in descent reverse random; do
 env BESTN_EVALUATION="$task_run/evaluation-$task_arm" bash experiments/mattergen/best_of_n/evaluate.sh
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/feedback_report.py --run "$task_run"
tar -czf "$task_run-summary.tar.gz" -C "$task_run" feedback-plan.json feedback-native-controls.json feedback-results.json feedback.complete.json feedback-report.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
