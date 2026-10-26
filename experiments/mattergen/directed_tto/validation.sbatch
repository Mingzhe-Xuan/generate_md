#!/bin/bash
#SBATCH --job-name=Directed48-validation
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/smoacs/logs/directed48-%j.out
#SBATCH --error=results/smoacs/logs/directed48-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_run="${VALIDATION_RUN_ROOT:?}"
if [[ ! -d "$task_run" ]]; then
 .venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/baseline_amendment.py --source "${VALIDATION_SOURCE_ROOT:?}" --run "$task_run"
fi
.venv-smoacs-blackwell/bin/python experiments/mattergen/directed_tto/validation_run.py --run "$task_run"
for task_arm in retained-descent retained-reverse retained-random ordinary-descent ordinary-reverse ordinary-random baseline raw-replay-baseline; do
 env BESTN_EVALUATION="$task_run/evaluation-$task_arm" bash experiments/mattergen/best_of_n/evaluate.sh
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/validation_report.py --run "$task_run"
tar -czf "$task_run-summary.tar.gz" -C "$task_run" validation-plan.json baseline-amendment.json validation-native.json validation-results.json validation.complete.json validation-report.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
