#!/bin/bash
#SBATCH --job-name=QuenchGuided16
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/smoacs/logs/quench-guided-%j.out
#SBATCH --error=results/smoacs/logs/quench-guided-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_run="${QUENCH_GUIDED_ROOT:?}"
.venv-smoacs-blackwell/bin/python experiments/mattergen/quench_guided/guided_run.py --source "${QUENCH_GUIDED_SOURCE:?}" --run "$task_run"
for task_family in cf_guided random_guided isotropic; do
 for task_arm in descent reverse random raw_cf probes; do
  env BESTN_EVALUATION="$task_run/evaluation-$task_family-$task_arm" bash experiments/mattergen/best_of_n/evaluate.sh
 done
done
env BESTN_EVALUATION="$task_run/evaluation-baseline" bash experiments/mattergen/best_of_n/evaluate.sh
.venv-smoacs-postprocess/bin/python experiments/mattergen/quench_guided/guided_report.py --run "$task_run"
tar -czf "$task_run-summary.tar.gz" -C "$task_run" plan.json native.json results.json complete.json report.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
