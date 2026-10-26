#!/bin/bash
#SBATCH --job-name=DirectedGrad16
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/smoacs/logs/directed-%j.out
#SBATCH --error=results/smoacs/logs/directed-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_run="${DIRECTED_RUN_ROOT:?}"
.venv-smoacs-blackwell/bin/python experiments/mattergen/directed_tto/experiment.py --run "$task_run"
for task_arm in seed plain-descent plain-reverse plain-random soft-descent soft-reverse soft-random unrolled8-descent unrolled8-reverse unrolled8-random; do
 env BESTN_EVALUATION="$task_run/evaluation-$task_arm" bash experiments/mattergen/best_of_n/evaluate.sh
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/directed_tto/report.py --run "$task_run"
tar -czf "$task_run-summary.tar.gz" -C "$task_run" plan.json native-probe.json direction-results.json direction.complete.json report.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
