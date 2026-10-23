#!/bin/bash
#SBATCH --job-name=mp-reference-parallel
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mp-parallel-%j.out
#SBATCH --error=results/smoacs/logs/mp-parallel-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_extra=()
if [[ "${SMOACS_REFERENCE_BENCHMARK:-0}" == 1 ]]; then task_extra+=(--benchmark); fi
if [[ -n "${SMOACS_REFERENCE_BENCHMARK_DIR:-}" ]]; then
 task_extra+=(--validated-benchmark "$SMOACS_REFERENCE_BENCHMARK_DIR")
fi
.venv-smoacs-postprocess/bin/python experiments/smoacs/references/parallel.py \
 --selection "${SMOACS_REFERENCE_SELECTION:?}" --workers 4 "${task_extra[@]}"
