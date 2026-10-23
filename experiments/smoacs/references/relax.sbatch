#!/bin/bash
#SBATCH --job-name=mp-reference-relax
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mp-relax-%j.out
#SBATCH --error=results/smoacs/logs/mp-relax-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
task_extra=()
if [[ -n "${SMOACS_REFERENCE_LIMIT:-}" ]]; then task_extra+=(--limit "$SMOACS_REFERENCE_LIMIT"); fi
if [[ -n "${SMOACS_REFERENCE_IDS:-}" ]]; then
 read -ra task_ids <<< "$SMOACS_REFERENCE_IDS"
 task_extra+=(--candidate-id "${task_ids[@]}")
fi
if [[ "${SMOACS_REQUIRE_CONVERGED:-0}" == 1 ]]; then task_extra+=(--require-converged); fi
.venv-smoacs-postprocess/bin/python experiments/smoacs/postprocess/relax.py \
 --inventory "${SMOACS_REFERENCE_SELECTION:?}/mattersim-inventory.json" \
 --output "${SMOACS_REFERENCE_SELECTION:?}/mattersim-relaxation" \
 --checkpoint results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth "${task_extra[@]}"
