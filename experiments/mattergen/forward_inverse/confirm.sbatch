#!/bin/bash
#SBATCH --job-name=FwdInvConfirm
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/forward-inverse-confirm-%j.out
#SBATCH --error=results/smoacs/logs/forward-inverse-confirm-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-mattergen/bin/python experiments/mattergen/forward_inverse/confirm_curl.py \
    --source "${FORWARD_INVERSE_SOURCE:?}" --run "${FORWARD_INVERSE_CONFIRM:?}"
tar -czf "$FORWARD_INVERSE_CONFIRM/evidence.tar.gz" -- \
    "$FORWARD_INVERSE_CONFIRM/plan.json" "$FORWARD_INVERSE_CONFIRM/complete.json" \
    "$FORWARD_INVERSE_CONFIRM/physics" "$FORWARD_INVERSE_CONFIRM/"*/*.npz
sha256sum "$FORWARD_INVERSE_CONFIRM/evidence.tar.gz" > "$FORWARD_INVERSE_CONFIRM/evidence.tar.gz.sha256"
