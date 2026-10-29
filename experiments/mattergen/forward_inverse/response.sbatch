#!/bin/bash
#SBATCH --job-name=FwdInvResponse
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/forward-inverse-response-%j.out
#SBATCH --error=results/smoacs/logs/forward-inverse-response-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-mattergen/bin/python experiments/mattergen/forward_inverse/precision_response.py \
    --run "${FORWARD_INVERSE_ROOT:?}" --output "${FORWARD_INVERSE_RESPONSE:?}"
tar -czf "$FORWARD_INVERSE_RESPONSE/evidence.tar.gz" -- \
    "$FORWARD_INVERSE_RESPONSE/complete.json" "$FORWARD_INVERSE_RESPONSE/"*.npz
sha256sum "$FORWARD_INVERSE_RESPONSE/evidence.tar.gz" > "$FORWARD_INVERSE_RESPONSE/evidence.tar.gz.sha256"
