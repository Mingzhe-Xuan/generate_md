#!/bin/bash
#SBATCH --job-name=FwdInvProperty
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/forward-inverse-property-%j.out
#SBATCH --error=results/smoacs/logs/forward-inverse-property-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 DGLBACKEND=pytorch
resume_args=()
if [[ -n "${FORWARD_INVERSE_RESUME:-}" ]]; then resume_args=(--resume-from "$FORWARD_INVERSE_RESUME"); fi
.venv-mattergen/bin/python experiments/mattergen/forward_inverse/property_pullback.py --run "${FORWARD_INVERSE_ROOT:?}" "${resume_args[@]}"
archive_args=( \
    "$FORWARD_INVERSE_ROOT/plan.json" "$FORWARD_INVERSE_ROOT/selection.json" \
    "$FORWARD_INVERSE_ROOT/complete.json" "$FORWARD_INVERSE_ROOT/physics" \
    "$FORWARD_INVERSE_ROOT/"*/result.json "$FORWARD_INVERSE_ROOT/"*/*/result.json )
if [[ -n "${FORWARD_INVERSE_RESUME:-}" ]]; then
    archive_args+=( "$FORWARD_INVERSE_RESUME/plan.json" "$FORWARD_INVERSE_RESUME/failure.json" \
        "$FORWARD_INVERSE_RESUME/selection.json" "$FORWARD_INVERSE_RESUME/progress.json" \
        "$FORWARD_INVERSE_RESUME/physics" "$FORWARD_INVERSE_RESUME/"*/result.json "$FORWARD_INVERSE_RESUME/"*/*/result.json )
fi
tar -czf "$FORWARD_INVERSE_ROOT/evidence.tar.gz" -- "${archive_args[@]}"
sha256sum "$FORWARD_INVERSE_ROOT/evidence.tar.gz" > "$FORWARD_INVERSE_ROOT/evidence.tar.gz.sha256"
