#!/bin/bash
#SBATCH --job-name=FwdInvAudit
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/smoacs/logs/forward-inverse-analysis-%j.out
#SBATCH --error=results/smoacs/logs/forward-inverse-analysis-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mattergen/bin/python experiments/mattergen/forward_inverse/analyze.py \
    --run "${FORWARD_INVERSE_ROOT:?}" --output "${FORWARD_INVERSE_ANALYSIS:?}"
tar -czf "$FORWARD_INVERSE_ANALYSIS/evidence.tar.gz" -- \
    "$FORWARD_INVERSE_ROOT" "$FORWARD_INVERSE_ANALYSIS/report.json" \
    "$FORWARD_INVERSE_ANALYSIS/curvature.png" "$FORWARD_INVERSE_ANALYSIS/curvature.svg"
sha256sum "$FORWARD_INVERSE_ANALYSIS/evidence.tar.gz" > "$FORWARD_INVERSE_ANALYSIS/evidence.tar.gz.sha256"
