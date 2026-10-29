#!/bin/bash
#SBATCH --job-name=FwdInvPropAudit
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/smoacs/logs/forward-inverse-property-report-%j.out
#SBATCH --error=results/smoacs/logs/forward-inverse-property-report-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mattergen/bin/python experiments/mattergen/forward_inverse/property_report.py \
    --run "${FORWARD_INVERSE_ROOT:?}" --output "${FORWARD_INVERSE_REPORT:?}"
tar -czf "$FORWARD_INVERSE_REPORT/evidence.tar.gz" -- \
    "$FORWARD_INVERSE_REPORT/report.json" "$FORWARD_INVERSE_REPORT/property-response.png" \
    "$FORWARD_INVERSE_REPORT/property-response.svg"
sha256sum "$FORWARD_INVERSE_REPORT/evidence.tar.gz" > "$FORWARD_INVERSE_REPORT/evidence.tar.gz.sha256"
