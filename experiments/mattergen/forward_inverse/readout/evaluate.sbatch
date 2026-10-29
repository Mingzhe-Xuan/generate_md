#!/bin/bash
#SBATCH --job-name=FwdReadoutAudit
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/forward-readout-evaluate-%j.out
#SBATCH --error=results/smoacs/logs/forward-readout-evaluate-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mattergen/bin/python experiments/mattergen/forward_inverse/readout/evaluate.py --run "${READOUT_ROOT:?}"
tar -czf "$READOUT_ROOT/evidence.tar.gz" -- \
    "$READOUT_ROOT/selection.json" "$READOUT_ROOT/feature-plan.json" "$READOUT_ROOT/feature-complete.json" \
    "$READOUT_ROOT/features.npz" "$READOUT_ROOT/parents" "$READOUT_ROOT/evaluation"
sha256sum "$READOUT_ROOT/evidence.tar.gz" > "$READOUT_ROOT/evidence.tar.gz.sha256"
