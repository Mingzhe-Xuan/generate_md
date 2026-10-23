#!/bin/bash
#SBATCH --job-name=openmx-production-gate
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=results/smoacs/logs/openmx-dispatch-%j.out
#SBATCH --error=results/smoacs/logs/openmx-dispatch-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
python3 experiments/smoacs/openmx/dispatch.py --selection "${SMOACS_REFERENCE_SELECTION:?}" \
 --assessment "${SMOACS_REFERENCE_ASSESSMENT:?}" --calibration "${SMOACS_OPENMX_CALIBRATION:?}"
