#!/bin/bash
#SBATCH --job-name=FwdInvSelect
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/forward-inverse-select-%j.out
#SBATCH --error=results/smoacs/logs/forward-inverse-select-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mattergen/bin/python experiments/mattergen/forward_inverse/prepare_anchors.py --run "${FORWARD_INVERSE_ROOT:?}" --systems "${FORWARD_INVERSE_SYSTEMS:-Si,C,TiO2,CaTiO3}" --per-system "${FORWARD_INVERSE_PER_SYSTEM:-1}"
