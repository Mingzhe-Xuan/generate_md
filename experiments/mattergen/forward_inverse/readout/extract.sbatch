#!/bin/bash
#SBATCH --job-name=FwdReadoutGPU
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/forward-readout-extract-%j.out
#SBATCH --error=results/smoacs/logs/forward-readout-extract-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 DGLBACKEND=pytorch
.venv-mattergen/bin/python experiments/mattergen/forward_inverse/readout/extract.py --run "${READOUT_ROOT:?}"
