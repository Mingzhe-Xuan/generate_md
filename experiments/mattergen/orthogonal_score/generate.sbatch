#!/bin/bash
#SBATCH --job-name=OrthogonalScore
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/orthogonal-score-%j.out
#SBATCH --error=results/smoacs/logs/orthogonal-score-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
.venv-mattergen/bin/python experiments/mattergen/orthogonal_score/run_experiment.py --run "${ORTHOGONAL_ROOT:?}"
