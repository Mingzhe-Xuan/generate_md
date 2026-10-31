#!/bin/bash
#SBATCH --job-name=GRPOcheckpoint
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=results/smoacs/logs/grpo-checkpoint-%j.out
#SBATCH --error=results/smoacs/logs/grpo-checkpoint-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mattergen/bin/python -m unittest discover -s experiments/mattergen/grpo -p test_checkpoint_io.py
