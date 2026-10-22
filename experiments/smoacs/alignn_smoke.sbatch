#!/bin/bash
#SBATCH --job-name=smoacs-alignn
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=results/smoacs/logs/alignn-%j.out
#SBATCH --error=results/smoacs/logs/alignn-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from parent repository}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export DGLBACKEND=pytorch
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
"$PWD/.venv-smoacs-blackwell/bin/python" experiments/smoacs/run_alignn_smoke.py \
 --source third_party/SMOACS --assets results/smoacs/assets-199 \
 --weights results/smoacs/alignn-weights
