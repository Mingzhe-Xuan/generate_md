#!/bin/bash
#SBATCH --job-name=mp-public-refs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mp-refs-%j.out
#SBATCH --error=results/smoacs/logs/mp-refs-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2
export SMOACS_REFERENCE_PROXY=socks5h://127.0.0.1:1080
curl --proxy "$SMOACS_REFERENCE_PROXY" --head --fail --connect-timeout 10 --max-time 25 \
 https://materialsproject-build.s3.amazonaws.com/ > "results/smoacs/logs/mp-network-${SLURM_JOB_ID}.txt"
.venv-smoacs-reference/bin/python experiments/smoacs/references/prepare.py
