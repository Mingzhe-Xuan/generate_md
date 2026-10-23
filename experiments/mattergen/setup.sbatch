#!/bin/bash
#SBATCH --job-name=MatterGen-setup
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/mattergen-setup-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-setup-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export ALL_PROXY=socks5h://127.0.0.1:1080 HTTP_PROXY=socks5h://127.0.0.1:1080 HTTPS_PROXY=socks5h://127.0.0.1:1080
curl --fail --proxy "$ALL_PROXY" --head --connect-timeout 10 --max-time 30 https://github.com/ >/dev/null
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-smoacs-blackwell/bin/python experiments/mattergen/setup.py
