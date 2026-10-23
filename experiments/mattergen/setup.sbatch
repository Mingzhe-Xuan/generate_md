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
if curl --fail --proxy "$ALL_PROXY" --head --connect-timeout 10 --max-time 30 https://github.com/ >/dev/null; then
 export MATTERGEN_NETWORK_ROUTE=verified_socks
else
 curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 https://pypi.org/simple/ >/dev/null
 curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 https://data.pyg.org/whl/torch-2.7.0+cu128.html >/dev/null
 curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 \
 https://media.githubusercontent.com/media/microsoft/mattergen/92423660a8bd70e83679086e88f88596d484dc16/checkpoints/dft_band_gap/checkpoints/last.ckpt >/dev/null
 unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
 export MATTERGEN_NETWORK_ROUTE=verified_direct
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-smoacs-blackwell/bin/python experiments/mattergen/setup.py
