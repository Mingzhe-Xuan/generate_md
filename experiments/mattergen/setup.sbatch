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
if [[ "${MATTERGEN_FORCE_DIRECT:-0}" != 1 ]] && curl --fail --proxy "$ALL_PROXY" --head --connect-timeout 10 --max-time 30 https://github.com/ >/dev/null; then
 export MATTERGEN_NETWORK_ROUTE=verified_socks
else
 curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 https://pypi.org/simple/ >/dev/null
 for task_wheel in torch_scatter-2.1.2+pt27cu128 torch_sparse-0.6.18+pt27cu128 torch_cluster-1.6.3+pt27cu128; do
  if [[ ! -f "results/mattergen/wheels/$task_wheel-cp310-cp310-linux_x86_64.whl" ]]; then
   curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 https://data.pyg.org/whl/torch-2.7.0+cu128.html >/dev/null
   break
  fi
 done
 if [[ ! -f results/mattergen/checkpoints/dft_band_gap/checkpoints/last.ckpt ]]; then
  curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 \
  https://media.githubusercontent.com/media/microsoft/mattergen/92423660a8bd70e83679086e88f88596d484dc16/checkpoints/dft_band_gap/checkpoints/last.ckpt >/dev/null
 fi
 unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
 export MATTERGEN_NETWORK_ROUTE=verified_direct
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
.venv-smoacs-blackwell/bin/python experiments/mattergen/setup.py
