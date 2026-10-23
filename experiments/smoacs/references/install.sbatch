#!/bin/bash
#SBATCH --job-name=mp-ref-install
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/mp-install-%j.out
#SBATCH --error=results/smoacs/logs/mp-install-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
task_proxy=socks5h://127.0.0.1:1080
curl --proxy "$task_proxy" --head --fail --connect-timeout 10 --max-time 25 https://pypi.org/simple/
task_cache=results/smoacs/mp-public-reference/downloads
mkdir -p "$task_cache"
task_socks="$task_cache/PySocks-1.7.1-py3-none-any.whl"
curl --proxy "$task_proxy" --fail --location --connect-timeout 10 --max-time 90 \
 https://files.pythonhosted.org/packages/8d/59/b4572118e098ac8e46e399a1dd0f2d85403ce8bbaad9ec79373ed6badaf9/PySocks-1.7.1-py3-none-any.whl \
 --output "$task_socks"
echo "2725bd0a9925919b9b51739eea5f9e2bae91e83288108a9ad338b2e3a4435ee5  $task_socks" | sha256sum -c -
.venv-smoacs-reference/bin/python -m pip install --no-index "$task_socks"
.venv-smoacs-reference/bin/python -m pip install --proxy "$task_proxy" --only-binary=:all: \
 --index-url https://pypi.org/simple -r experiments/smoacs/references/requirements.txt
.venv-smoacs-reference/bin/python -m pip check
.venv-smoacs-reference/bin/python -m pip freeze > "results/smoacs/mp-public-reference/freeze-${SLURM_JOB_ID}.txt"
