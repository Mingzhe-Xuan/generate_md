#!/bin/bash
#SBATCH --job-name=smoacs-blackwell-install
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/blackwell-install-%j.out
#SBATCH --error=results/smoacs/logs/blackwell-install-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source experiments/smoacs/job_status.sh
task_python="$PWD/.venv-smoacs-blackwell/bin/python"
test -x "$task_python"
"$PWD/.venv-smoacs-paper/bin/python" -m pip --python "$task_python" install \
 --no-index --only-binary=:all: --find-links results/smoacs/blackwell-wheels \
 --find-links results/smoacs/direct-wheels -r experiments/smoacs/requirements-blackwell.txt
"$task_python" -m pip check
"$task_python" -m pip freeze --all > "results/smoacs/env/blackwell-freeze-${SLURM_JOB_ID}.txt"
