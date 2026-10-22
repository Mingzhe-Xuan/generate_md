#!/bin/bash
#SBATCH --job-name=smoacs-install
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/install-%j.out
#SBATCH --error=results/smoacs/logs/install-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source experiments/smoacs/job_status.sh
task_python="${SMOACS_PYTHON:-$PWD/.venv-smoacs-paper/bin/python}"
task_wheels="${SMOACS_WHEELS:?Set SMOACS_WHEELS to a completed wheel directory}"
mkdir -p results/smoacs/env
"$task_python" -m pip install --no-index --only-binary=:all:   --find-links "$task_wheels" -r experiments/smoacs/requirements-paper.txt   'torch==2.4.0+cu124' 'dgl==2.4.0+cu124'
"$task_python" -m pip check
"$task_python" -m pip freeze --all > "results/smoacs/env/freeze-${SLURM_JOB_ID}.txt"
