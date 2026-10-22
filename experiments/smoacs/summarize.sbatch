#!/bin/bash
#SBATCH --job-name=smoacs-summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=results/smoacs/logs/summary-%j.out
#SBATCH --error=results/smoacs/logs/summary-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source experiments/smoacs/job_status.sh
"${SMOACS_PYTHON:-$PWD/.venv-smoacs-paper/bin/python}" experiments/smoacs/summarize_section21.py --csv "${SMOACS_RESULT_CSV:?}" --expected-candidates "${SMOACS_EXPECTED_CANDIDATES:?}" --output "results/smoacs/assessment/summary-${SLURM_JOB_ID}.json"
