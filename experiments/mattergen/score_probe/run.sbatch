#!/bin/bash
#SBATCH --job-name=ScoreJointProbe
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=results/smoacs/logs/score-probe-%j.out
#SBATCH --error=results/smoacs/logs/score-probe-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
.venv-mattergen/bin/python experiments/mattergen/score_probe/run_probe.py --source "${SCORE_PROBE_SOURCE:?}" --run "${SCORE_PROBE_ROOT:?}"
.venv-mattergen/bin/python experiments/mattergen/score_probe/report_probe.py --run "$SCORE_PROBE_ROOT"
