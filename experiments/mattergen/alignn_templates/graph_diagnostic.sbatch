#!/bin/bash
#SBATCH --job-name=ALIGNN-graph-diagnostic
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/smoacs/logs/alignn-graph-diagnostic-%j.out
#SBATCH --error=results/smoacs/logs/alignn-graph-diagnostic-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
.venv-smoacs-blackwell/bin/python experiments/mattergen/alignn_templates/graph_diagnostic.py \
 --run "$PWD/results/mattergen/alignn-full-361" \
 --output "$PWD/results/mattergen/alignn-graph-diagnostic-$SLURM_JOB_ID"
