#!/bin/bash
#SBATCH --job-name=FwdInvGraph
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/forward-inverse-graph-%j.out
#SBATCH --error=results/smoacs/logs/forward-inverse-graph-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 DGLBACKEND=pytorch
.venv-mattergen/bin/python experiments/mattergen/forward_inverse/graph_counterfactual.py \
    --source "${FORWARD_INVERSE_SOURCE:?}" --run "${FORWARD_INVERSE_GRAPH:?}"
tar -czf "$FORWARD_INVERSE_GRAPH/evidence.tar.gz" -- "$FORWARD_INVERSE_GRAPH/complete.json"
sha256sum "$FORWARD_INVERSE_GRAPH/evidence.tar.gz" > "$FORWARD_INVERSE_GRAPH/evidence.tar.gz.sha256"
