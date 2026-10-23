#!/bin/bash
#SBATCH --job-name=cf-alignn-validate
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=results/smoacs/logs/cross-predict-%j.out
#SBATCH --error=results/smoacs/logs/cross-predict-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
.venv-smoacs-blackwell/bin/python experiments/smoacs/cross_predictor/predict.py \
 --inventory results/smoacs/postprocess-206/inventory-249/inventory.json \
 --predictor results/smoacs/postprocess-206/predictor-245 \
 --relaxation results/smoacs/postprocess-206/relaxation \
 --author-csv results/smoacs/optimization/section21-Struct-AtomicMask-grad-206/result.csv \
 --weights results/smoacs/alignn-weights \
 --output "results/smoacs/cross-predictor-206/prediction-${SLURM_JOB_ID}"
