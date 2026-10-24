#!/bin/bash
#SBATCH --job-name=MatterGen-CF-templates
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/mattergen-CF-templates-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-CF-templates-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_source="${MATTERGEN_TEMPLATE_SOURCE:-$PWD/results/mattergen/gap2p5-task128-20261024/full128}"
task_run="$PWD/results/mattergen/cf-template-pilot-$SLURM_JOB_ID"
.venv-smoacs-blackwell/bin/python experiments/mattergen/cf_templates/run.py --source "$task_source" --output "$task_run"
python3 experiments/smoacs/adam_cycles/export.py --run "$task_run" --cycles 1
task_eval="$task_run/evaluation-cycle1"
.venv-smoacs-blackwell/bin/python experiments/smoacs/postprocess/predict_relaxed.py \
 --inventory "$task_eval/inventory-249/inventory.json" --assets results/smoacs/assets-199 --root "$task_eval" --device cuda
.venv-smoacs-blackwell/bin/python experiments/smoacs/cross_predictor/predict.py \
 --inventory "$task_eval/inventory-249/inventory.json" --predictor "$task_eval/predictor-$SLURM_JOB_ID" \
 --relaxation "$task_eval/relaxation" --author-csv "$task_eval/result.csv" \
 --weights results/smoacs/alignn-weights --output "$task_eval/cross-prediction"
python3 experiments/mattergen/cf_templates/report.py --run "$task_run" --source "$task_source"
