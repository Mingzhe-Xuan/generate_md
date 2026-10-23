#!/bin/bash
#SBATCH --job-name=alternating-eval
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/alternating-eval-%j.out
#SBATCH --error=results/smoacs/logs/alternating-eval-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
task_run="${SMOACS_AB_RUN:?}"
task_root="$task_run/postprocess"
.venv-smoacs-paper/bin/python experiments/smoacs/alternating/export.py --run "$task_run"
.venv-smoacs-paper/bin/python experiments/smoacs/postprocess/predict_relaxed.py \
 --inventory "$task_root/inventory-249/inventory.json" --assets results/smoacs/assets-199 --root "$task_root"
.venv-smoacs-blackwell/bin/python experiments/smoacs/cross_predictor/predict.py \
 --inventory "$task_root/inventory-249/inventory.json" --predictor "$task_root/predictor-$SLURM_JOB_ID" \
 --relaxation "$task_root/relaxation" --author-csv "$task_root/result.csv" \
 --weights results/smoacs/alignn-weights --output "$task_run/cross-prediction-$SLURM_JOB_ID"
.venv-smoacs-paper/bin/python experiments/smoacs/cross_predictor/summarize.py \
 --input "$task_run/cross-prediction-$SLURM_JOB_ID" --output "$task_run/cross-analysis-$SLURM_JOB_ID"
python3 - "$task_run" "$SLURM_JOB_ID" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);job=sys.argv[2]
(root/'evaluation.json').write_text(json.dumps(dict(
 cross_summary=str(root/('cross-analysis-'+job)/'summary.json'),
 cross_predictions=str(root/('cross-prediction-'+job)/'predictions.json')),indent=2)+'\n')
PY
