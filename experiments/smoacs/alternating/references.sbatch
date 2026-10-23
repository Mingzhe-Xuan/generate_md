#!/bin/bash
#SBATCH --job-name=alternating-SUN
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/smoacs/logs/alternating-SUN-%j.out
#SBATCH --error=results/smoacs/logs/alternating-SUN-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
export SMOACS_REFERENCE_PROXY=""
task_run="${SMOACS_AB_RUN:?}"
task_root="$task_run/postprocess"
task_selection="$task_run/references/selected-$SLURM_JOB_ID"
curl --noproxy '*' --head --connect-timeout 10 --max-time 30 https://materialsproject-build.s3.amazonaws.com/ >/dev/null
.venv-smoacs-reference/bin/python experiments/smoacs/references/prepare.py \
 --inventory "$task_root/inventory-249/inventory.json" --output "$task_run/references" \
 --cache results/smoacs/mp-public-reference/range-cache
.venv-smoacs-postprocess/bin/python experiments/smoacs/references/convert.py \
 --selection "$task_selection" --proposal-root "$task_root"
task_count="$(.venv-smoacs-postprocess/bin/python -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["candidates"]))' "$task_selection/mattersim-inventory.json")"
if [[ "$task_count" -gt 0 ]]; then
 .venv-smoacs-postprocess/bin/python experiments/smoacs/alternating/reference_cache.py --selection "$task_selection"
 .venv-smoacs-postprocess/bin/python experiments/smoacs/references/parallel.py --selection "$task_selection" --workers 4 --benchmark
 .venv-smoacs-postprocess/bin/python experiments/smoacs/references/parallel.py --selection "$task_selection" --workers 4 \
  --validated-benchmark "$task_selection/parallel-benchmark-$SLURM_JOB_ID"
fi
.venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py --selection "$task_selection" --proposal-root "$task_root"
.venv-smoacs-paper/bin/python experiments/smoacs/alternating/report.py --run "$task_run" \
 --sun "$task_selection/assessment-$SLURM_JOB_ID"
