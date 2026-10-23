#!/bin/bash
#SBATCH --job-name=AdamAB-SUN
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/smoacs/logs/adam-SUN-%j.out
#SBATCH --error=results/smoacs/logs/adam-SUN-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
export SMOACS_REFERENCE_PROXY=""
task_run="${SMOACS_ADAM_RUN:?}"
task_selection="$task_run/references/selected-$SLURM_JOB_ID"
curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 \
 https://materialsproject-build.s3.amazonaws.com/collections/summary/version=2026-04-13/part-00000-d0a74fbe-68f8-4a2d-9d43-67cb6a1bc912-c000.zstd.parquet >/dev/null
.venv-smoacs-reference/bin/python experiments/smoacs/references/prepare.py \
 --inventory "$task_run/reference-scope.json" --output "$task_run/references" \
 --cache results/smoacs/mp-public-reference/range-cache
.venv-smoacs-postprocess/bin/python experiments/smoacs/references/convert.py \
 --selection "$task_selection" --proposal-root "$task_run/evaluation-cycle1" --all-selected
task_count="$(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["candidates"]))' "$task_selection/mattersim-inventory.json")"
if [[ "$task_count" -gt 128 ]]; then
 .venv-smoacs-postprocess/bin/python experiments/smoacs/alternating/reference_cache.py --selection "$task_selection"
 .venv-smoacs-postprocess/bin/python experiments/smoacs/references/parallel.py --selection "$task_selection" --workers 4 --benchmark
 .venv-smoacs-postprocess/bin/python experiments/smoacs/references/parallel.py --selection "$task_selection" --workers 4 \
  --validated-benchmark "$task_selection/parallel-benchmark-$SLURM_JOB_ID"
elif [[ "$task_count" -gt 0 ]]; then
 python3 experiments/smoacs/adam_cycles/reuse.py --selection "$task_selection"
 .venv-smoacs-postprocess/bin/python experiments/smoacs/postprocess/relax.py \
  --inventory "$task_selection/mattersim-inventory.json" --output "$task_selection/mattersim-relaxation" \
  --checkpoint results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth
else
 mkdir -p "$task_selection/mattersim-relaxation"
fi
for task_cycle in 1 3; do
 .venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py --selection "$task_selection" \
  --proposal-root "$task_run/evaluation-cycle$task_cycle" --output "$task_run/evaluation-cycle$task_cycle/SUN"
done
.venv-smoacs-paper/bin/python experiments/smoacs/adam_cycles/report.py --run "$task_run"
