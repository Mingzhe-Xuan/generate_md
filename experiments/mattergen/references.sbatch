#!/bin/bash
#SBATCH --job-name=MatterGen-SUN
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=results/smoacs/logs/mattergen-SUN-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-SUN-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ALL_PROXY=socks5h://127.0.0.1:1080 HTTP_PROXY=socks5h://127.0.0.1:1080 HTTPS_PROXY=socks5h://127.0.0.1:1080
if curl --fail --proxy "$ALL_PROXY" --head --connect-timeout 10 --max-time 30 https://github.com/ >/dev/null; then
 export SMOACS_REFERENCE_PROXY="$ALL_PROXY"
else
 curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 \
 https://materialsproject-build.s3.amazonaws.com/collections/summary/version=2026-04-13/part-00000-d0a74fbe-68f8-4a2d-9d43-67cb6a1bc912-c000.zstd.parquet >/dev/null
 unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
 export SMOACS_REFERENCE_PROXY=""
fi
task_run="${MATTERGEN_RUN:?}"
task_selection="$task_run/references/selected-$SLURM_JOB_ID"
.venv-smoacs-reference/bin/python experiments/smoacs/references/prepare.py \
 --inventory "$task_run/reference-scope.json" --output "$task_run/references" --cache results/smoacs/mp-public-reference/range-cache
.venv-smoacs-postprocess/bin/python experiments/smoacs/references/convert.py \
 --selection "$task_selection" --proposal-root "$task_run" --all-selected
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
.venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py --selection "$task_selection" --proposal-root "$task_run" --output "$task_run/SUN"
python3 experiments/mattergen/report.py --run "$task_run" --selection "$task_selection"
python3 experiments/mattergen/resume_jobs.py --manifest "${MATTERGEN_PAUSED_MANIFEST:?}" --report "$task_run/report.json"
