#!/bin/bash
#SBATCH --job-name=MatterGen-CF-fullSUN
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=results/smoacs/logs/mattergen-CF-SUN-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-CF-SUN-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
export SMOACS_REFERENCE_PROXY=""
task_run="${MATTERGEN_CF_FULL_RUN:?}"
task_source="${MATTERGEN_TEMPLATE_SOURCE:?}"
export MATTERGEN_CF_COHORT=reference
task_memory="$task_run/reference-memory-$SLURM_JOB_ID"
mkdir -p "$task_memory"
printf '%s\n' "$task_memory" > "$task_run/reference-memory-latest.txt"
python3 experiments/mattergen/cf_templates/monitor.py --output "$task_memory" \
 --stop-file "$task_memory/memory.stop" &
task_monitor=$!
curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 \
 https://materialsproject-build.s3.amazonaws.com/collections/summary/version=2026-04-13/part-00000-d0a74fbe-68f8-4a2d-9d43-67cb6a1bc912-c000.zstd.parquet >/dev/null
.venv-smoacs-reference/bin/python experiments/smoacs/references/prepare.py \
 --inventory "$task_run/reference-scope.json" --output "$task_run/references" --cache results/smoacs/mp-public-reference/range-cache
task_selection="$task_run/references/selected-$SLURM_JOB_ID"
.venv-smoacs-postprocess/bin/python experiments/smoacs/references/convert.py \
 --selection "$task_selection" --proposal-root "$task_run/evaluation" --all-selected
python3 experiments/mattergen/cf_templates/cache_references.py --selection "$task_selection"
.venv-smoacs-postprocess/bin/python experiments/smoacs/postprocess/relax.py \
 --inventory "$task_selection/mattersim-inventory.json" --output "$task_selection/mattersim-relaxation" \
 --checkpoint results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth
.venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py --selection "$task_selection" --proposal-root "$task_source" --output "$task_run/SUN-direct"
.venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py --selection "$task_selection" --proposal-root "$task_run/evaluation" --output "$task_run/SUN-hybrid"
printf '%s\n' "$task_selection" > "$task_run/SUN-selection.txt"
python3 experiments/mattergen/cf_templates/full_report.py --run "$task_run" --source "$task_source"
touch "$task_run/SUN.complete"
touch "$task_memory/memory.stop"
wait "$task_monitor"
