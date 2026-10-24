#!/bin/bash
#SBATCH --job-name=MatterGen-BestN-pilot
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/mattergen-bestN-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-bestN-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export MATTERGEN_CF_COHORT=bestN_pilot
task_run="${MATTERGEN_BESTN_RUN:-$PWD/results/mattergen/bestN-$SLURM_JOB_ID}"
task_source="${MATTERGEN_TEMPLATE_SOURCE:-$PWD/results/mattergen/gap2p5-task128-20261024/full128}"
mkdir -p "$task_run"
touch "$task_run/SKIP_SUN"
task_prepare=()
if [[ -n "${MATTERGEN_BESTN_AMPLITUDE:-}" ]]; then
 task_prepare+=(--amplitude "$MATTERGEN_BESTN_AMPLITUDE")
fi
.venv-smoacs-postprocess/bin/python experiments/mattergen/best_of_n/prepare.py --source "$task_source" --output "$task_run" \
 --proposal-cap "${MATTERGEN_BESTN_PROPOSAL_CAP:-0.4}" "${task_prepare[@]}"
if [[ ! -f "$task_run/cf20/complete.json" ]]; then
 .venv-smoacs-blackwell/bin/python experiments/mattergen/cf_templates/run.py \
  --source "$task_source" --output "$task_run/cf20" --cohort pilot --fixed-species --steps 20 \
  --branch-manifest "$task_run/manifest.json"
fi
if [[ ! -f "$task_run/cf20/evaluation-cycle1/inventory-249/inventory.json" ]]; then
 python3 experiments/smoacs/adam_cycles/export.py --run "$task_run/cf20" --cycles 1
fi
.venv-smoacs-postprocess/bin/python experiments/mattergen/best_of_n/run_raw.py --run "$task_run"
env BESTN_EVALUATION="$task_run/raw-evaluation" bash experiments/mattergen/best_of_n/evaluate.sh
env BESTN_EVALUATION="$task_run/cf20/evaluation-cycle1" bash experiments/mattergen/best_of_n/evaluate.sh
.venv-smoacs-postprocess/bin/python experiments/mattergen/best_of_n/report.py --run "$task_run"
git rev-parse HEAD > "$task_run/source-commit.txt"
tar -czf "${task_run}-evidence.tar.gz" --exclude=loader-tmp --exclude='capture-*' \
 --exclude='*.pth' --exclude='*.zip' --exclude='*.npz' --exclude='*.npy' -C "$task_run" .
sha256sum "${task_run}-evidence.tar.gz" > "${task_run}-evidence.tar.gz.sha256"
