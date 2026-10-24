#!/bin/bash
#SBATCH --job-name=MatterGen-ALIGNN-CF200
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mattergen-alignn-full-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-alignn-full-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
: "${ALIGNN_GRADIENT_PROBE:?Successful pilot gradient probe required}"
task_root="${ALIGNN_FULL_RUN:-$PWD/results/mattergen/alignn-full-$SLURM_JOB_ID}"
mkdir -p "$task_root"
touch "$task_root/SKIP_SUN"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
export ATOMGPTLAB_CACHE="$PWD/results/smoacs/jarvis-cache"
if [[ -n "${ALIGNN_CONTINUE_JOB:-}" ]]; then
 python3 experiments/mattergen/alignn_templates/continuation.py --run "$task_root" \
  --source "${MATTERGEN_TEMPLATE_SOURCE:-$PWD/results/mattergen/gap2p5-task128-20261024/full128}" --previous-job "$ALIGNN_CONTINUE_JOB"
fi
for task_cohort in neutral unmasked; do
 env ALIGNN_TEMPLATE_RUN="$task_root/alignn200/$task_cohort" ALIGNN_TEMPLATE_COHORT="$task_cohort" \
  bash experiments/mattergen/alignn_templates/pipeline.sh
done
for task_cohort in neutral unmasked; do
 env MATTERGEN_CF_RUN="$task_root/cf200/$task_cohort" MATTERGEN_CF_COHORT="$task_cohort" \
  MATTERGEN_CF_STEPS=200 MATTERGEN_CF_FIXED_SPECIES=1 MATTERGEN_CF_SKIP_SUN=1 \
  bash experiments/mattergen/cf_templates/pipeline.sh
done
.venv-smoacs-postprocess/bin/python experiments/mattergen/alignn_templates/report.py \
 --run "$task_root" --source results/mattergen/gap2p5-task128-20261024/full128
git rev-parse HEAD > "$task_root/source-commit.txt"
tar -czf "${task_root}-evidence.tar.gz" --exclude=loader-tmp --exclude='capture-*' \
 --exclude='*.pth' --exclude='*.zip' --exclude='*.npz' --exclude='*.npy' -C "$task_root" .
sha256sum "${task_root}-evidence.tar.gz" > "${task_root}-evidence.tar.gz.sha256"
