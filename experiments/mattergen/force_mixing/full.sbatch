#!/bin/bash
#SBATCH --job-name=MatterGen-force-mix-full
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mattergen-force-mix-full-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-force-mix-full-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
: "${FORCE_MIX_PROBE:?Native validated pilot proof required}"
task_root="${FORCE_MIX_FULL_ROOT:-$PWD/results/mattergen/force-mix-full-$SLURM_JOB_ID}"
mkdir -p "$task_root"
touch "$task_root/SKIP_SUN"
task_monitor=""
mixed_cleanup() {
 local task_code=$?
 trap - EXIT
 touch "$task_root/memory.stop"
 if [[ -n "$task_monitor" ]]; then wait "$task_monitor" || true; fi
 printf '{"job_id":"%s","exit_code":%d,"ended_at":"%s"}\n' "$SLURM_JOB_ID" "$task_code" "$(date -u +%FT%TZ)" \
  > "results/smoacs/status/job-$SLURM_JOB_ID.json"
 exit "$task_code"
}
trap mixed_cleanup EXIT
python3 experiments/mattergen/cf_templates/monitor.py --output "$task_root" --stop-file "$task_root/memory.stop" &
task_monitor=$!
run_lambda() {
 local task_weight="$1"
 for task_cohort in neutral unmasked; do
  env FORCE_MIX_RUN="$task_root/lambda-$task_weight/$task_cohort" FORCE_MIX_WEIGHT="$task_weight" \
   FORCE_MIX_COHORT="$task_cohort" MATTERGEN_CF_COHORT="force_lambda_${task_weight}_${task_cohort}" \
   bash experiments/mattergen/force_mixing/pipeline.sh
 done
}
task_pids=()
for task_weight in 0.1 0.2 0.3; do
 run_lambda "$task_weight" > "$task_root/lambda-$task_weight.out" 2> "$task_root/lambda-$task_weight.err" &
 task_pids+=("$!")
done
task_failed=0
for task_pid in "${task_pids[@]}"; do wait "$task_pid" || task_failed=1; done
if [[ "$task_failed" != 0 ]]; then exit 1; fi
touch "$task_root/memory.stop"
wait "$task_monitor"
task_monitor=""
.venv-smoacs-postprocess/bin/python experiments/mattergen/force_mixing/report.py --run "$task_root" \
 --source results/mattergen/gap2p5-task128-20261024/full128 \
 --baseline results/mattergen/alignn-full-361/cf200
git rev-parse HEAD > "$task_root/source-commit.txt"
tar -czf "${task_root}-evidence.tar.gz" --exclude=loader-tmp --exclude='capture-*' \
 --exclude='*.pth' --exclude='*.zip' --exclude='*.npz' --exclude='*.npy' -C "$task_root" .
sha256sum "${task_root}-evidence.tar.gz" > "${task_root}-evidence.tar.gz.sha256"
