#!/bin/bash
#SBATCH --job-name=MatterGen-CF-B-repair
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/mattergen-CF-repair-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-CF-repair-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
task_run="${MATTERGEN_CF_FULL_RUN:?}"
task_source="${MATTERGEN_TEMPLATE_SOURCE:-$PWD/results/mattergen/gap2p5-task128-20261024/full128}"
task_memory="$task_run/repair-memory-$SLURM_JOB_ID"
mkdir -p "$task_memory" results/smoacs/status
task_monitor=""
repair_cleanup() {
 local task_code=$?
 trap - EXIT
 touch "$task_memory/memory.stop"
 if [[ -n "$task_monitor" ]]; then wait "$task_monitor" || true; fi
 printf '{"job_id":"%s","exit_code":%d,"ended_at":"%s"}\n' "$SLURM_JOB_ID" "$task_code" "$(date -u +%FT%TZ)" \
  > "results/smoacs/status/job-$SLURM_JOB_ID.json"
 exit "$task_code"
}
trap repair_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python3 experiments/mattergen/cf_templates/monitor.py --output "$task_memory" --stop-file "$task_memory/memory.stop" &
task_monitor=$!
env MATTERGEN_CF_RUN="$task_run/neutral" MATTERGEN_CF_COHORT=neutral \
 bash experiments/mattergen/cf_templates/pipeline.sh > "$task_run/neutral-repair-$SLURM_JOB_ID.out" 2> "$task_run/neutral-repair-$SLURM_JOB_ID.err" &
task_neutral=$!
env MATTERGEN_CF_RUN="$task_run/unmasked" MATTERGEN_CF_COHORT=unmasked \
 bash experiments/mattergen/cf_templates/pipeline.sh > "$task_run/unmasked-repair-$SLURM_JOB_ID.out" 2> "$task_run/unmasked-repair-$SLURM_JOB_ID.err" &
task_unmasked=$!
task_neutral_code=0; wait "$task_neutral" || task_neutral_code=$?
task_unmasked_code=0; wait "$task_unmasked" || task_unmasked_code=$?
if [[ "$task_neutral_code" != 0 || "$task_unmasked_code" != 0 ]]; then exit 1; fi
python3 experiments/mattergen/cf_templates/full_report.py --run "$task_run" --source "$task_source" --refresh-merged
sbatch --parsable --export="ALL,MATTERGEN_CF_FULL_RUN=$task_run,MATTERGEN_TEMPLATE_SOURCE=$task_source" \
 experiments/mattergen/cf_templates/references.sbatch > "$task_run/SUN-job-id.txt"
