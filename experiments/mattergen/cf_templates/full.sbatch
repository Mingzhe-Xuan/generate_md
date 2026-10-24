#!/bin/bash
#SBATCH --job-name=MatterGen-CF-full128
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/mattergen-CF-full-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-CF-full-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export MATTERGEN_CF_FULL_RUN="${MATTERGEN_CF_FULL_RUN:-$PWD/results/mattergen/cf-full-$SLURM_JOB_ID}"
export MATTERGEN_TEMPLATE_SOURCE="${MATTERGEN_TEMPLATE_SOURCE:-$PWD/results/mattergen/gap2p5-task128-20261024/full128}"
mkdir -p "$MATTERGEN_CF_FULL_RUN" results/smoacs/status
task_monitor=""
task_status_tag="$SLURM_JOB_ID"
if [[ -n "${SLURM_STEP_ID:-}" && "$SLURM_STEP_ID" != batch && "$SLURM_STEP_ID" != extern ]]; then
 task_status_tag="$SLURM_JOB_ID-step$SLURM_STEP_ID"
fi
full_cleanup() {
 local task_code=$?
 trap - EXIT
 touch "$MATTERGEN_CF_FULL_RUN/memory.stop"
 if [[ -n "$task_monitor" ]]; then wait "$task_monitor" || true; fi
 printf '{"job_id":"%s","exit_code":%d,"ended_at":"%s"}\n' "$SLURM_JOB_ID" "$task_code" "$(date -u +%FT%TZ)" \
  > "results/smoacs/status/job-$task_status_tag.json"
 exit "$task_code"
}
trap full_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python3 experiments/mattergen/cf_templates/monitor.py --output "$MATTERGEN_CF_FULL_RUN" --stop-file "$MATTERGEN_CF_FULL_RUN/memory.stop" &
task_monitor=$!
env MATTERGEN_CF_RUN="$MATTERGEN_CF_FULL_RUN/neutral" MATTERGEN_CF_COHORT=neutral \
 bash experiments/mattergen/cf_templates/pipeline.sh > "$MATTERGEN_CF_FULL_RUN/neutral.out" 2> "$MATTERGEN_CF_FULL_RUN/neutral.err" &
task_neutral=$!
env MATTERGEN_CF_RUN="$MATTERGEN_CF_FULL_RUN/unmasked" MATTERGEN_CF_COHORT=unmasked \
 bash experiments/mattergen/cf_templates/pipeline.sh > "$MATTERGEN_CF_FULL_RUN/unmasked.out" 2> "$MATTERGEN_CF_FULL_RUN/unmasked.err" &
task_unmasked=$!
task_neutral_code=0; wait "$task_neutral" || task_neutral_code=$?
task_unmasked_code=0; wait "$task_unmasked" || task_unmasked_code=$?
if [[ "$task_neutral_code" != 0 || "$task_unmasked_code" != 0 ]]; then
 printf 'Strategy exits: neutral=%s unmasked=%s\n' "$task_neutral_code" "$task_unmasked_code" >&2
 exit 1
fi
python3 experiments/mattergen/cf_templates/full_report.py --run "$MATTERGEN_CF_FULL_RUN" --source "$MATTERGEN_TEMPLATE_SOURCE"
if [[ -f "$MATTERGEN_CF_FULL_RUN/SKIP_SUN" || "${MATTERGEN_CF_SKIP_SUN:-0}" == 1 ]]; then
 printf '%s\n' 'SUN skipped by user request; property evaluation complete.'
 exit 0
fi
sbatch --parsable --export="ALL,MATTERGEN_CF_FULL_RUN=$MATTERGEN_CF_FULL_RUN,MATTERGEN_TEMPLATE_SOURCE=$MATTERGEN_TEMPLATE_SOURCE" \
 experiments/mattergen/cf_templates/references.sbatch > "$MATTERGEN_CF_FULL_RUN/SUN-job-id.txt"
