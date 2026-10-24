#!/bin/bash
#SBATCH --job-name=MatterGen-fixed-coordinator
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=results/smoacs/logs/mattergen-CF-shared-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-CF-shared-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
task_coordinator="$SLURM_JOB_ID"
task_owner="${MATTERGEN_CF_OWNER:?Existing owned GPU job is required}"
[[ "$task_owner" =~ ^[0-9]+$ ]]
task_owner_run="$PWD/results/mattergen/cf-full-$task_owner"
export MATTERGEN_CF_FULL_RUN="$PWD/results/mattergen/cf-fixed-$task_coordinator"
export MATTERGEN_CF_FIXED_SPECIES=1 MATTERGEN_CF_STEPS=600 MATTERGEN_CF_SKIP_SUN=1
mkdir -p "$MATTERGEN_CF_FULL_RUN" results/smoacs/status
task_registration=""
coordinator_cleanup() {
 local task_code=$?
 trap - EXIT
 if [[ -n "$task_registration" ]]; then
  printf '{"coordinator":"%s","completed":true,"exit_code":%d}\n' "$task_coordinator" "$task_code" > "$task_registration.tmp"
  mv "$task_registration.tmp" "$task_registration"
 fi
 printf '{"job_id":"%s","exit_code":%d,"ended_at":"%s"}\n' "$task_coordinator" "$task_code" "$(date -u +%FT%TZ)" > "results/smoacs/status/job-$task_coordinator.json"
 exit "$task_code"
}
trap coordinator_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
task_owner_user="$(squeue -h -j "$task_owner" -o %u 2>/dev/null || true)"
task_owner_state="$(squeue -h -j "$task_owner" -o %T 2>/dev/null || true)"
if [[ "$task_owner_state" != RUNNING ]]; then
 sbatch --parsable --job-name=MatterGen-CF600-fixed --mem=64G --export=ALL \
  experiments/mattergen/cf_templates/full.sbatch > "$MATTERGEN_CF_FULL_RUN/fallback-job-id.txt"
 printf '%s\n' 'Previous allocation ended; submitted an independent GPU job on the same node.'
 exit 0
fi
[[ "$task_owner_user" == "$USER" && -d "$task_owner_run" ]]
mkdir -p "$task_owner_run/shared-children"
task_registration="$task_owner_run/shared-children/$task_coordinator.json"
printf '{"coordinator":"%s","completed":false}\n' "$task_coordinator" > "$task_registration.tmp"
mv "$task_registration.tmp" "$task_registration"
touch "$task_owner_run/SKIP_SUN" "$MATTERGEN_CF_FULL_RUN/SKIP_SUN"
printf '{"owner_job":"%s","coordinator":"%s","shared_gpu":true}\n' "$task_owner" "$task_coordinator" > "$MATTERGEN_CF_FULL_RUN/allocation.json"
srun --jobid="$task_owner" --overlap --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=05:30:00 --export=ALL \
 bash experiments/mattergen/cf_templates/full.sbatch \
 > "$MATTERGEN_CF_FULL_RUN/allocation-step.out" 2> "$MATTERGEN_CF_FULL_RUN/allocation-step.err"
