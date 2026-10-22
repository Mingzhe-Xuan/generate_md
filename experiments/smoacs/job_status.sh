#!/bin/bash
# Source after changing into the parent repository root.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf '%s\n' 'Slurm job ID is required for the status recorder.' >&2
  return 2
fi
mkdir -p results/smoacs/status
smoacs_record_exit() {
  local task_exit=$?
  trap - EXIT
  printf '{"job_id":"%s","exit_code":%d,"ended_at":"%s"}\n' \
    "$SLURM_JOB_ID" "$task_exit" "$(date -u +%FT%TZ)" \
    > "results/smoacs/status/job-${SLURM_JOB_ID}.json"
  exit "$task_exit"
}
trap smoacs_record_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
