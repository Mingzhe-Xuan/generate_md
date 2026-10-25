#!/bin/bash
#SBATCH --job-name=Basin16-confirm
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/smoacs/logs/basin-confirm-%j.out
#SBATCH --error=results/smoacs/logs/basin-confirm-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
: "${BASIN_DISCOVERY_ROOT:?}"
task_decision=0
python3 experiments/mattergen/basin_search/confirmation.py --source "$BASIN_DISCOVERY_ROOT" || task_decision=$?
if [[ "$task_decision" == 3 ]]; then
 printf '%s\n' 'No new method passed discovery; confirmation skipped, goal remains unachieved.'
 exit 0
fi
if [[ "$task_decision" != 0 ]]; then exit "$task_decision"; fi
export BASIN_RUN_ROOT="$PWD/results/mattergen/basin-confirm-$SLURM_JOB_ID"
export BASIN_SEARCH_SEED=20261027
export BASIN_FROZEN_PLAN="$BASIN_DISCOVERY_ROOT/plan.json"
bash experiments/mattergen/basin_search/run.sbatch
