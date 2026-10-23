#!/bin/bash
#SBATCH --job-name=MatterGen-SUN-serial
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=results/smoacs/logs/mattergen-SUN-resume-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-SUN-resume-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_run="${MATTERGEN_RUN:?}"
task_selection="${MATTERGEN_REFERENCE_SELECTION:?}"
python3 experiments/mattergen/reference_resume.py --run "$task_run" --selection "$task_selection"
.venv-smoacs-postprocess/bin/python experiments/smoacs/postprocess/relax.py \
 --inventory "$task_selection/mattersim-inventory.json" --output "$task_selection/mattersim-relaxation" \
 --checkpoint results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth
if [[ ! -f "$task_run/SUN/summary.json" ]]; then
 if [[ -d "$task_run/SUN" ]]; then mv "$task_run/SUN" "$task_run/SUN-incomplete-$SLURM_JOB_ID"; fi
 .venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py --selection "$task_selection" --proposal-root "$task_run" --output "$task_run/SUN"
fi
python3 experiments/mattergen/report.py --run "$task_run" --selection "$task_selection"
python3 experiments/mattergen/resume_jobs.py --manifest "${MATTERGEN_PAUSED_MANIFEST:?}" --report "$task_run/report.json"
