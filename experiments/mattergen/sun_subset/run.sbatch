#!/bin/bash
#SBATCH --job-name=NoiseCF20-SUN12
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/noiseCF-SUN-%j.out
#SBATCH --error=results/smoacs/logs/noiseCF-SUN-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_run="$PWD/results/mattergen/sun-noise03-cf20-$SLURM_JOB_ID"
.venv-smoacs-dft/bin/python experiments/mattergen/sun_subset/prepare.py \
 --source "${BESTN_SUN_SOURCE:?}" --output "$task_run"
python3 experiments/mattergen/cf_templates/cache_references.py --selection "$task_run/references" \
 --extra-selection results/mattergen/cf-full-340/references/selected-343
.venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py \
 --selection "$task_run/references" --proposal-root "$task_run" --output "$task_run/SUN"
python3 experiments/mattergen/sun_subset/report.py --run "$task_run"
git rev-parse HEAD > "$task_run/source-commit.txt"
tar -czf "$task_run-summary.tar.gz" -C "$task_run" manifest.json report.json report.md source-commit.txt \
 SUN/summary.json SUN/candidates.json SUN/reference-errors.json references/cache-reuse.json references/mattersim-inventory.json
sha256sum "$task_run-summary.tar.gz" > "$task_run-summary.tar.gz.sha256"
