#!/bin/bash
#SBATCH --job-name=OrthogonalMPSelect
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/orthogonal-select-%j.out
#SBATCH --error=results/smoacs/logs/orthogonal-select-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-smoacs-reference/bin/python experiments/mattergen/orthogonal_score/offline_references.py --run "${ORTHOGONAL_ROOT:?}"
.venv-smoacs-postprocess/bin/python experiments/smoacs/references/convert.py --selection "$ORTHOGONAL_ROOT/references" --proposal-root "$ORTHOGONAL_ROOT/combined" --all-selected
.venv-smoacs-postprocess/bin/python experiments/mattergen/cf_templates/cache_references.py --selection "$ORTHOGONAL_ROOT/references" --extra-selection results/mattergen/cf-full-340/references/selected-343
