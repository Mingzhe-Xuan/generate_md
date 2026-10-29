#!/bin/bash
#SBATCH --job-name=OrthogonalSUN
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/smoacs/logs/orthogonal-SUN-%j.out
#SBATCH --error=results/smoacs/logs/orthogonal-SUN-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for arm in baseline orthogonal; do
 .venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py --selection "${ORTHOGONAL_ROOT:?}/references" --proposal-root "$ORTHOGONAL_ROOT/$arm" --output "$ORTHOGONAL_ROOT/$arm/SUN"
done
.venv-smoacs-dft/bin/python experiments/mattergen/orthogonal_score/report.py --run "$ORTHOGONAL_ROOT"
tar --exclude=./references/novelty-structures.jsonl.gz -czf "$ORTHOGONAL_ROOT-evidence.tar.gz" -C "$ORTHOGONAL_ROOT" .
sha256sum "$ORTHOGONAL_ROOT-evidence.tar.gz" > "$ORTHOGONAL_ROOT-evidence.tar.gz.sha256"
mkdir "$ORTHOGONAL_ROOT-evidence.parts"
split -b 512K -d -a 3 "$ORTHOGONAL_ROOT-evidence.tar.gz" "$ORTHOGONAL_ROOT-evidence.parts/part-"
(cd "$ORTHOGONAL_ROOT-evidence.parts" && sha256sum part-* > SHA256SUMS)
