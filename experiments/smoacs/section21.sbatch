#!/bin/bash
#SBATCH --job-name=smoacs-section21
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/section21-%j.out
#SBATCH --error=results/smoacs/logs/section21-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the parent repository root}"
task_assets="${SMOACS_ASSETS:?Set SMOACS_ASSETS to the prepared asset directory}"
task_python="${SMOACS_PYTHON:-$PWD/.venv-smoacs-paper/bin/python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export DGLBACKEND=pytorch
"$task_python" experiments/smoacs/run_section21.py \
 --source third_party/SMOACS --assets "$task_assets" \
 --archive results/smoacs/downloads/reproducibility.tar.gz \
 --output-root results/smoacs/optimization \
 --exp-name "section21-${SMOACS_INITIALIZATION:-Struct-AtomicMask}-${SMOACS_METHOD:-grad}-${SLURM_JOB_ID}" \
 --device "${SMOACS_DEVICE:-cuda}" \
 --initialization "${SMOACS_INITIALIZATION:-Struct-AtomicMask}" \
 --method "${SMOACS_METHOD:-grad}" --seed "${SMOACS_SEED:-42}" \
 --candidates "${SMOACS_CANDIDATES:-256}" --batch-size "${SMOACS_BATCH_SIZE:-256}" \
 --steps "${SMOACS_STEPS:-200}" \
 --mutation-steps ${SMOACS_MUTATION_STEPS:-100 150}
