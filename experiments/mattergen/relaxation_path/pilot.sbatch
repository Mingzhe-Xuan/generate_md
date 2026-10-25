#!/bin/bash
#SBATCH --job-name=MatterGen-B-path
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/smoacs/logs/mattergen-B-path-%j.out
#SBATCH --error=results/smoacs/logs/mattergen-B-path-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DGLBACKEND=pytorch
task_root="$PWD/results/mattergen/relaxation-path-$SLURM_JOB_ID"
python3 experiments/mattergen/relaxation_path/replay.py --output "$task_root"
.venv-smoacs-blackwell/bin/python experiments/mattergen/relaxation_path/score.py --run "$task_root"
python3 experiments/mattergen/relaxation_path/report.py --run "$task_root"
git rev-parse HEAD > "$task_root/source-commit.txt"
tar -czf "${task_root}-evidence.tar.gz" -C "$task_root" .
sha256sum "${task_root}-evidence.tar.gz" > "${task_root}-evidence.tar.gz.sha256"
