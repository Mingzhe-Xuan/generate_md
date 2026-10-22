#!/bin/bash
#SBATCH --job-name=cpu206-model
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=results/smoacs/logs/post-model-%j.out
#SBATCH --error=results/smoacs/logs/post-model-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
task_root=results/smoacs/postprocess-206
task_model="$task_root/models/mattersim-v1.0.0-5M.pth"
tar -xOf "$task_root/downloads/mattersim-full-d458f54.tar.gz" \
 differentiable-mattersim-d458f54d42bb9c23fcd0957b33e00135e541125c/pretrained_models/mattersim-v1.0.0-5M.pth \
 > "$task_model.tmp"
printf '%s  %s\n' e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5 "$task_model.tmp" | sha256sum -c -
mv "$task_model.tmp" "$task_model"
