#!/bin/bash
#SBATCH --job-name=openmx-U-probe
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/openmx-U-%j.out
#SBATCH --error=results/smoacs/logs/openmx-U-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
if [[ "${SETVARS_COMPLETED:-}" != 1 ]]; then
 source /opt/intel/oneapi/setvars.sh > "results/smoacs/logs/oneapi-U-${SLURM_JOB_ID}.out" 2>&1
fi
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 I_MPI_PIN=0
export I_MPI_PMI_LIBRARY="$PWD/results/smoacs/postprocess-206/runtime/pmi/usr/lib/x86_64-linux-gnu/libpmi2.so.0"
.venv-smoacs-postprocess/bin/python experiments/smoacs/openmx/u_probe.py
