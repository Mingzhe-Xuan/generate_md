#!/bin/bash
#SBATCH --job-name=cpu206-term-test
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --output=results/smoacs/logs/post-term-%j.out
#SBATCH --error=results/smoacs/logs/post-term-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
if [[ "${SETVARS_COMPLETED:-}" != 1 ]]; then
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
fi
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 I_MPI_PIN=0
export I_MPI_PMI_LIBRARY="$PWD/results/smoacs/postprocess-206/runtime/pmi/usr/lib/x86_64-linux-gnu/libpmi2.so.0"
.venv-smoacs-dft/bin/python experiments/smoacs/postprocess/termination_probe.py
