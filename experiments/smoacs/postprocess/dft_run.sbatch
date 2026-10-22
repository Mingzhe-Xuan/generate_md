#!/bin/bash
#SBATCH --job-name=cpu206-dft
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/post-dft-%j.out
#SBATCH --error=results/smoacs/logs/post-dft-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
if [[ "${SETVARS_COMPLETED:-}" != 1 ]]; then
  source /opt/intel/oneapi/setvars.sh > "results/smoacs/logs/oneapi-dft-${SLURM_JOB_ID}.out" 2>&1
fi
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_PMI_LIBRARY="$PWD/results/smoacs/postprocess-206/runtime/pmi/usr/lib/x86_64-linux-gnu/libpmi2.so.0"
export I_MPI_PIN=0
test -r "$I_MPI_PMI_LIBRARY"
task_probe="$PWD/results/smoacs/postprocess-206/runtime/mpi-probe-${SLURM_JOB_ID}"
mpicc -O0 experiments/smoacs/postprocess/mpi_probe.c -o "$task_probe"
srun --mpi=pmi2 --cpu-bind=cores -n "$SLURM_NTASKS" "$task_probe"
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
.venv-smoacs-dft/bin/python experiments/smoacs/postprocess/dft_run.py \
 --inventory "${SMOACS_INVENTORY:?}" --candidate-id "${SMOACS_DFT_ID:?}" \
 --root results/smoacs/postprocess-206
