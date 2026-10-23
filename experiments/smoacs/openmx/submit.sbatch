#!/bin/bash
#SBATCH --job-name=cpu206-openmx
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/smoacs/logs/openmx-%j.out
#SBATCH --error=results/smoacs/logs/openmx-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
if [[ "${SETVARS_COMPLETED:-}" != 1 ]]; then
  source /opt/intel/oneapi/setvars.sh > "results/smoacs/logs/oneapi-openmx-${SLURM_JOB_ID}.out" 2>&1
fi
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 I_MPI_PIN=0
export I_MPI_PMI_LIBRARY="$PWD/results/smoacs/postprocess-206/runtime/pmi/usr/lib/x86_64-linux-gnu/libpmi2.so.0"
test -r "$I_MPI_PMI_LIBRARY"
mkdir -p results/smoacs/openmx-206/runtime
task_probe="$PWD/results/smoacs/openmx-206/runtime/mpi-probe-${SLURM_JOB_ID}"
mpicc -O0 experiments/smoacs/postprocess/mpi_probe.c -o "$task_probe"
srun --mpi=pmi2 --cpu-bind=cores -n "$SLURM_NTASKS" "$task_probe"
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
task_extra=()
if [[ -n "${SMOACS_OPENMX_REUSE:-}" ]]; then task_extra+=(--reuse-attempt "$SMOACS_OPENMX_REUSE"); fi
if [[ -n "${SMOACS_OPENMX_BASIS_LEVEL:-}" ]]; then task_extra+=(--basis-level "$SMOACS_OPENMX_BASIS_LEVEL"); fi
if [[ -n "${SMOACS_OPENMX_MESH_INCREMENT:-}" ]]; then task_extra+=(--mesh-increment "$SMOACS_OPENMX_MESH_INCREMENT"); fi
.venv-smoacs-postprocess/bin/python experiments/smoacs/openmx/run.py --candidate-id "${SMOACS_OPENMX_ID:?}" "${task_extra[@]}"
