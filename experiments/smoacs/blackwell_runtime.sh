#!/bin/bash
# Source only in the task's Blackwell Slurm jobs, after selecting task_python.
task_site="$("$task_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export CUDA_PATH="${SMOACS_CUDA_RUNTIME_ROOT:-$task_site/nvidia/cuda_runtime}"
export CUDA_HOME="$CUDA_PATH"
test -f "$CUDA_PATH/include/cuda_runtime.h"
if [[ "${SMOACS_REQUIRE_CUDA_KERNELS:-0}" == 1 ]]; then
  if [[ ! -f "$CUDA_PATH/lib/libcudadevrt.a" && ! -f "$CUDA_PATH/lib64/libcudadevrt.a" ]]; then
    printf "%s\n" "Complete CUDA runtime with libcudadevrt.a is required." >&2
    exit 2
  fi
fi
task_cuda_libs=""
for task_lib in "$CUDA_PATH/lib" "$CUDA_PATH/lib64"; do
  if [[ -d "$task_lib" ]]; then task_cuda_libs="$task_cuda_libs$task_lib:"; fi
done
for task_lib in "$task_site"/nvidia/*/lib; do
  if [[ -d "$task_lib" ]]; then
    task_cuda_libs="$task_cuda_libs$task_lib:"
  fi
done
export LD_LIBRARY_PATH="$task_cuda_libs${LD_LIBRARY_PATH:-}"
export CUPY_CACHE_DIR="$PWD/results/smoacs/cupy-blackwell-cache"
mkdir -p "$CUPY_CACHE_DIR"
