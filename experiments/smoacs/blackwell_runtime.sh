#!/bin/bash
# Source only in the task's Blackwell Slurm jobs, after selecting task_python.
task_site="$("$task_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export CUDA_PATH="$task_site/nvidia/cuda_runtime"
export CUDA_HOME="$CUDA_PATH"
test -f "$CUDA_PATH/include/cuda_runtime.h"
task_cuda_libs=""
for task_lib in "$task_site"/nvidia/*/lib; do
  if [[ -d "$task_lib" ]]; then
    task_cuda_libs="$task_cuda_libs$task_lib:"
  fi
done
export LD_LIBRARY_PATH="$task_cuda_libs${LD_LIBRARY_PATH:-}"
export CUPY_CACHE_DIR="$PWD/results/smoacs/cupy-blackwell-cache"
mkdir -p "$CUPY_CACHE_DIR"
