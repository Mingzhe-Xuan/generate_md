#!/bin/bash
#SBATCH --job-name=smoacs-post-env
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/smoacs/logs/post-install-%j.out
#SBATCH --error=results/smoacs/logs/post-install-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
export OMP_NUM_THREADS=4
task_stage="${SMOACS_STAGE:?Set mattersim or dft}"
case "$task_stage" in
  mattersim) task_python="$PWD/.venv-smoacs-postprocess/bin/python" ;;
  dft) task_python="$PWD/.venv-smoacs-dft/bin/python" ;;
  *) exit 2 ;;
esac
test -x "$task_python"
# Explicitly test the compute node's direct route; login loopback is not reused.
curl --head --fail --silent --show-error --connect-timeout 10 --max-time 30 https://pypi.org/simple/ >/dev/null
"$PWD/.venv-smoacs-paper/bin/python" -m pip --python "$task_python" install \
 --no-index --find-links results/smoacs/blackwell-wheels pip==24.3.1
if [[ "$task_stage" == mattersim ]]; then
  "$task_python" -m pip install --no-index --find-links results/smoacs/blackwell-wheels \
    --find-links results/smoacs/direct-wheels \
    "$PWD/results/smoacs/blackwell-wheels/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
fi
"$task_python" -m pip install --prefer-binary \
 --find-links results/smoacs/blackwell-wheels --find-links results/smoacs/direct-wheels \
 --find-links results/smoacs/postprocess-206/wheels \
 --find-links results/smoacs/postprocess-206/dft-wheels \
 --extra-index-url https://download.pytorch.org/whl/cu128 \
 -r "experiments/smoacs/postprocess/requirements-${task_stage}.txt"
if [[ "$task_stage" == mattersim ]]; then
  printf '%s  %s\n' 830a99afbd6de32061924d473c711f1176ede13ad39cf1351f9a8dcdd67e71af \
    results/smoacs/postprocess-206/downloads/mattersim-d458f54.tar.gz | sha256sum -c -
  task_source="$PWD/results/smoacs/postprocess-206/runtime/mattersim-d458f54"
  mkdir "$task_source"
  tar -xzf results/smoacs/postprocess-206/downloads/mattersim-d458f54.tar.gz -C "$task_source"
  export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0+git.d458f54
  "$task_python" -m pip install --no-deps --no-build-isolation "$task_source"
fi
"$task_python" -m pip check
"$task_python" -m pip freeze --all > "results/smoacs/postprocess-206/${task_stage}-freeze-${SLURM_JOB_ID}.txt"
