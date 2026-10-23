#!/bin/bash
#SBATCH --job-name=alternating-SUN
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/smoacs/logs/alternating-SUN-%j.out
#SBATCH --error=results/smoacs/logs/alternating-SUN-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source experiments/smoacs/job_status.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY
export SMOACS_REFERENCE_PROXY=""
task_run="${SMOACS_AB_RUN:?}"
task_root="$task_run/postprocess"
task_selection="$task_run/references/selected-$SLURM_JOB_ID"
curl --fail --noproxy '*' --head --connect-timeout 10 --max-time 30 \
 https://materialsproject-build.s3.amazonaws.com/collections/summary/version=2026-04-13/part-00000-d0a74fbe-68f8-4a2d-9d43-67cb6a1bc912-c000.zstd.parquet >/dev/null
python3 - "$task_run" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);rows=[]
for label,base in [('AB',root/'postprocess'),('matched',root/'matched-baseline')]:
 for c in json.loads((base/'inventory-249/inventory.json').read_text())['candidates']:
  record=json.loads((base/'relaxation'/Path(c['candidate_id']).stem/'result.json').read_text())
  if record['status']=='converged':rows.append(dict(candidate_id=label+'_'+c['candidate_id'],elements=c['elements']))
(root/'reference-scope.json').write_text(json.dumps(dict(denominator=len(rows),candidates=rows))+'\n')
PY
.venv-smoacs-reference/bin/python experiments/smoacs/references/prepare.py \
 --inventory "$task_run/reference-scope.json" --output "$task_run/references" \
 --cache results/smoacs/mp-public-reference/range-cache
.venv-smoacs-postprocess/bin/python experiments/smoacs/references/convert.py \
 --selection "$task_selection" --proposal-root "$task_root" --all-selected
task_count="$(.venv-smoacs-postprocess/bin/python -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["candidates"]))' "$task_selection/mattersim-inventory.json")"
if [[ "$task_count" -gt 0 ]]; then
 .venv-smoacs-postprocess/bin/python experiments/smoacs/alternating/reference_cache.py --selection "$task_selection"
 .venv-smoacs-postprocess/bin/python experiments/smoacs/references/parallel.py --selection "$task_selection" --workers 4 --benchmark
 .venv-smoacs-postprocess/bin/python experiments/smoacs/references/parallel.py --selection "$task_selection" --workers 4 \
  --validated-benchmark "$task_selection/parallel-benchmark-$SLURM_JOB_ID"
fi
.venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py --selection "$task_selection" --proposal-root "$task_root"
.venv-smoacs-dft/bin/python experiments/smoacs/references/assess.py --selection "$task_selection" \
 --proposal-root "$task_run/matched-baseline" --output "$task_run/matched-baseline/assessment-$SLURM_JOB_ID"
.venv-smoacs-paper/bin/python experiments/smoacs/alternating/report.py --run "$task_run" \
 --sun "$task_selection/assessment-$SLURM_JOB_ID" --matched-sun "$task_run/matched-baseline/assessment-$SLURM_JOB_ID"
