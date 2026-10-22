#!/bin/bash
#SBATCH --job-name=vasp-capability
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=results/smoacs/logs/vasp-capability-%j.out
#SBATCH --error=results/smoacs/logs/vasp-capability-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from parent repository}"
source experiments/smoacs/job_status.sh
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_HYDRA_BOOTSTRAP=slurm
task_dir="$SLURM_SUBMIT_DIR/results/smoacs/vasp-capability-$SLURM_JOB_ID"
mkdir "$task_dir"
(
cd "$task_dir"
task_vasp="${SMOACS_VASP_EXECUTABLE:-/opt/vasp.5.4.4/bin/vasp_std}"
task_potential="${SMOACS_VASP_SI_POTCAR:-/opt/vasp.6.4.3/testsuite/POTCARS/POTCAR.kopt_si_path_scan}"
test -x "$task_vasp"
test -r "$task_potential"
cp "$task_potential" POTCAR
sha256sum "$task_vasp" POTCAR > input-sha256.txt
ldd "$task_vasp" > libraries.txt
if grep -q 'not found' libraries.txt; then exit 1; fi
cat > POSCAR <<'EOF'
Si two-atom capability test
5.43
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
EOF
cat > INCAR <<'EOF'
SYSTEM = Si capability test
ENCUT = 300
PREC = Normal
EDIFF = 1E-6
NELM = 60
ISMEAR = 0
SIGMA = 0.05
ALGO = Normal
NSW = 0
IBRION = -1
ISPIN = 1
LWAVE = .FALSE.
LCHARG = .FALSE.
LREAL = .FALSE.
EOF
cat > KPOINTS <<'EOF'
Si capability mesh
0
Gamma
2 2 2
0 0 0
EOF
mpirun -np "${SLURM_NTASKS:?}" "$task_vasp" > vasp.out 2> vasp.err
grep -q 'aborting loop because EDIFF is reached' OUTCAR
grep -q 'General timing and accounting' OUTCAR
python3 - <<'PY'
import json, math, os, re
from pathlib import Path
out = Path('OUTCAR').read_text(errors='replace')
energy = float(re.findall(r'free\s+energy\s+TOTEN\s*=\s*([-+\d.Ee]+)', out)[-1])
assert math.isfinite(energy)
version = next(line.strip() for line in out.splitlines() if 'vasp.' in line)
Path('result.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],
    version=version, mpi_ranks=int(os.environ['SLURM_NTASKS']),
    electronic_converged=True, normal_termination=True, toten_eV=energy,
    purpose='two-Si static SCF capability only; not a converged scientific benchmark'), indent=2)+'\n')
PY
)
