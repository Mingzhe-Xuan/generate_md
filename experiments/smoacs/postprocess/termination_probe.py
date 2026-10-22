"""Exercise scoped termination on real owned Slurm children, not scientific outcomes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import psutil
from safe_custodian import terminate_launchers


def main():
    job_id = os.environ['SLURM_JOB_ID']
    root = Path('results/smoacs/postprocess-206')/('termination-probe-'+job_id)
    root = root.resolve()
    target, protected = root/'vasp-stage', root/'protected-stage'
    target.mkdir(parents=True)
    protected.mkdir()
    fixture = Path('results/smoacs/vasp-capability-221')
    for name in ['POTCAR','POSCAR']:
        shutil.copyfile(fixture/name, target/name)
    (target/'INCAR').write_text('ENCUT=520\nEDIFF=1e-10\nNSW=0\nISMEAR=0\nSIGMA=0.05\nLWAVE=.FALSE.\nLCHARG=.FALSE.\n')
    (target/'KPOINTS').write_text('Termination fixture\n0\nGamma\n8 8 8\n0 0 0\n')
    prefix = ['srun','--exclusive','--mpi=pmi2','--cpu-bind=cores','-n','1']
    children = []
    report = dict(job_id=job_id, passed=False)
    try:
        with (target/'stdout').open('w') as vout, (protected/'stdout').open('w') as pout:
            owned = subprocess.Popen(prefix+['/opt/vasp.5.4.4/bin/vasp_std'], cwd=target, stdout=vout, stderr=vout)
            children.append(owned)
            other = subprocess.Popen(prefix+['/bin/sleep','30'], cwd=protected, stdout=pout, stderr=pout)
            children.append(other)
            time.sleep(2)
            if owned.poll() is not None or other.poll() is not None:
                raise RuntimeError('Controlled launcher fixture exited before the test.')
            actions = terminate_launchers(target, root, job_id, psutil.Process().children(recursive=True))
            if [a['pid'] for a in actions] != [owned.pid]:
                raise RuntimeError('Termination did not select exactly the owned VASP launcher.')
            owned.wait(timeout=15)
            if other.poll() is not None:
                raise RuntimeError('Protected unrelated Slurm child was terminated.')
            report.update(passed=True, terminated_pid=owned.pid, protected_pid=other.pid,
                          protected_still_running=True, global_fallback=False)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
        (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
