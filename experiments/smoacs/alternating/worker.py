"""Persistent MatterSim GPU worker; communicate through JSON lines."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Submit inference through Slurm.')
    import numpy as np
    import torch
    import ase.constraints
    from ase.filters import Filter,ExpCellFilter
    if not hasattr(ase.constraints,'Filter'):ase.constraints.Filter=Filter
    from ase.io import read,write
    from ase.constraints import FixSymmetry
    from ase.optimize import BFGS
    from mattersim.forcefield import MatterSimCalculator,potential
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    from alignn_checkpoint import TrustedCheckpointTorch
    expected=json.loads((root/'postprocess/sources.json').read_text())['mattersim']['checkpoint_sha256']
    if digest(args.checkpoint)!=expected:raise ValueError('MatterSim checkpoint mismatch.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required.')
    old=potential.torch;potential.torch=TrustedCheckpointTorch(torch,[expected])
    try:
        with redirect_stdout(sys.stderr):
            calc=MatterSimCalculator(load_path=str(args.checkpoint.resolve()),device='cuda',load_training_state=False)
    finally:potential.torch=old
    if not all(p.is_cuda for p in calc.potential.model.parameters()):raise RuntimeError('NonCUDA calculator.')
    print(json.dumps(dict(ready=True,checkpoint_sha256=expected,torch=torch.__version__)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        results=[]
        for task in request['tasks']:
            source=Path(task['source']);folder=Path(task['output']);folder.mkdir(parents=True,exist_ok=True)
            target=folder/'result.json';sha=digest(source)
            if target.exists():
                row=json.loads(target.read_text())
                if row['input_sha256']!=sha:raise ValueError('Stale B cache.')
                results.append(row);continue
            row=dict(candidate_id=task['candidate_id'],input_sha256=sha,checkpoint_sha256=expected,status='running')
            started=time.perf_counter()
            try:
                with redirect_stdout(sys.stderr):
                    atoms=read(source,format='vasp');numbers=atoms.numbers.copy();atoms.calc=calc
                    initial=float(atoms.get_potential_energy());atoms.set_constraint(FixSymmetry(atoms))
                    filtered=ExpCellFilter(atoms);optimizer=BFGS(filtered,logfile=None)
                    clocks=[];optimizer.attach(lambda:clocks.append(time.perf_counter()),interval=1)
                    converged=bool(optimizer.run(fmax=.01,steps=400))
                    forces=filtered.get_forces();energy=float(atoms.get_potential_energy())
                    if not all(np.isfinite(x).all() for x in (forces,energy,atoms.cell.array,atoms.positions)):
                        raise ValueError('Nonfinite B result.')
                    if not np.array_equal(numbers,atoms.numbers):raise ValueError('B changed atom order/composition.')
                    atoms.set_constraint(None);write(folder/'relaxed.vasp',atoms,format='vasp',direct=True,sort=False)
                row.update(status='converged' if converged else 'unconverged',converged=converged,
                    optimizer_steps=optimizer.get_number_of_steps(),initial_energy_eV=initial,final_energy_eV=energy,
                    max_filter_force=float(np.linalg.norm(forces,axis=1).max()),
                    max_raw_atomic_force=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max()),
                    relaxed_path=str((folder/'relaxed.vasp').resolve()),relaxed_sha256=digest(folder/'relaxed.vasp'),
                    step_seconds=np.diff(clocks).tolist(),num_atoms=len(atoms))
            except Exception as error:row.update(status='failed',error=repr(error))
            row['elapsed_seconds']=time.perf_counter()-started
            target.write_text(json.dumps(row,indent=2)+'\n');results.append(row)
        print(json.dumps(dict(results=results)),flush=True)


if __name__=='__main__':main()
