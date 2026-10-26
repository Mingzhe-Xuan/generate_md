"""Persistent counted CUDA MatterSim points and unchanged full canonical quenches."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    import torch
    import numpy as np
    import ase.constraints
    from ase import Atoms
    from ase.filters import Filter,ExpCellFilter
    if not hasattr(ase.constraints,'Filter'):ase.constraints.Filter=Filter
    from ase.constraints import FixSymmetry
    from ase.optimize import BFGS
    from ase.io import read,write
    from ase.neighborlist import neighbor_list
    from mattersim.forcefield import MatterSimCalculator,potential
    source=Path(__file__).resolve().parents[2]/'smoacs';sys.path[:0]=[str(source),str(source/'postprocess')]
    from alignn_checkpoint import TrustedCheckpointTorch
    from mattersim_support import model_max_z,validate_atomic_numbers,error_metadata
    expected=json.loads((source/'postprocess/sources.json').read_text())['mattersim']['checkpoint_sha256']
    if sha(args.checkpoint)!=expected or not torch.cuda.is_available():raise RuntimeError('Pinned CUDA potential required.')
    old=potential.torch;potential.torch=TrustedCheckpointTorch(torch,[expected])
    try:
        with redirect_stdout(sys.stderr):calc=MatterSimCalculator(load_path=str(args.checkpoint.resolve()),device='cuda',load_training_state=False)
    finally:potential.torch=old
    if not all(p.is_cuda for p in calc.potential.model.parameters()):raise RuntimeError('CPU fallback.')
    calc.potential.model.eval();maximum=model_max_z(calc.potential.model);calls=[0];calculate=calc.calculate;poisoned=False
    def counted(*a,**kw):calls[0]+=1;return calculate(*a,**kw)
    calc.calculate=counted
    emit=lambda row:print('__AB_RPC__'+json.dumps(row,allow_nan=False),flush=True)
    emit(dict(ready=True,checkpoint_sha256=expected,model_max_z=maximum,device=torch.cuda.get_device_name(),torch=torch.__version__))
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        results=[]
        for task in request['tasks']:
            started=time.perf_counter();nfe=calls[0];row=dict(candidate_id=task['id'],action=task['action'],checkpoint_sha256=expected)
            folder=None;created=False
            try:
                if poisoned:raise RuntimeError('Poisoned CUDA context.')
                with redirect_stdout(sys.stderr):
                    if task['action']=='point':atoms=Atoms(numbers=task['numbers'],cell=task['cell'],scaled_positions=task['frac'],pbc=True)
                    elif task['action']=='relax':
                        folder=Path(task['output']);folder.mkdir(parents=True,exist_ok=False);created=True;(folder/'task.json').write_text(json.dumps(task,indent=2)+'\n')
                        row['input_sha256']=sha(task['source']);atoms=read(task['source'],format='vasp')
                    else:raise ValueError('Unknown action.')
                    atoms.set_constraint(None);validate_atomic_numbers(atoms.numbers,maximum);numbers=atoms.numbers.copy();atoms.calc=calc
                    if not np.isfinite(atoms.positions).all() or np.linalg.det(atoms.cell.array)<=1e-6:raise ValueError('Invalid pose.')
                    if task['action']=='point':
                        energy=float(atoms.get_potential_energy());force=atoms.get_forces(apply_constraint=False);stress=atoms.get_stress(voigt=False,apply_constraint=False)
                        if not all(np.isfinite(v).all() for v in (energy,force,stress)):raise ValueError('Nonfinite physical point.')
                        row.update(status='completed',energy_eV=energy,forces_eV_A=force.tolist(),stress_eV_A3=stress.tolist())
                    else:
                        initial=float(atoms.get_potential_energy());atoms.set_constraint(FixSymmetry(atoms));filtered=ExpCellFilter(atoms)
                        opt=BFGS(filtered,logfile=None,maxstep=.05);converged=bool(opt.run(fmax=.01,steps=1600))
                        filtered_force=float(np.linalg.norm(filtered.get_forces(),axis=1).max());raw=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max());energy=float(atoms.get_potential_energy())
                        if not np.isfinite([energy,filtered_force,raw]).all():raise ValueError('Nonfinite relaxation.')
                        atoms.set_constraint(None);path=folder/'relaxed.vasp';write(path,atoms,format='vasp',direct=True,sort=False)
                        row.update(status='converged' if converged else 'unconverged',converged=converged,optimizer_steps=opt.get_number_of_steps(),
                            maxstep=.05,step_limit=1600,preserve_symmetry=True,initial_energy_eV=initial,final_energy_eV=energy,max_filter_force=filtered_force,
                            max_raw_atomic_force=raw,separated=not len(neighbor_list('d',atoms,.5-1e-9)),num_atoms=len(atoms),volume_A3=atoms.get_volume(),
                            relaxed_path=str(path.resolve()),relaxed_sha256=sha(path))
                    if not np.array_equal(numbers,atoms.numbers):raise ValueError('Species changed.')
            except Exception as error:
                metadata=error_metadata(error);row.update(status='failed',error=repr(error),traceback=traceback.format_exc(),**metadata)
                if metadata['infrastructure_failure']:poisoned=True
            row.update(NFE=calls[0]-nfe,elapsed_seconds=time.perf_counter()-started)
            if created:(folder/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            results.append(row)
        emit(dict(results=results,total_NFE=calls[0]))


if __name__=='__main__':main()
