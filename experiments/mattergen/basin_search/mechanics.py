"""Persistent CUDA MatterSim mechanics with counted evaluations and soft MD."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    import numpy as np
    import torch
    import ase.constraints
    from ase.filters import Filter,ExpCellFilter
    if not hasattr(ase.constraints,'Filter'):ase.constraints.Filter=Filter
    from ase.constraints import FixSymmetry
    from ase.io import read,write
    from ase.optimize import BFGS
    from ase.md.verlet import VelocityVerlet
    from ase import units
    from mattersim.forcefield import MatterSimCalculator,potential
    from proposals import soft_directions,mutate
    smoacs=Path(__file__).resolve().parents[2]/'smoacs';sys.path[:0]=[str(smoacs),str(smoacs/'postprocess')]
    from alignn_checkpoint import TrustedCheckpointTorch
    from mattersim_support import validate_atomic_numbers,model_max_z,error_metadata
    expected=json.loads((smoacs/'postprocess/sources.json').read_text())['mattersim']['checkpoint_sha256']
    if sha(args.checkpoint)!=expected:raise ValueError('MatterSim checkpoint changed.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required.')
    original=potential.torch;potential.torch=TrustedCheckpointTorch(torch,[expected])
    try:
        with redirect_stdout(sys.stderr):calc=MatterSimCalculator(load_path=str(args.checkpoint.resolve()),device='cuda',load_training_state=False)
    finally:potential.torch=original
    if not all(v.is_cuda for v in calc.potential.model.parameters()):raise RuntimeError('MatterSim CPU fallback.')
    maximum=model_max_z(calc.potential.model);calls=[0];actual_calculate=calc.calculate
    def counted(*a,**kw):calls[0]+=1;return actual_calculate(*a,**kw)
    calc.calculate=counted;poisoned=False;mode_cache={}
    emit=lambda value:print('__AB_RPC__'+json.dumps(value,allow_nan=False),flush=True)
    emit(dict(ready=True,checkpoint_sha256=expected,device=torch.cuda.get_device_name(),torch=torch.__version__,model_max_z=maximum))
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        results=[]
        for task in request['tasks']:
            start=time.perf_counter();nfe=calls[0];source=Path(task['source']);folder=Path(task['output']);folder.mkdir(parents=True,exist_ok=False)
            (folder/'task.json').write_text(json.dumps(task,indent=2)+'\n')
            row=dict(candidate_id=task['id'],action=task['action'],input_sha256=sha(source),checkpoint_sha256=expected)
            try:
                if poisoned:raise RuntimeError('Poisoned CUDA context; terminate and inspect infrastructure.')
                with redirect_stdout(sys.stderr):
                    atoms=read(source,format='vasp');atoms.set_constraint(None);numbers=atoms.numbers.copy();validate_atomic_numbers(numbers,maximum);atoms.calc=calc
                    if task['action']=='relax':
                        energy0=float(atoms.get_potential_energy());symmetry=task.get('preserve_symmetry',True)
                        if symmetry:atoms.set_constraint(FixSymmetry(atoms))
                        filtered=ExpCellFilter(atoms);opt=BFGS(filtered,logfile=None,maxstep=.05)
                        converged=bool(opt.run(fmax=.01,steps=1600));forces=filtered.get_forces();energy=float(atoms.get_potential_energy())
                        raw=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max())
                        if not all(np.isfinite(x).all() for x in (forces,energy,atoms.positions,atoms.cell.array)):raise ValueError('Nonfinite B endpoint.')
                        atoms.set_constraint(None);matrix=atoms.get_all_distances(mic=True);np.fill_diagonal(matrix,np.inf)
                        row.update(status='converged' if converged else 'unconverged',converged=converged,optimizer_steps=opt.get_number_of_steps(),
                            maxstep=.05,step_limit=1600,preserve_symmetry=symmetry,initial_energy_eV=energy0,final_energy_eV=energy,
                            max_filter_force=float(np.linalg.norm(forces,axis=1).max()),max_raw_atomic_force=raw,
                            separated=bool(matrix.min()>=.5),num_atoms=len(atoms))
                        output=folder/'relaxed.vasp';write(output,atoms,format='vasp',direct=True,sort=False)
                        row.update(relaxed_path=str(output.resolve()),relaxed_sha256=sha(output))
                    elif task['action']=='md':
                        rng=np.random.default_rng(task['seed']);positions=atoms.positions.copy();epsilon=.01
                        if row['input_sha256'] not in mode_cache:
                            hessian=np.zeros((3*len(atoms),3*len(atoms)))
                            for column in range(3*len(atoms)):
                                i,axis=divmod(column,3);atoms.positions=positions.copy();atoms.positions[i,axis]+=epsilon;plus=atoms.get_forces().reshape(-1).copy()
                                atoms.positions=positions.copy();atoms.positions[i,axis]-=epsilon;minus=atoms.get_forces().reshape(-1).copy()
                                hessian[:,column]=-(plus-minus)/(2*epsilon)
                            atoms.positions=positions;values,modes=soft_directions(hessian,atoms.get_masses())
                            mode_cache[row['input_sha256']]=(hessian,values,modes)
                        hessian,values,modes=mode_cache[row['input_sha256']]
                        np.savez_compressed(folder/'curvature.npz',hessian=hessian,eigenvalues=values,modes=modes)
                        reference=read(task['reference'],format='vasp')
                        escaped=mutate(atoms,reference,rng,.02,collective=False,strain=.03);atoms.set_cell(escaped.cell,scale_atoms=False);atoms.positions=escaped.positions
                        velocities=np.einsum('m,mij->ij',rng.normal(size=min(4,len(modes))),modes[:4])
                        mass=atoms.get_masses();velocities-=np.average(velocities,axis=0,weights=mass)
                        atoms.set_velocities(velocities);K=atoms.get_kinetic_energy()
                        if K<=0:raise ValueError('Zero MD kinetic energy.')
                        target=1.5*(len(atoms)-1)*units.kB*task['temperature'];atoms.set_velocities(velocities*np.sqrt(target/K));atoms.calc=calc
                        dynamics=VelocityVerlet(atoms,.5*units.fs,logfile=None);trace=[];total0=float(atoms.get_total_energy());initial_positions=atoms.positions.copy()
                        for step in range(61):
                            energy=float(atoms.get_potential_energy());total=float(atoms.get_total_energy());matrix=atoms.get_all_distances(mic=True);np.fill_diagonal(matrix,np.inf)
                            if not np.isfinite(total) or abs(total-total0)/len(atoms)>.05 or matrix.min()<.7:raise ValueError('MD drift/collision guard rejected escape.')
                            trace.append(dict(step=step,potential_eV=energy,total_eV=total,kinetic_eV=float(atoms.get_kinetic_energy())))
                            if step<60:dynamics.run(1)
                        atoms.set_momenta(None);output=folder/'escape.vasp';write(output,atoms,format='vasp',direct=True,sort=False)
                        (folder/'md-trace.json').write_text(json.dumps(trace,indent=2)+'\n')
                        row.update(status='escaped',source=str(output.resolve()),sha256=sha(output),MD_steps=60,timestep_fs=.5,temperature_K=task['temperature'],
                            eigenvalues=values.tolist(),coordinate_RMS_A=float(np.sqrt(np.mean(np.sum((atoms.positions-initial_positions)**2,axis=1)))),
                            maximum_energy_drift_eV_atom=max(abs(r['total_eV']-total0) for r in trace)/len(atoms))
                    else:raise ValueError('Unknown mechanics action.')
                    if not np.array_equal(atoms.numbers,numbers):raise ValueError('Mechanics changed species/order.')
            except Exception as error:
                metadata=error_metadata(error);row.update(status='failed',error=repr(error),traceback=traceback.format_exc(),**metadata)
                if metadata['infrastructure_failure']:poisoned=True
            row.update(NFE=calls[0]-nfe,elapsed_seconds=time.perf_counter()-start)
            (folder/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');results.append(row)
        emit(dict(results=results,total_NFE=calls[0]))


if __name__=='__main__':main()
