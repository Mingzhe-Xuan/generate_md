"""Persistent GPU MatterSim total-energy/raw-force/raw-stress service, no relaxation."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from protocol import pose_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('All model computation through Slurm.')
    import numpy as np
    import torch
    from ase import Atoms
    from mattersim.forcefield import MatterSimCalculator,potential
    source=Path(__file__).resolve().parents[2]/'smoacs'
    sys.path[:0]=[str(source),str(source/'postprocess')]
    from alignn_checkpoint import TrustedCheckpointTorch
    from mattersim_support import model_max_z,validate_atomic_numbers,error_metadata
    expected=json.loads((source/'postprocess/sources.json').read_text())['mattersim']['checkpoint_sha256']
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=expected:
        raise ValueError('MatterSim checkpoint changed.')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU force field required.')
    old=potential.torch
    try:
        potential.torch=TrustedCheckpointTorch(torch,[expected])
        with redirect_stdout(sys.stderr):
            calc=MatterSimCalculator(load_path=str(args.checkpoint.resolve()),device='cuda',load_training_state=False)
    finally:
        potential.torch=old
    if not all(p.is_cuda for p in calc.potential.model.parameters()):
        raise RuntimeError('Force model is not on CUDA.')
    calc.potential.model.eval()
    maximum=model_max_z(calc.potential.model)
    ready=dict(ready=True,checkpoint_sha256=expected,model_max_z=maximum,device=torch.cuda.get_device_name(),
        torch=torch.__version__,force_units='eV/Angstrom',stress_units='eV/Angstrom^3',energy='total eV',constraints=False)
    print('__AB_RPC__'+json.dumps(ready),flush=True)
    poisoned=None
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):
            break
        rows=[]
        for task in request['tasks']:
            started=time.perf_counter()
            row=dict(candidate_id=task['candidate_id'],pose_sha256=pose_hash(task),checkpoint_sha256=expected)
            try:
                if row['pose_sha256']!=task['pose_sha256']:
                    raise ValueError('Force request pose hash mismatch.')
                validate_atomic_numbers(task['numbers'],maximum)
                if poisoned:
                    raise RuntimeError('Poisoned CUDA context after '+poisoned)
                cell=np.asarray(task['cell'],dtype=float)
                frac=np.asarray(task['frac'],dtype=float)
                if not np.isfinite(cell).all() or not np.isfinite(frac).all() or np.linalg.det(cell)<=1e-6:
                    raise ValueError('Invalid requested geometry.')
                with redirect_stdout(sys.stderr):
                    atoms=Atoms(numbers=task['numbers'],cell=cell,scaled_positions=frac,pbc=True)
                    atoms.calc=calc
                    energy=float(atoms.get_potential_energy())
                    force=atoms.get_forces(apply_constraint=False)
                    stress=atoms.get_stress(voigt=False,apply_constraint=False)
                if not all(np.isfinite(v).all() for v in (energy,force,stress)):
                    raise ValueError('Nonfinite physical force response.')
                if force.shape!=(len(atoms),3) or stress.shape!=(3,3):
                    raise ValueError('Invalid physical force/stress shape.')
                row.update(status='completed',energy_eV=energy,forces_eV_A=force.tolist(),stress_eV_A3=stress.tolist())
            except Exception as error:
                detail=error_metadata(error)
                row.update(status='failed',error=repr(error),**detail)
                if detail['error_type']=='cuda_context_error':
                    poisoned=task['candidate_id']
            row['elapsed_seconds']=time.perf_counter()-started
            rows.append(row)
        print('__AB_RPC__'+json.dumps(dict(results=rows),allow_nan=False),flush=True)


if __name__=='__main__':
    main()
