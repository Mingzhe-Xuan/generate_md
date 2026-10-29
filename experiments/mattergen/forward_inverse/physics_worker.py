"""Pinned CUDA MatterSim single points and explicitly unconstrained research relaxations."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Physics work requires Slurm.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import ase.constraints
    from ase import Atoms
    from ase.filters import Filter, ExpCellFilter
    from ase.optimize import BFGS
    from ase.io import write
    if not hasattr(ase.constraints, 'Filter'):
        ase.constraints.Filter = Filter
    from mattersim.forcefield import MatterSimCalculator, potential
    source = Path(__file__).resolve().parents[2]/'smoacs'
    sys.path[:0] = [str(source), str(source/'postprocess')]
    from alignn_checkpoint import TrustedCheckpointTorch
    from mattersim_support import validate_atomic_numbers, model_max_z, error_metadata
    expected = json.loads((source/'postprocess/sources.json').read_text())['mattersim']['checkpoint_sha256']
    if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() != expected or not torch.cuda.is_available():
        raise ValueError('Pinned CUDA MatterSim required.')
    original = potential.torch
    potential.torch = TrustedCheckpointTorch(torch, [expected])
    try:
        with redirect_stdout(sys.stderr):
            calc = MatterSimCalculator(load_path=str(Path(args.checkpoint).resolve()), device='cuda', load_training_state=False)
    finally:
        potential.torch = original
    maximum = model_max_z(calc.potential.model)
    if not all(p.is_cuda for p in calc.potential.model.parameters()):
        raise RuntimeError('CPU fallback.')
    calc.potential.model.eval()
    calls = [0]
    original_calculate = calc.calculate
    def counted(*a, **kw):
        calls[0] += 1
        return original_calculate(*a, **kw)
    calc.calculate = counted
    emit = lambda obj: print('__AB_RPC__' + json.dumps(obj, allow_nan=False), flush=True)
    emit(dict(ready=True, checkpoint_sha256=expected, model_max_z=maximum, device=torch.cuda.get_device_name(), torch=torch.__version__))
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('stop'):
            break
        results = []
        for task in request['tasks']:
            started = time.perf_counter()
            before = calls[0]
            folder = None
            result = dict(id=task['id'])
            try:
                validate_atomic_numbers(task['numbers'], maximum)
                atoms = Atoms(numbers=task['numbers'], cell=task['cell'], scaled_positions=task['frac'], pbc=True)
                if not np.isfinite(atoms.positions).all() or np.linalg.det(atoms.cell.array) <= 1e-6:
                    raise ValueError('Invalid physical geometry.')
                numbers = atoms.numbers.copy()
                atoms.calc = calc
                with redirect_stdout(sys.stderr):
                    initial = float(atoms.get_potential_energy())
                    if task['action'] == 'relax':
                        folder = Path(task['output'])
                        folder.mkdir(parents=True, exist_ok=False)
                        (folder/'input.json').write_text(json.dumps(task, indent=2)+'\n')
                        filtered = ExpCellFilter(atoms) if task.get('cell_relax', True) else atoms
                        optimizer = BFGS(filtered, logfile=None, maxstep=.05)
                        converged = bool(optimizer.run(fmax=task['fmax'], steps=task['max_steps']))
                        write(folder/'relaxed.vasp', atoms, format='vasp', direct=True, sort=False)
                        result.update(status='converged' if converged else 'unconverged', converged=converged,
                            steps=optimizer.get_number_of_steps(), preserve_symmetry=False, cell_relax=task.get('cell_relax', True),
                            fmax=task['fmax'], max_steps=task['max_steps'], initial_energy_eV=initial)
                    elif task['action'] == 'point':
                        result['status'] = 'completed'
                    else:
                        raise ValueError('Unknown physics action.')
                    energy = float(atoms.get_potential_energy())
                    force = atoms.get_forces(apply_constraint=False)
                    if not np.isfinite(energy) or not np.isfinite(force).all() or not np.array_equal(numbers, atoms.numbers):
                        raise ValueError('Physical finite/species audit failed.')
                    result.update(energy_eV=energy, forces_eV_A=force.tolist(), max_raw_force=float(np.linalg.norm(force, axis=1).max()),
                        cell=atoms.cell.array.tolist(), frac=atoms.get_scaled_positions(wrap=False).tolist(), numbers=atoms.numbers.tolist())
            except Exception as error:
                result.update(status='failed', error=repr(error), traceback=traceback.format_exc(), **error_metadata(error))
            result.update(NFE=calls[0]-before, seconds=time.perf_counter()-started)
            if folder is not None:
                (folder/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            results.append(result)
        emit(dict(results=results, NFE=calls[0]))


if __name__ == '__main__':
    main()
