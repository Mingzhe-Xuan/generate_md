"""MatterSim 5M variable-cell relaxation of immutable CPU206 candidates."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from inventory import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--candidate-id', nargs='+')
    parser.add_argument('--require-converged', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit through Slurm.')
    sources = json.loads(Path(__file__).with_name('sources.json').read_text())
    if sha256(args.checkpoint) != sources['mattersim']['checkpoint_sha256']:
        raise ValueError('Official 5M checkpoint hash mismatch.')
    import numpy as np
    import torch
    import ase.constraints
    from ase.filters import Filter
    if not hasattr(ase.constraints, 'Filter'):
        ase.constraints.Filter = Filter
    from ase.io import read, write
    from ase.constraints import FixSymmetry
    from ase.filters import ExpCellFilter
    from ase.optimize import BFGS
    from mattersim.forcefield import MatterSimCalculator
    from mattersim.forcefield import potential
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from alignn_checkpoint import TrustedCheckpointTorch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU relaxation required; refusing implicit CPU fallback.')
    original = potential.torch
    potential.torch = TrustedCheckpointTorch(torch, [sha256(args.checkpoint)])
    try:
        calculator = MatterSimCalculator(load_path=str(args.checkpoint.resolve()),
                                        device='cuda', load_training_state=False)
    finally:
        potential.torch = original
    if not all(p.is_cuda for p in calculator.potential.model.parameters()):
        raise RuntimeError('MatterSim parameters are not on CUDA.')
    inventory = json.loads(args.inventory.read_text())
    candidates = inventory['candidates']
    if args.candidate_id:
        known = {c['candidate_id'] for c in candidates}
        if set(args.candidate_id)-known:
            raise ValueError('Unknown pilot candidate ID.')
        candidates = [c for c in candidates if c['candidate_id'] in args.candidate_id]
    candidates = candidates[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(job_id=os.environ['SLURM_JOB_ID'], device=torch.cuda.get_device_name(),
                       torch=torch.__version__, checkpoint_sha256=sha256(args.checkpoint),
                       source_versions=sources, inventory_sha256=sha256(args.inventory),
                       steps=400, fmax=0.01, optimizer='BFGS', filter='ExpCellFilter',
                       constrain_symmetry=True, total_denominator=inventory.get('denominator', len(candidates)),
                       input_role=inventory.get('role', 'cpu206_proposals'))
    worker = os.environ.get('SMOACS_RELAX_WORKER_ID')
    if worker is not None:
        environment['worker_id'] = worker
    suffix = '-worker-'+worker if worker is not None else ''
    (args.output/('environment-'+os.environ['SLURM_JOB_ID']+suffix+'.json')).write_text(json.dumps(environment, indent=2)+'\n')
    for candidate in candidates:
        name = candidate['candidate_id']
        out = args.output/Path(name).stem
        result_path = out/'result.json'
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous['input_sha256'] != candidate['sha256']:
                raise ValueError('Existing candidate record has different provenance.')
            continue
        if sha256(candidate['source']) != candidate['sha256']:
            raise ValueError('Immutable input structure changed: '+name)
        out.mkdir(exist_ok=True)
        record = dict(candidate_id=name, input_sha256=candidate['sha256'],
                      status='running', converged=False, checkpoint_sha256=environment['checkpoint_sha256'])
        try:
            atoms = read(candidate['source'], format='vasp')
            atoms.calc = calculator
            initial_energy = float(atoms.get_potential_energy())
            initial_volume = float(atoms.get_volume())
            atoms.set_constraint(FixSymmetry(atoms))
            cell_filter = ExpCellFilter(atoms)
            optimizer = BFGS(cell_filter, logfile=str(out/'relax.log'), trajectory=str(out/'relax.traj'))
            converged = bool(optimizer.run(fmax=0.01, steps=400))
            filter_forces = cell_filter.get_forces()
            forces = atoms.get_forces(apply_constraint=False)
            stress = atoms.get_stress()
            energy = float(atoms.get_potential_energy())
            if not all(np.isfinite(x).all() for x in [filter_forces, forces, stress, energy, atoms.cell.array, atoms.positions]):
                raise ValueError('Non-finite relaxation result.')
            atoms.set_constraint(None)
            write(out/'relaxed.vasp', atoms, format='vasp', direct=True, sort=False)
            record.update(status='converged' if converged else 'unconverged', converged=converged,
                          optimizer_steps=optimizer.get_number_of_steps(),
                          initial_energy_eV=initial_energy, final_energy_eV=energy,
                          final_energy_per_atom_eV=energy/len(atoms), num_atoms=len(atoms),
                          initial_volume_A3=initial_volume, final_volume_A3=float(atoms.get_volume()),
                          max_filter_force=float(np.linalg.norm(filter_forces, axis=1).max()),
                          max_raw_atomic_force=float(np.linalg.norm(forces, axis=1).max()),
                          stress_eV_A3=stress.tolist(),
                          relaxed_path=str((out/'relaxed.vasp').resolve()),
                          relaxed_sha256=sha256(out/'relaxed.vasp'))
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            (out/'traceback.txt').write_text(traceback.format_exc())
        result_path.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    print('RELAXATION_RECORDS_COMPLETE', len(candidates), flush=True)
    if args.require_converged:
        results = [json.loads((args.output/Path(c['candidate_id']).stem/'result.json').read_text()) for c in candidates]
        if not all(r['status'] == 'converged' for r in results):
            raise RuntimeError('Pilot relaxation did not converge for every input.')


if __name__ == '__main__':
    main()
