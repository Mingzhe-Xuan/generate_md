"""Evaluate original candidates without relaxation using the pinned MatterSim5M."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'postprocess'), str(ROOT)]
from inventory import sha256
from alignn_checkpoint import TrustedCheckpointTorch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit inference through Slurm.')
    import numpy as np
    import torch
    import ase.constraints
    from ase.filters import Filter
    if not hasattr(ase.constraints, 'Filter'):
        ase.constraints.Filter = Filter
    from ase.io import read
    from mattersim.forcefield import MatterSimCalculator, potential
    from importlib.metadata import version
    sources = json.loads((ROOT/'postprocess/sources.json').read_text())
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != sources['mattersim']['checkpoint_sha256']:
        raise ValueError('Unverified MatterSim checkpoint.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no implicit CPU fallback.')
    original = potential.torch
    potential.torch = TrustedCheckpointTorch(torch, [checkpoint_hash])
    try:
        calc = MatterSimCalculator(load_path=str(args.checkpoint.resolve()), device='cuda', load_training_state=False)
    finally:
        potential.torch = original
    if not all(p.is_cuda for p in calc.potential.model.parameters()):
        raise RuntimeError('Model parameters must be on CUDA.')
    inventory = json.loads(args.inventory.read_text())
    candidates = inventory['candidates']
    if inventory['denominator'] != len(candidates) or len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('Invalid fixed inventory denominator/IDs.')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(job_id=os.environ['SLURM_JOB_ID'], inventory_sha256=sha256(args.inventory),
                    checkpoint_sha256=checkpoint_hash, device=torch.cuda.get_device_name(),
                    packages={n:version(n) for n in ('torch','mattersim','ase','numpy')},
                    method='single-point on immutable original geometry; no optimization')
    (args.output/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    rows = []
    for candidate in candidates:
        if sha256(candidate['source']) != candidate['sha256']:
            raise ValueError('Original structure changed.')
        row = dict(candidate_id=candidate['candidate_id'], input_sha256=candidate['sha256'])
        try:
            atoms = read(candidate['source'], format='vasp')
            if len(atoms) != candidate['num_atoms']:
                raise ValueError('Original atom count mismatch.')
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            if not np.isfinite(energy):
                raise ValueError('Nonfinite single-point energy.')
            row.update(status='completed', final_energy_eV=energy,
                       energy_eV_atom=energy/len(atoms), num_atoms=len(atoms))
        except Exception as error:
            row.update(status='failed', error=repr(error))
        rows.append(row)
        (args.output/'candidates.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
