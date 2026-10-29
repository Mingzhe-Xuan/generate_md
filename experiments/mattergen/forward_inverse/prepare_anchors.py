"""Property-blind, reproducible GSS anchor selection under Slurm."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from csp_model import sha


def symmetric_cell(cell):
    values, vectors = np.linalg.eigh(cell @ cell.T)
    if values.min() <= 0 or np.linalg.det(cell) <= 0:
        raise ValueError('Invalid lattice orientation/metric.')
    return (vectors * np.sqrt(values)) @ vectors.T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--systems', default='Si,C,TiO2,CaTiO3')
    parser.add_argument('--per-system', type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use Slurm.')
    from pymatgen.core import Structure
    dataset = Path('results/mattergen/csp-benchmarks-20261029').resolve()
    complete = json.loads((dataset/'complete.json').read_text())
    if not complete['passed'] or complete['gss_reference_structures'] != 4919:
        raise ValueError('Unverified reference dataset.')
    refs = dataset/'gss/gss_reference_structures'
    with (refs/'index.csv').open(newline='') as stream:
        index = list(csv.DictReader(stream))
    systems = sorted(set(row['system'] for row in index)) if args.systems == 'all' else args.systems.split(',')
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=False)
    selected, examined = [], []
    for system in systems:
        candidates = [row for row in index if row['system'] == system]
        candidates.sort(key=lambda row: hashlib.sha256(('20261052|'+system+'|'+row['file']).encode()).hexdigest())
        chosen = 0
        for row in candidates:
            path = refs/system/row['file']
            try:
                structure = Structure.from_file(path, primitive=False)
                numbers = np.array(structure.atomic_numbers, dtype=int)
                reason = None
                if not structure.is_ordered or not 2 <= len(structure) <= 12:
                    reason = 'Requires ordered structure with2..12atoms for this local-response study.'
                elif numbers.max() > 94:
                    reason = 'Outside current MatterSim element encoder.'
                elif not np.isfinite(structure.frac_coords).all() or structure.volume <= 0:
                    reason = 'Invalid geometry.'
                examined.append(dict(system=system, file=row['file'], num_atoms=len(structure), reason=reason))
                if reason:
                    continue
                cell = symmetric_cell(structure.lattice.matrix)
                selected.append(dict(id=system+f'-{chosen:02d}', system=system, source=str(path), source_sha256=sha(path),
                    num_atoms=len(numbers), numbers=numbers.tolist(), cell=cell.tolist(), frac=structure.frac_coords.tolist(),
                    original_cell=structure.lattice.matrix.tolist(), rss_frequency=int(row['n_rss_proposals'])))
                chosen += 1
                if chosen == args.per_system:
                    break
            except Exception as error:
                examined.append(dict(system=system, file=row['file'], error=repr(error)))
        if chosen != args.per_system:
            raise ValueError('Not enough eligible references for '+system)
    result = dict(seed=20261052, selected=selected, examined=examined, systems=systems,
        per_system=args.per_system, num_atoms=[2,12], selection='Hash order with fixed input-domain eligibility only; no energy/property/score ranking.',
        dataset_complete_sha256=sha(dataset/'complete.json'), index_sha256=sha(refs/'index.csv'),
        checkpoint_training_overlap='Unknown; mechanism study is not an unseen-CSP benchmark.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), job_id=os.environ['SLURM_JOB_ID'])
    (root/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
