"""Summarize all128 immutable candidates without conflating unknown stability with success."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
from inventory import sha256


def count_statuses(rows, expected=128):
    if len(rows) != expected or len({r['candidate_id'] for r in rows}) != expected:
        raise ValueError('Summary denominator or candidate identity mismatch.')
    counts = Counter(r['status'] for r in rows)
    return dict(denominator=expected, counts=dict(counts),
                relaxation_converged_fraction=counts['converged']/expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit assessment through Slurm.')
    import numpy as np
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    inventory = json.loads(args.inventory.read_text())
    rows = []
    structures = []
    for candidate in inventory['candidates']:
        name = candidate['candidate_id']
        record_path = args.root/'relaxation'/Path(name).stem/'result.json'
        row = dict(candidate_id=name, status='missing', original_gap_eV=candidate['original_gap'],
                   original_eform_eV_atom=candidate['original_eform'],
                   stability_status='unknown_reference_phases_unavailable',
                   novelty_status='unknown_reference_structures_unavailable')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['input_sha256'] != candidate['sha256']:
                raise ValueError('Candidate record provenance mismatch.')
            row.update(record)
            if record['status'] in ['converged','unconverged']:
                if sha256(record['relaxed_path']) != record['relaxed_sha256']:
                    raise ValueError('Relaxed structure changed.')
                relaxed = Structure.from_file(record['relaxed_path'])
                original = Structure.from_file(candidate['source'])
                if relaxed.composition != original.composition:
                    raise ValueError('Relaxation changed candidate composition.')
                distances = relaxed.distance_matrix
                np.fill_diagonal(distances, np.inf)
                row.update(min_periodic_distance_A=float(distances.min()),
                           energy_change_eV_atom=(record['final_energy_eV']-record['initial_energy_eV'])/len(relaxed),
                           volume_ratio=record['final_volume_A3']/record['initial_volume_A3'])
                if record['status'] == 'converged':
                    structures.append((len(rows), relaxed))
        attempts = sorted((args.root/'dft'/Path(name).stem).glob('attempt-*/result.json'),
                          key=lambda p:int(p.parent.name.split('-')[-1]))
        if attempts:
            dft = json.loads(attempts[-1].read_text())
            row['dft_status'] = dft['status']
            for key in ['bandgap_eV','energy_per_atom_eV','target_gap_success']:
                if key in dft:
                    row['dft_'+key] = dft[key]
        else:
            row['dft_status'] = 'not_run'
        rows.append(row)
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5,
                               primitive_cell=True, scale=True, attempt_supercell=True)
    groups = []
    for index, structure in structures:
        for group in groups:
            if matcher.fit(structure, group[0][1]):
                group.append((index, structure))
                break
        else:
            groups.append([(index, structure)])
    for group in groups:
        for index, _ in group:
            rows[index]['unique_group_representative'] = rows[group[0][0]]['candidate_id']
    summary = count_statuses(rows, inventory['denominator'])
    summary.update(unique_relaxed_groups=len(groups),
                   dft_status_counts=dict(Counter(r['dft_status'] for r in rows)),
                   stable_fraction=None, sun_fraction=None,
                   reason='Consistent relaxed competing phases and complete MP reference structure access unavailable.',
                   inventory_sha256=sha256(args.inventory))
    out = args.root/('assessment-'+os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (out/'candidates.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
