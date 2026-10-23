"""Convert an audited public MP selection to immutable MatterSim inputs in Slurm."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_record(record, output):
    from pymatgen.core import Structure
    identifier = record['material_id']
    if not re.fullmatch(r'(mp|mvc)-[A-Za-z0-9]+', identifier):
        raise ValueError('Unexpected reference ID.')
    structure = Structure.from_dict(record['structure'])
    if not structure.is_ordered:
        raise ValueError('Disordered reference requires an explicit protocol.')
    if len(structure) != record['nsites']:
        raise ValueError('Structure site count mismatch.')
    elements = sorted(e.symbol for e in structure.composition.elements)
    if elements != sorted(record['chemsys'].split('-')):
        raise ValueError('Reference chemical system mismatch.')
    source = Path(output)/(identifier+'.vasp')
    structure.to(filename=str(source), fmt='poscar')
    return dict(candidate_id=identifier+'.vasp', source=str(source.resolve()),
                sha256=digest(source), elements=elements, num_atoms=len(structure),
                formula=structure.composition.formula,
                mp_material_id=identifier, mp_selection_hull=record['energy_above_hull'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit through Slurm.')
    selection = json.loads((args.selection/'summary.json').read_text())
    if 'published Zenodo20149017' not in selection.get('reference_filter', ''):
        raise ValueError('Require published-protocol selection, not preliminary parent-ID selection.')
    output = args.selection/'mattersim-inputs'
    output.mkdir(exist_ok=False)
    source = args.selection/'reference-structures.jsonl.gz'
    candidates, failures = [], []
    with gzip.open(source, 'rt') as stream:
        for line in stream:
            record = json.loads(line)
            try:
                candidates.append(convert_record(record, output))
            except Exception as error:
                failures.append(dict(material_id=record['material_id'], chemsys=record['chemsys'], error=repr(error)))
    if len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate converted reference IDs.')
    manifest = dict(role='MP_reference_phases', denominator=len(candidates)+len(failures),
                    candidates=candidates, conversion_failures=failures,
                    snapshot=selection['snapshot'], selection_sha256=digest(args.selection/'summary.json'),
                    reference_records_sha256=digest(source))
    (args.selection/'mattersim-inventory.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(converted=len(candidates), failures=len(failures), selection=str(args.selection))), flush=True)


if __name__ == '__main__':
    main()
