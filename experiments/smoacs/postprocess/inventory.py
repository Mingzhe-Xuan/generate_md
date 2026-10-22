"""Inventory immutable CPU206 structures and installed POTCAR identities in Slurm."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_files(rows, directory, expected=128):
    if len(rows) != expected:
        raise ValueError('Candidate denominator mismatch.')
    names = [row['original_fnames'] for row in rows]
    if len(set(names)) != expected or any(Path(n).name != n for n in names):
        raise ValueError('Duplicate or unsafe original candidate IDs.')
    paths = [Path(directory)/('Optimized_based_on_'+name) for name in names]
    if any(not p.is_file() for p in paths):
        raise ValueError('Missing CPU candidate structure.')
    return paths


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit inventory through Slurm.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Potcar
    from paper_dft import make_paper_maker, stage_makers
    from importlib.metadata import version
    generator = stage_makers(make_paper_maker())[0].input_set_generator
    sources = json.loads(Path(__file__).with_name('sources.json').read_text())
    csv_path = args.input/'result.csv'
    if sha256(csv_path) != sources['cpu_result_csv_sha256']:
        raise ValueError('Original CPU result CSV changed.')
    rows = list(csv.DictReader(csv_path.open()))
    paths = candidate_files(rows, args.input/'poscar', sources['expected_candidates'])
    records = []
    requested = set()
    for row, path in zip(rows, paths):
        structure = Structure.from_file(path)
        symbols = generator.get_input_set(structure, potcar_spec=True).potcar
        requested.update(symbols)
        records.append(dict(candidate_id=row['original_fnames'],
                            source=str(path.resolve()), sha256=sha256(path),
                            num_atoms=len(structure), formula=structure.composition.formula,
                            elements=sorted(str(e) for e in structure.composition.elements),
                            potcar_symbols=symbols,
                            original_gap=float(row['bandgap_onehot']),
                            original_eform=float(row['ef_onehot']),
                            original_success=row.get('success'),
                            original_valid=row.get('valid_structure')))
    available = {}
    errors = []
    potential_roots = [Path('/opt/vasp.6.4.3/testsuite/POTCARS'),
                       Path('/opt/vasp.6.4.3-wannier90/testsuite/POTCARS')]
    paths_to_scan = sorted(p for root in potential_roots for p in root.glob('POTCAR.*'))
    for path in paths_to_scan:
        try:
            # Only retain identities; never publish or commit potential contents.
            potentials = Potcar.from_file(path)
            for potential in potentials:
                functionals, symbols = potential.identify_potcar(mode='data')
                for symbol in set(symbols) & requested:
                    if 'PBE_54' in functionals:
                        available.setdefault(symbol, []).append(dict(
                            source=str(path), sha256_file=sha256(path),
                            functional_matches=functionals, symbol_matches=symbols,
                            titel=potential.TITEL))
        except Exception as exc:
            errors.append(dict(path=path.name, error=type(exc).__name__))
    args.output.mkdir(exist_ok=False, parents=True)
    report = dict(source_csv_sha256=sha256(csv_path), denominator=len(records),
                  candidates=records, required_potcar_symbols=sorted(requested),
                  available_pbe54=available,
                  missing_pbe54=sorted(requested-set(available)),
                  potential_parse_errors=errors,
                  potcar_generator=type(generator).__name__,
                  python_packages={n:version(n) for n in ['atomate2','pymatgen']},
                  source_versions=sources)
    (args.output/'inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(candidates=len(records), required_potcars=sorted(requested),
                          missing_pbe54=report['missing_pbe54']), indent=2))


if __name__ == '__main__':
    main()
