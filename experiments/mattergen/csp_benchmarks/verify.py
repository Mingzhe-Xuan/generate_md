"""Slurm dataset integrity/schema inventory and representative CIF parsing only."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import traceback

from integrity import extract_checked, hashes, verify_asset


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scan_csv(path):
    ids = Counter()
    first = last = None
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not {'material_id', 'cif', 'pretty_formula'}.issubset(fields or []):
            raise ValueError('Unexpected MP20 schema.')
        for row in reader:
            if None in row or not row['material_id'] or not row['cif']:
                raise ValueError('Malformed MP20 row.')
            ids[row['material_id']] += 1
            if first is None:
                first = row
            last = row
    if first is None:
        raise ValueError('Empty split.')
    return dict(rows=sum(ids.values()), unique_ids=len(ids), duplicate_ids={k: v for k, v in ids.items() if v > 1},
                columns=fields), set(ids), [first, last]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use Slurm for dataset verification.')
    import numpy as np
    from pymatgen.core import Structure
    started = time.perf_counter()
    root = args.root.resolve()
    here = Path(__file__).resolve().parent
    try:
        source = json.loads((here / 'sources.json').read_text())
        downloaded = json.loads((root / 'download-complete.json').read_text())
        if downloaded['source'] != source:
            raise ValueError('Source manifest changed after download.')
        files = {spec['path']: verify_asset(root / spec['path'], spec) for spec in source['assets']}
        save(root / 'file-manifest.json', files)
        extract_checked(root / 'downloads/gss_reference_structures.tar.gz', root / 'gss')
        refs = root / 'gss/gss_reference_structures'
        with (refs / 'index.csv').open(newline='') as stream:
            index = list(csv.DictReader(stream))
        indexed = set()
        counts = Counter()
        for row in index:
            path = (refs / row['system'] / row['file']).resolve()
            if not path.is_relative_to(refs.resolve()) or not path.is_file() or path in indexed:
                raise ValueError('Invalid or duplicated GSS index entry.')
            if int(row['n_rss_proposals']) < 1:
                raise ValueError('Invalid RSS proposal count.')
            indexed.add(path)
            counts[row['system']] += 1
        if indexed != set(p.resolve() for p in refs.rglob('*.cif')) or dict(counts) != source['gss_expected_counts']:
            raise ValueError('GSS index/reference file mismatch.')
        save(root / 'gss-summary.json', dict(total=len(index), systems=dict(counts), reference_path=str(refs),
             index_sha256=hashes(refs / 'index.csv')['sha256'], model_energy='Author MatterSim reference minima; not DFT/hull validation.'))
        probes = root / 'structure-probes'
        probes.mkdir(exist_ok=True)
        parse_records = []
        def parse(cif, name):
            path = probes / (name + '.cif')
            path.write_text(cif)
            structure = Structure.from_str(cif, fmt='cif')
            if not structure.is_ordered or not np.isfinite(structure.frac_coords).all() or not np.isfinite(structure.volume) or structure.volume <= 0:
                raise ValueError('Invalid representative CIF: ' + name)
            parse_records.append(dict(name=name, num_atoms=len(structure), formula=structure.composition.formula,
                volume_A3=structure.volume, sha256=hashes(path)['sha256']))
        for system in counts:
            path = sorted((refs / system).glob('*.cif'))[0]
            parse(path.read_text(), 'gss-' + system)
        splits = {}
        id_sets = {}
        for split in ('train', 'val', 'test'):
            record, ids, examples = scan_csv(root / 'mp20' / (split + '.csv'))
            splits[split] = record
            id_sets[split] = ids
            for i, example in enumerate(examples):
                parse(example['cif'], f'mp20-{split}-{i}-{example["material_id"]}')
        pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]
        overlaps = {a + '/' + b: sorted(id_sets[a] & id_sets[b]) for a, b in pairs}
        save(root / 'mp20-summary.json', dict(splits=splits, total=sum(r['rows'] for r in splits.values()),
             cross_split_id_overlaps=overlaps, checkpoint_training_overlap='UNKNOWN: current CSP trained on Alex-MP20; no exact training-manifest audit.'))
        save(root / 'structure-probes.json', dict(scope='One GSS CIF/system and first/last MP20 CIF/split; not full-dataset structure validation.', results=parse_records))
        complete = dict(passed=True, job_id=os.environ['SLURM_JOB_ID'], seconds=time.perf_counter()-started,
            source=source, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            module_files={p.name: hashes(p)['sha256'] for p in here.iterdir() if p.is_file()},
            gss_reference_structures=len(index), gss_systems=len(counts), mp20_rows={k: v['rows'] for k, v in splits.items()},
            cif_probes=len(parse_records), full_cif_validation=False, checkpoint_overlap_audited=False,
            inference_or_evaluation_run=False, files=files)
        save(root / 'complete.json', complete)
        archive = root / 'preparation-evidence.tar.gz'
        with tarfile.open(archive, 'w:gz') as handle:
            for name in ('complete.json', 'file-manifest.json', 'gss-summary.json', 'mp20-summary.json',
                         'structure-probes.json', 'structure-probes', 'download-complete.json'):
                handle.add(root / name, arcname=name)
        (root / 'preparation-evidence.tar.gz.sha256').write_text(hashes(archive)['sha256'] + '  preparation-evidence.tar.gz\n')
        print(json.dumps(complete), flush=True)
    except Exception:
        save(root / 'verification-failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
