"""Select labeled structures outside the frozen predictor's 60000 training IDs."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np
from core import GAP_SHA, sha, save, composition, role


def main(root, dataset):
    from pymatgen.core import Element
    root.mkdir(parents=True, exist_ok=False)
    model_zip = Path('results/smoacs/alignn-weights/mp_gappbe_alignnn.zip')
    if sha(model_zip) != GAP_SHA: raise ValueError('Gap checkpoint changed.')
    with zipfile.ZipFile(model_zip) as archive:
        ids = json.loads(archive.read(next(p for p in archive.namelist() if p.endswith('ids_train_val_test.json'))))
        published = {r['id']: r for r in csv.DictReader(io.StringIO(archive.read(next(p for p in archive.namelist() if p.endswith('prediction_results_test_set.csv'))).decode()))}
        config = json.loads(archive.read(next(p for p in archive.namelist() if p.endswith('config.json'))))
    if {k: len(v) for k, v in ids.items()} != dict(id_train=60000, id_val=5000, id_test=4239):
        raise ValueError('Unexpected predictor split.')
    wanted = set(ids['id_val']+ids['id_test'])
    if len(wanted) != 9239 or wanted & set(ids['id_train']): raise ValueError('Predictor ID leakage.')
    with zipfile.ZipFile(dataset) as archive:
        with archive.open('megnet.json') as f: data = json.load(f)
    found = {}; first_keys = list(data[0])
    for row in data:
        key = str(row['id'])
        if key in wanted:
            if key in found: raise ValueError('Duplicate dataset ID: '+key)
            found[key] = row
    if set(found) != wanted: raise ValueError('Missing wanted IDs: '+str(sorted(wanted-set(found))[:10]))
    rows = []; elements = {}
    for original, key in [('val', x) for x in ids['id_val']]+[('test', x) for x in ids['id_test']]:
        source = found[key]; atoms = source['atoms']; symbols = atoms['elements']
        for symbol in symbols: elements.setdefault(symbol, Element(symbol).Z)
        numbers = [elements[e] for e in symbols]; group = composition(symbols)
        cell = np.asarray(atoms['lattice_mat'], dtype=float); coords = np.asarray(atoms['coords'], dtype=float)
        frac = coords@np.linalg.inv(cell) if atoms.get('cartesian', False) else coords
        truth = float(source['gap pbe'])
        if not np.isfinite(truth) or truth < 0 or not np.isfinite(cell).all() or not np.isfinite(frac).all():
            raise ValueError('Invalid DFT record '+key)
        if np.linalg.det(cell) <= 1e-8 or min(numbers) < 1 or max(numbers) > 98: raise ValueError('Unsupported record '+key)
        if original == 'test' and abs(float(published[key]['target'])-truth) > 1e-6:
            raise ValueError('Published DFT target mismatch '+key)
        rows.append(dict(id=key, original=original, role=role(group, original), group=group, truth=truth,
            numbers=numbers, cell=cell.tolist(), frac=np.remainder(frac, 1.).tolist()))
    with (root/'samples.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row, allow_nan=False)+'\n')
    from collections import Counter
    save(root/'prepare.json', dict(passed=True, source=str(dataset), dataset_sha256=sha(dataset),
        dataset_records=len(data), first_record_keys=first_keys, model_zip_sha256=GAP_SHA,
        model_config=config, original_split_counts={k: len(v) for k, v in ids.items()},
        predictor_training_ids_excluded=True, counts=dict(Counter(r['role'] for r in rows)),
        composition_groups=len({r['group'] for r in rows}), samples_sha256=sha(root/'samples.jsonl'),
        published_test_predictions={k: float(v['prediction']) for k, v in published.items()},
        warning='Original validation may have influenced predictor selection; test is untouched by reliability training. Equilibrium ID benchmark, not noisy/OOD certification.'))
    print(json.dumps({k: v for k, v in json.loads((root/'prepare.json').read_text()).items() if k in ('counts','dataset_records','composition_groups')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=Path('results/smoacs/jarvis-cache/jarvis_data/megnet.json.zip'))
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Use Slurm.')
    main(a.run.absolute(), a.dataset.absolute())
