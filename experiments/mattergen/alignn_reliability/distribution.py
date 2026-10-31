"""Audit DFT bandgap label modes of the exact frozen ALIGNN training split."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from core import GAP_SHA, sha, save


def training_reference(text):
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if not {'target', 'prediction'} <= set(fields):
        raise ValueError('Unknown published training CSV schema.')
    rows = list(reader)
    if 'id' not in fields:
        return None, dict(available=True, comparable_by_id=False, fields=fields, rows=len(rows),
            reason='Published training CSV has no IDs; do not infer correspondence from row order.')
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate published training IDs.')
    return {row['id']: float(row['target']) for row in rows}, dict(available=True, comparable_by_id=True,
        fields=fields, rows=len(rows))


def ranked_counts(values, limit=10):
    counts = Counter(float(v) for v in values)
    return [dict(value_eV=value, count=count) for value, count in
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def binned(values, width, limit=10):
    """Bins [k*width,(k+1)*width); integer boundaries are snapped within roundoff."""
    scaled = np.asarray(values, dtype=float) / width
    nearest = np.rint(scaled)
    scaled = np.where(np.abs(scaled - nearest) <= 1e-12, nearest, scaled)
    counts = Counter(np.floor(scaled).astype(np.int64).tolist())
    return [dict(lower_eV=round(k * width, 12), upper_eV=round((k + 1) * width, 12), count=n)
            for k, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def summarize(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Expected nonempty finite nonnegative bandgap labels.')
    positive = x[x > 0]

    def subset(a):
        if not len(a):
            return dict(n=0, exact_top=[], rounded_top={}, binned_top={})
        return dict(n=len(a), mean_eV=float(a.mean()),
            quantiles_eV={str(q): float(np.quantile(a, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
            exact_top=ranked_counts(a),
            rounded_top={str(decimals): ranked_counts(np.round(a, decimals)) for decimals in (2, 1)},
            binned_top={str(width): binned(a, width) for width in (.05, .1, .2)})

    return dict(n=len(x), zero_count=int(np.sum(x == 0)), zero_fraction=float(np.mean(x == 0)),
        all=subset(x), positive=subset(positive),
        target_windows=[dict(target_eV=target, tolerance_eV=tau,
            count=int(np.sum(np.abs(x - target) <= tau + 1e-12)),
            fraction=float(np.mean(np.abs(x - target) <= tau + 1e-12)))
            for target in (0., .5, 1., 1.5, 2., 2.5, 3.) for tau in (.1, .2, .5)],
        convention='Exact modes use stored floats; rounded modes use numpy rounding. '
                   'Histograms use left-closed right-open bins anchored at0, with1e-12 bin-index snapping.')


def main(root, dataset, model_zip):
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    if sha(model_zip) != GAP_SHA:
        raise ValueError('Frozen gap checkpoint changed.')
    with zipfile.ZipFile(model_zip) as archive:
        names = archive.namelist()
        ids = json.loads(archive.read(next(n for n in names if n.endswith('ids_train_val_test.json'))))
        expected = dict(id_train=60000, id_val=5000, id_test=4239)
        if {k: len(v) for k, v in ids.items()} != expected:
            raise ValueError('Unexpected split counts.')
        for key, items in ids.items():
            if len(set(items)) != len(items):
                raise ValueError('Duplicate IDs in ' + key)
        train = set(ids['id_train'])
        if train & set(ids['id_val'] + ids['id_test']) or set(ids['id_val']) & set(ids['id_test']):
            raise ValueError('Original splits overlap.')
        train_predictions = [n for n in names if n.endswith('prediction_results_train_set.csv')]
        published = None
        comparison = dict(available=False, comparable_by_id=False)
        if train_predictions:
            published, comparison = training_reference(archive.read(train_predictions[0]).decode())
    with zipfile.ZipFile(dataset) as archive:
        with archive.open('megnet.json') as f:
            data = json.load(f)
    found = {}
    for row in data:
        key = str(row['id'])
        if key in train:
            if key in found:
                raise ValueError('Duplicate dataset ID ' + key)
            found[key] = float(row['gap pbe'])
    if set(found) != train:
        raise ValueError('Not all training IDs found.')
    gaps = np.array([found[key] for key in ids['id_train']])
    stats = summarize(gaps)
    if published is not None:
        if set(published) != train:
            raise ValueError('Published training target IDs differ.')
        delta = max(abs(found[key] - published[key]) for key in train)
        if delta > 1e-6:
            raise ValueError('Published training DFT targets differ.')
        comparison.update(n=len(published), max_target_error_eV=delta)
    with gzip.open(root / 'training-gaps.tsv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t'); writer.writerow(['id', 'gap_pbe_eV'])
        writer.writerows((key, found[key]) for key in ids['id_train'])
    result = dict(passed=True, source='MEGNet cached gap pbe labels, selected by exact checkpoint id_train',
        model_zip_sha256=GAP_SHA, dataset_sha256=sha(dataset), dataset_records=len(data),
        training_ids_sha256=hashlib.sha256(json.dumps(ids['id_train']).encode()).hexdigest(),
        archive_training_target_comparison=comparison, statistics=stats,
        training_gaps_sha256=sha(root / 'training-gaps.tsv.gz'), seconds=time.perf_counter()-started,
        code_sha256=sha(__file__), slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(root / 'distribution.json', result)
    lines = ['# Current ALIGNN training bandgap labels', '',
        'Exact frozen MP-gap checkpoint training IDs; DFT labels, not model predictions.', '',
        f"Training n={stats['n']}; zero-gap n={stats['zero_count']} ({100*stats['zero_fraction']:.4f}%).", '']
    for name in ('all', 'positive'):
        part = stats[name]
        lines += [f'## {name} (n={part["n"]})', '', 'Exact modes: ' + json.dumps(part['exact_top'][:5]), '',
                  '0.1eV histogram peaks: ' + json.dumps(part['binned_top']['0.1'][:5]), '']
    (root / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, n=stats['n'], zero_count=stats['zero_count'],
        exact_modes=stats['all']['exact_top'][:3], positive_bins=stats['positive']['binned_top']['0.1'][:3])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=Path('results/smoacs/jarvis-cache/jarvis_data/megnet.json.zip'))
    p.add_argument('--model-zip', type=Path, default=Path('results/smoacs/alignn-weights/mp_gappbe_alignnn.zip'))
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm for dataset processing.')
    main(a.run.absolute(), a.dataset.absolute(), a.model_zip.absolute())
