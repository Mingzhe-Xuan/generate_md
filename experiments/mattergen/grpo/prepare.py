"""Exact training-label cohort for direct pretrained MatterGen GRPO."""
import argparse
from collections import Counter
import hashlib
import json
from math import gcd
from functools import reduce
import os
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np

GAP_SHA = '8e9309fcb435a88481ccbc945d16cb1343f9ac24d8198607f753ce59b687242e'
DATA_SHA = '0f5deb0beebd22131b98ab2be551b12edfed9838841e627210f564c4e83f425f'
BASE_SHA = '81668ee12afc1ee1b037f362420730de3460bfd2d36e547585fdcb911a3dfdef'
BASE_SIZE = 461370350


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def group(elements):
    counts = Counter(elements); scale = reduce(gcd, counts.values())
    return '|'.join(f'{s}:{counts[s]//scale}' for s in sorted(counts))


def split(key):
    bucket = int(hashlib.sha256(('grpo-gap25-20261031|'+key).encode()).hexdigest()[:12], 16) % 10000
    return 'train' if bucket < 8000 else ('validation' if bucket < 9000 else 'test')


def target(gap):
    value = float(gap)
    if not np.isfinite(value) or value < 0:
        raise ValueError('Invalid DFT label.')
    return abs(value - 2.5) <= .1 + 1e-12


def eligibility(numbers, cell, frac):
    z = np.asarray(numbers); c = np.asarray(cell); q = np.asarray(frac)
    if not 1 <= len(z) <= 20:
        return 'atom_count'
    if not np.all((z >= 1) & (z <= 98)):
        return 'ALIGNN_element_support'
    if c.shape != (3, 3) or q.shape != (len(z), 3) or not np.isfinite(c).all() or not np.isfinite(q).all():
        return 'geometry'
    if np.linalg.det(c) <= 1e-8:
        return 'cell_orientation_or_volume'
    return None


def main(root):
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    dataset = Path('results/smoacs/jarvis-cache/jarvis_data/megnet.json.zip')
    weights = Path('results/smoacs/alignn-weights/mp_gappbe_alignnn.zip')
    if sha(dataset) != DATA_SHA or sha(weights) != GAP_SHA:
        raise ValueError('Cached source provenance differs.')
    with zipfile.ZipFile(weights) as z:
        ids = json.loads(z.read(next(n for n in z.namelist() if n.endswith('ids_train_val_test.json'))))
    if {k: len(v) for k, v in ids.items()} != dict(id_train=60000, id_val=5000, id_test=4239):
        raise ValueError('Unexpected original split sizes.')
    partitions = {str(v): key for key, values in ids.items() for v in values}
    if len(partitions) != 69239:
        raise ValueError('Original split IDs overlap.')
    with zipfile.ZipFile(dataset) as z:
        with z.open('megnet.json') as f:
            data = json.load(f)
    from pymatgen.core import Element
    selected = []; accounting = Counter(); seen = set()
    for row in data:
        key = str(row['id'])
        if key in seen:
            raise ValueError('Duplicate source ID.')
        seen.add(key)
        if key not in partitions:
            raise ValueError('Source ID outside checkpoint splits.')
        if not target(row['gap pbe']):
            continue
        original = partitions[key]; accounting[original+'_target'] += 1
        a = row['atoms']; elements = a['elements']; numbers = [Element(e).Z for e in elements]
        c = np.asarray(a['lattice_mat'], float); q = np.asarray(a['coords'], float)
        if a.get('cartesian', False):
            q = q @ np.linalg.inv(c)
        reason = eligibility(numbers, c, q)
        if reason is not None:
            accounting[original+'_reject_'+reason] += 1
            continue
        reduced = group(elements)
        selected.append(dict(id=key, original=original, role=split(reduced), group=reduced,
            truth_eV=float(row['gap pbe']), numbers=numbers, cell=c.tolist(), frac=np.remainder(q, 1.).tolist(),
            MatterSim_supported=max(numbers) <= 94))
        accounting[original+'_eligible'] += 1
    if seen != set(partitions) or accounting['id_train_target'] != 1548:
        raise ValueError('Dataset completeness/previous target count differs.')
    rows = [r for r in selected if r['original'] == 'id_train']
    if len(rows) < 100 or len({r['role'] for r in rows}) != 3:
        raise ValueError('Insufficient training-context cohort.')
    with (root/'cohort.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    role_counts = Counter(r['role'] for r in rows)
    role_groups = {name: {r['group'] for r in rows if r['role'] == name} for name in role_counts}
    for a in role_groups:
        for b in role_groups:
            if a != b and role_groups[a] & role_groups[b]:
                raise ValueError('Grouped role leakage.')
    plan = dict(passed=True, dataset_sha256=DATA_SHA, ALIGNN_zip_sha256=GAP_SHA,
        base_checkpoint_expected_sha256=BASE_SHA, base_checkpoint_size=BASE_SIZE,
        target_eV=2.5, tolerance_eV=.1, accounting=dict(accounting), rows=len(rows), role_counts=dict(role_counts),
        group_counts={k: len(v) for k, v in role_groups.items()},
        num_atoms={role: dict(Counter(len(r['numbers']) for r in rows if r['role'] == role)) for role in role_counts},
        MatterSim_unsupported_ids=[r['id'] for r in rows if not r['MatterSim_supported']],
        cohort_sha256=sha(root/'cohort.jsonl'),
        original_val_test_target_eligible_counts={k: accounting[k+'_eligible'] for k in ('id_val', 'id_test')},
        data_use='No SFT warmup. Train-cohort atom-count contexts for online full-composition generation; '
                 'reference geometries/DFT gaps are not truths for generated structures.',
        exposure='Original ALIGNN training IDs intentionally used as requested target training source. '
                 'MatterGen pretraining overlap not removed by grouped role split. Generated-structure '
                 'ALIGNN rewards/evaluations measure surrogate adherence, not DFT validation.',
        seconds=time.perf_counter()-started, slurm_job_id=os.environ['SLURM_JOB_ID'],
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        code_sha256=sha(__file__))
    (root/'cohort-audit.json').write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: plan[k] for k in ('passed', 'accounting', 'rows', 'role_counts', 'num_atoms', 'seconds')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Dataset processing requires Slurm.')
    main(a.run.resolve())
