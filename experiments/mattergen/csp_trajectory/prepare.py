"""Slurm-only property-blind GSS composition selection; no geometry conditioning."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys


def main(root):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use Slurm.')
    root.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'forward_inverse/prepare_anchors.py'),
                    '--run', str(root/'selection'), '--systems', 'all', '--per-system', '1'], check=True)
    from ase.data import chemical_symbols
    selection = json.loads((root/'selection/selection.json').read_text())
    compositions = []
    signatures = set()
    for row in selection['selected']:
        counts = Counter(row['numbers'])
        signature = tuple(sorted(counts.items()))
        if signature in signatures:
            raise ValueError('Ambiguous composition-system mapping.')
        signatures.add(signature)
        compositions.append(dict(system=row['system'], counts={chemical_symbols[z]: n for z,n in sorted(counts.items())},
            atomic_counts={str(z): n for z,n in sorted(counts.items())}, reference_sha256=row['source_sha256']))
    if len(compositions) != 12:
        raise ValueError('Expected all 12 GSS systems.')
    (root/'composition_plan.json').write_text(json.dumps(dict(compositions=compositions,
        paths_per_system=2, batch_size=8, num_batches=3, seed=20261060,
        geometry_input='Native random prior only; selected reference contributes integer composition counts.',
        selection_file='selection/selection.json'), indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
