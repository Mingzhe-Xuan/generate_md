"""Reuse validated reference records with fixed source priority, never energy ranking."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm for reference cache processing.')
    inventory = json.loads((args.selection/'mattersim-inventory.json').read_text())
    output = args.selection/'mattersim-relaxation'
    output.mkdir(exist_ok=False)
    roots = [Path(v)/'mattersim-relaxation' for v in (
        'results/smoacs/full128-three-methods-GPU-20261023/references/selected-316',
        'results/mattergen/gap2p5-task128-20261024/full128/references/selected-322',
        'results/smoacs/mp-public-reference/selected-261')]
    model = 'e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5'
    reused, statuses = [], Counter()
    for candidate in inventory['candidates']:
        options = []
        for priority, root in enumerate(roots):
            folder = root/Path(candidate['candidate_id']).stem
            if not (folder/'result.json').exists():
                continue
            record = json.loads((folder/'result.json').read_text())
            if record['input_sha256'] != candidate['sha256'] or record.get('checkpoint_sha256') != model:
                continue
            if record['status'] not in ('converged', 'unconverged', 'failed'):
                continue
            if record['status'] == 'converged':
                if not record['converged'] or not 0 <= record['max_filter_force'] <= .01 or digest(folder/'relaxed.vasp') != record['relaxed_sha256']:
                    raise ValueError('Corrupt reference cache.')
            options.append((record['status'] != 'converged', priority, folder, record))
        if not options:
            continue
        _, _, folder, record = min(options, key=lambda row: row[:2])
        destination = output/folder.name
        destination.mkdir()
        shutil.copy2(folder/'result.json', destination/'result.json')
        if (folder/'relaxed.vasp').exists():
            shutil.copy2(folder/'relaxed.vasp', destination/'relaxed.vasp')
        statuses[record['status']] += 1
        reused.append(dict(candidate_id=candidate['candidate_id'], source=str(folder), record_sha256=digest(folder/'result.json')))
    (args.selection/'cache-reuse.json').write_text(json.dumps(dict(reused=len(reused), statuses=dict(statuses),
        records=reused, policy='Prefer force-converged; then fixed source priority316/322/261. No energy ranking.'), indent=2)+'\n')
    print(json.dumps(dict(reused=len(reused), statuses=dict(statuses))), flush=True)


if __name__ == '__main__':
    main()
