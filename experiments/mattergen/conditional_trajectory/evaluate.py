"""Posthoc ALIGNN and MatterSim single points, never called during sampling."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from metrics import geometry
from run import HERE, save, sha
from property_model import Properties
from physics_client import Physics


def main(root):
    if (root/'complete.json').exists(): raise ValueError('Refuse completed evaluation.')
    sampling = json.loads((root/'sampling.json').read_text())
    target = json.loads((root/'plan.json').read_text())['target']
    props_root = root/'alignn'; props_root.mkdir()
    properties = Properties(props_root)
    physics = Physics(root/'physics')
    rows = []; started = time.perf_counter()
    try:
        for path in sampling['paths']:
            if sha(root/path['file']) != path['sha256']: raise ValueError('Trajectory hash changed.')
            tasks = []; pending = []
            with np.load(root/path['file'], allow_pickle=False) as d:
                if len(d['step']) != 203: raise ValueError('State coverage changed.')
                for k, (numbers, cell, frac) in enumerate(zip(d['numbers'], d['cell'], d['frac'])):
                    reasons, nearest = geometry(numbers, cell, frac, 98)
                    row = dict(arm=path['arm'], parent=path['parent'], batch=path['batch'], frame=k,
                        step=int(d['step'][k]), stage=str(d['stage'][k]), t=float(d['t'][k]), target=target,
                        num_atoms=len(numbers), masked=int(np.sum(numbers == 101)), reasons=reasons,
                        minimum_pair_A=nearest, volume_per_atom_A3=float(np.linalg.det(cell)/len(numbers)),
                        gap=None, eform=None, energy_eV_atom=None, force_rms=None, max_force=None)
                    if not reasons:
                        result, _ = properties.evaluate(numbers, cell, frac@cell,
                            native=row['stage'] == 'terminal' and path['parent'] in (0, 4, 8))
                        row.update(gap=float(result['values'][0]), eform=float(result['values'][1]), graph_hash=result['graph_hash'])
                    if not reasons and np.all(numbers <= 94):
                        tasks.append(dict(id=f"{path['arm']}:{path['parent']}:{k}", action='point',
                            numbers=numbers.tolist(), cell=cell.tolist(), frac=frac.tolist()))
                        pending.append(row)
                    row['physics_status'] = 'pending' if row in pending else ('unsupported_types' if not reasons else 'geometry_or_types_excluded')
                    rows.append(row)
            if tasks:
                for result, row in zip(physics.request(tasks), pending):
                    row['physics_status'] = result['status']
                    if result['status'] == 'completed':
                        forces = np.asarray(result['forces_eV_A'])
                        row.update(energy_eV_atom=result['energy_eV']/row['num_atoms'],
                            force_rms=float(np.sqrt(np.mean(np.sum(forces**2, axis=1)))), max_force=result['max_raw_force'])
            save(root/'evaluation-progress.json', dict(paths=len(rows)//203, states=len(rows),
                valid_alignn=sum(r['gap'] is not None for r in rows), calls=properties.metadata['calls'],
                seconds=time.perf_counter()-started))
            print((root/'evaluation-progress.json').read_text(), flush=True)
        if len(rows) != 4872 or not properties.predictor.native_controls: raise ValueError('Evaluation coverage/control failure.')
        save(root/'complete.json', dict(rows=rows, sampling=sampling, alignn=properties.metadata,
            controls=properties.predictor.native_controls, physics=physics.metadata, physics_NFE=physics.NFE,
            statuses=dict(Counter(r['physics_status'] for r in rows)), seconds=time.perf_counter()-started))
    finally: physics.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Use Slurm.')
    main(args.run.absolute())
