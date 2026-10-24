"""Generate immutable paired branch inputs from the original four raw parents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from pymatgen.core import Structure
from policy import AMPLITUDES,displacement

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cf_templates'))
from template_policy import select_pilot


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Prepare batch inputs through Slurm.')
    source=args.source.resolve()
    output=args.output.resolve()
    if not output.is_relative_to((Path.cwd()/'results/mattergen').resolve()):
        raise ValueError('Use current task results.')
    inventory=source/'inventory-249/inventory.json'
    proposals=output/'proposals'
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['source_inventory_sha256']!=digest(inventory) or any(digest(Path(r['source']))!=r['sha256'] for r in manifest['candidates']):
            raise ValueError('Existing paired inputs changed.')
        return
    candidates=json.loads(inventory.read_text())['candidates']
    structures={}
    rows=[]
    for c in candidates:
        if c['original_valid']!='True':
            continue
        s=Structure.from_file(source/'structures'/c['candidate_id'])
        rows.append(dict(candidate_id=c['candidate_id'],eligible=True,num_atoms=len(s)))
        structures[c['candidate_id']]=s
    chosen=select_pilot(rows)
    proposals.mkdir(parents=True,exist_ok=False)
    records=[]
    parents=[]
    source_by_id={c['candidate_id']:c for c in candidates}
    for parent_index,parent in enumerate(chosen):
        key=parent['candidate_id']
        raw=source/'structures'/key
        if digest(raw)!=source_by_id[key]['sha256']:
            raise ValueError('Original source geometry changed.')
        original=structures[key]
        parents.append(dict(parent_id=key,source_sha256=digest(raw),num_atoms=len(original)))
        for branch,amplitude in enumerate((0.,)+AMPLITUDES):
            for attempt in range(100):
                seed=[42,parent_index,branch,attempt]
                delta=displacement(len(original),amplitude,seed)
                trial=Structure(original.lattice,original.species,original.cart_coords+delta,coords_are_cartesian=True,to_unit_cell=True)
                if not any(trial.get_all_neighbors(.5)):
                    break
            else:
                raise ValueError('Could not generate a collision-free paired input.')
            identifier=Path(key).stem+f'-b{branch:02d}.vasp'
            path=proposals/identifier
            trial.to(filename=str(path),fmt='poscar')
            records.append(dict(candidate_id=identifier,parent_id=key,branch_index=branch,amplitude_A=amplitude,
                draw_seed=seed,draw_attempts=attempt+1,source=str(path),sha256=digest(path),
                rms_displacement_A=float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))),max_displacement_A=float(np.linalg.norm(delta,axis=1).max())))
    manifest=dict(source_inventory_sha256=digest(inventory),parent_ids=[p['parent_id'] for p in parents],parents=parents,
        candidates=records,N=10,seed=42,amplitudes_A=list(AMPLITUDES),perturbation='Cartesian zero-mean exact per-atom RMS; per-site norm <=2RMS; same cell and ordered species.',
        minimum_periodic_distance_A=.5,selection='Nearest4/8/12/20 atoms and ID among original eligible inputs, no property or outcome selection.')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(parents=manifest['parent_ids'],paired_inputs=len(records),N=10)),flush=True)


if __name__=='__main__':
    main()
