"""Copy immutable12 proposal endpoints and prune the existing MP reference scope."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from policy import selected_cases,needed


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm for batch preparation.')
    source=args.source.resolve();root=args.output.resolve();plan=load(source/'plan.json');path=source/'selection-records.json'
    if not root.is_relative_to((Path.cwd()/'results/mattergen').resolve()):raise ValueError('Use task results.')
    cases=selected_cases(load(path),plan['parent_ids']);root.mkdir(parents=True,exist_ok=False)
    pool=Path(plan['source'])/'references/selected-322';ref_inventory=pool/'mattersim-inventory.json';refs=load(ref_inventory)
    systems=[];inventory=[];audit=[]
    for case in cases:
        row=case['prediction'];b=row['relaxation'];identifier=case['candidate_id'];old=Path(b['relaxed_path'])
        if not (2.3<=row['crystalformer_gap']<=2.7 and row['crystalformer_eform']<-.5 and row['structure_valid'] is True):raise ValueError('Saved CF-joint flag differs from properties.')
        if b['status']!='converged' or not b['converged'] or b['max_filter_force']>.01 or b['max_raw_atomic_force']>.01:raise ValueError('CF-joint endpoint fails existing force gate.')
        if b['checkpoint_sha256']!='e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5':raise ValueError('MatterSim checkpoint changed.')
        if sha(old)!=b['relaxed_sha256'] or row['geometry_sha256']!=b['relaxed_sha256']:raise ValueError('Scored terminal changed.')
        original=old.parent.parent.parent
        original_inventory=original/'inventory-249/inventory.json'
        candidate=next(c for c in load(original_inventory)['candidates'] if c['candidate_id']==identifier)
        if candidate['num_atoms']!=b['num_atoms']:raise ValueError('Terminal atom count changed.')
        if sha(Path(candidate['source']))!=candidate['sha256'] or b['input_sha256']!=candidate['sha256']:raise ValueError('Original A/B provenance differs.')
        if case['parent_id'] not in refs['eligible_proposal_ids']:raise ValueError('Existing reference pool does not cover this original parent.')
        folder=root/'relaxation'/Path(identifier).stem;folder.mkdir(parents=True)
        shutil.copy2(old,folder/'relaxed.vasp');record=dict(b,relaxed_path=str(folder/'relaxed.vasp'));write(folder/'result.json',record)
        inventory.append(candidate);systems.append(candidate['elements'])
        audit.append(dict(parent_id=case['parent_id'],candidate_id=identifier,branch_index=case['branch_index'],
            prediction=row,source_inventory=str(original_inventory),source_inventory_sha256=sha(original_inventory),
            original_result=str(old.parent/'result.json'),original_result_sha256=sha(old.parent/'result.json'),
            original_terminal=str(old),terminal_sha256=sha(old)))
    (root/'inventory-249').mkdir();write(root/'inventory-249/inventory.json',dict(role='Frozen finalBest10 CFjoint noise0.30 CF20 B',denominator=12,candidates=inventory))
    selection=root/'references';selection.mkdir()
    chosen=[r for r in refs['candidates'] if needed(r['elements'],systems)]
    failures=[r for r in refs['conversion_failures'] if needed(r['chemsys'].split('-'),systems)]
    for system in systems:
        if {r['candidate_id'] for r in chosen if set(r['elements'])<=set(system)}!={r['candidate_id'] for r in refs['candidates'] if set(r['elements'])<=set(system)}:raise ValueError('Reference pruning lost a competitor.')
    pruned=dict(refs,candidates=chosen,conversion_failures=failures,denominator=len(chosen)+len(failures),proposal_denominator=12,
        eligible_proposal_ids=[r['candidate_id'] for r in inventory],parent_reference_inventory_sha256=sha(ref_inventory))
    write(selection/'mattersim-inventory.json',pruned)
    novelty=pool/'novelty-structures.jsonl.gz'
    (selection/'novelty-structures.jsonl.gz').symlink_to(novelty.resolve())
    manifest=dict(source=str(source),source_records_sha256=sha(path),parent_count=128,selected_count=12,candidates=audit,
        reference_pool=str(pool),reference_inventory_sha256=sha(ref_inventory),reference_count=len(chosen),
        novelty_source=str(novelty),novelty_source_sha256=sha(novelty),chemical_systems=systems,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        selection='Frozen final CF-only Best10 under force/Ef gate, then CFjoint; identical12 under legacy gate. Never select by SUN.',
        uniqueness='Within these12 CF-joint terminal structures in original parent order.',
        reference_policy='Same MP2026-04-13 published protocol competitors; fixed-priority validated existing converged caches, no energy ranking or new relaxations.')
    write(root/'manifest.json',manifest);print(json.dumps(dict(selected=12,references=len(chosen),output=str(root))),flush=True)


if __name__=='__main__':main()
