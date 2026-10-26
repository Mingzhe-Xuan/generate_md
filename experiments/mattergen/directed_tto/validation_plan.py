"""Freeze all remaining48 eligible parents and seed27 before their new ray outcomes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from validation_policy import VALIDATION_POLICY,heldout_ids
from probe import load,save,sha,code_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Cohort batch processing must use Slurm.')
    pilot=a.pilot.resolve();old=load(pilot/'plan.json')
    if not (pilot/'report.json').is_file() or not load(pilot/'direction.complete.json')['completed']:raise ValueError('Need completed original pilot.')
    source=Path('results/mattergen/gap2p5-task128-20261024/full128').resolve();inventory_path=source/'inventory-249/inventory.json'
    if sha(inventory_path)!=old['source_inventory_sha256']:raise ValueError('Original source inventory changed.')
    rows=load(inventory_path)['candidates'];ids=[r['candidate_id'] for r in rows]
    B={p:load(source/'relaxation'/Path(p).stem/'result.json') for p in ids}
    selected=heldout_ids(ids,{r['candidate_id'] for r in rows if r['original_valid']=='True'},{p for p,b in B.items() if b['status']=='converged'},old['parent_ids'])
    scores=load(source/'cross-prediction/predictions.json');cases=[]
    for parent in selected:
        candidate=next(r for r in rows if r['candidate_id']==parent);b=B[parent];raw=source/'structures'/parent
        if sha(raw)!=candidate['sha256'] or b['input_sha256']!=candidate['sha256'] or sha(b['relaxed_path'])!=b['relaxed_sha256']:raise ValueError('Original heldout geometry changed.')
        cases.append(dict(parent_id=parent,inventory=candidate,B=b,raw_path=str(raw),neutral=True,predictions={r['stage']:r for r in scores if r['candidate_id']==parent}))
    root=a.run.resolve()
    if not root.is_relative_to((Path.cwd()/'results/mattergen').resolve()):raise ValueError('Use task results.')
    root.mkdir(parents=True,exist_ok=False)
    result=dict(policy=VALIDATION_POLICY,parent_ids=selected,cases=cases,pilot_ids=old['parent_ids'],pilot_root=str(pilot),
        pilot_plan_sha256=sha(pilot/'plan.json'),pilot_report_sha256=sha(pilot/'report.json'),source_inventory_sha256=sha(inventory_path),
        search_seed=20261027,native_math_sha256=code_hashes(),planning_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        selection='ALL48 remaining strict-neutral AND originalcanonicalB-converged parents, original128order, no property/old-method outcome screening.',
        inference='Primary population is new48, not pooled64 or16pilot. Plain method and acceptance were selected during pilot development; no new48 directional outcomes known at freeze.')
    save(root/'validation-plan.json',result)
    print(json.dumps(dict(parents=selected,count=len(selected),search_seed=20261027,primary=VALIDATION_POLICY['primary_selection'],plan_sha256=sha(root/'validation-plan.json'))),flush=True)


if __name__=='__main__':main()
