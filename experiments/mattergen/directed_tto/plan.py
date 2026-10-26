"""Freeze a new neutral-only16 before viewing directional outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from settings import choose,SEED,FAMILIES,ARMS,SCALES,POLICY


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--search-seed',type=int,default=SEED);p.add_argument('--frozen-plan',type=Path);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    source=Path('results/mattergen/gap2p5-task128-20261024/full128').resolve();inventory=load(source/'inventory-249/inventory.json');rows=inventory['candidates']
    ids=[r['candidate_id'] for r in rows];records={i:load(source/'relaxation'/Path(i).stem/'result.json') for i in ids}
    parents=choose(ids,{r['candidate_id'] for r in rows if r['original_valid']=='True'},{i for i,b in records.items() if b['status']=='converged'})
    if args.frozen_plan:
        old=load(args.frozen_plan)
        parents=old['parent_ids']
        if len(parents)!=16 or len(set(parents))!=16 or parents!=[i for i in ids if i in parents] or old['policy']!=POLICY or old['source_inventory_sha256']!=sha(source/'inventory-249/inventory.json'):raise ValueError('Frozen cohort changed.')
        old_cases={c['parent_id']:c for c in old['cases']}
        if any(records[i]['status']!='converged' or records[i]['relaxed_sha256']!=old_cases[i]['B']['relaxed_sha256'] for i in parents):raise ValueError('Frozen baseline changed.')
    scores=load(source/'cross-prediction/predictions.json');cases=[]
    for parent in parents:
        candidate=next(r for r in rows if r['candidate_id']==parent);b=records[parent];raw=source/'structures'/parent
        if sha(raw)!=candidate['sha256'] or b['input_sha256']!=candidate['sha256'] or sha(b['relaxed_path'])!=b['relaxed_sha256']:raise ValueError('Original source changed.')
        cases.append(dict(parent_id=parent,inventory=candidate,B=b,raw_path=str(raw),neutral=True,predictions={r['stage']:r for r in scores if r['candidate_id']==parent}))
    root=args.output.resolve()
    if not root.is_relative_to((Path.cwd()/'results/mattergen').resolve()):raise ValueError('Use own task results.')
    root.mkdir(parents=True,exist_ok=False)
    plan=dict(parent_ids=parents,cases=cases,cohort_seed=SEED,search_seed=args.search_seed,source_inventory_sha256=sha(source/'inventory-249/inventory.json'),parent_count=16,
        selection='Uniform16 original-order strict-neutral AND canonical-B-converged parents; no property/old-method outcome filter. New cohort, not old16.',
        families=FAMILIES,arms=ARMS,scales=SCALES,policy=POLICY,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        objective='Pure CF bandgap squared error, frozen ordered species, GPU gradients. Formation/force/energy/volume are final feasibility guards. Final full canonical B for every proposal.',
        future='Direction benchmark is a stage, not goal completion. Follow with feedback gradient optimization, independent seed, heldout ALIGNN and perturbation retention checks.')
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(parents=parents,root=str(root))),flush=True)


if __name__=='__main__':main()
