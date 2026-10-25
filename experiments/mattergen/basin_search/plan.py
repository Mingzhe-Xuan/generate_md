"""Freeze a property-blind cohort and all initial algorithms before outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from policy import choose_parents,ACCEPTANCE,METHODS,ROUNDS,WIDTH


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    source=Path('results/mattergen/gap2p5-task128-20261024/full128').resolve();inventory=source/'inventory-249/inventory.json';rows=load(inventory)['candidates']
    ids=[r['candidate_id'] for r in rows]
    records={key:load(source/'relaxation'/Path(key).stem/'result.json') for key in ids}
    parents=choose_parents(ids,{k for k,v in records.items() if v['status']=='converged'})
    root=args.output.resolve()
    if not root.is_relative_to((Path.cwd()/'results/mattergen').resolve()):raise ValueError('Use task results.')
    root.mkdir(parents=True,exist_ok=False)
    original_scores=load(source/'cross-prediction/predictions.json');cases=[]
    for key in parents:
        candidate=next(r for r in rows if r['candidate_id']==key);b=records[key];raw=source/'structures'/key
        if sha(raw)!=candidate['sha256'] or b['input_sha256']!=candidate['sha256'] or sha(Path(b['relaxed_path']))!=b['relaxed_sha256']:raise ValueError('Immutable baseline changed.')
        cases.append(dict(parent_id=key,inventory=candidate,B=b,raw_path=str(raw),neutral=candidate['original_valid']=='True',
            predictions={r['stage']:r for r in original_scores if r['candidate_id']==key}))
    manifest=dict(parent_ids=parents,parent_count=16,source=str(source),source_inventory_sha256=sha(inventory),cohort_seed=20261026,
        selection='Uniform sample from original-order canonical-converged127 parents; seed20261026; no property/neutrality/old-method outcome filter.',
        cases=cases,methods=METHODS,rounds=ROUNDS,width=WIDTH,B_budget_per_parent=ROUNDS*WIDTH,
        search_seed=20261026,acceptance=ACCEPTANCE,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        primary_comparator='Same-parent canonical direct MatterSim; symmetry-released directB is an additional control.',
        protocol='Fixed ordered elements; CF only ranks post-quench candidates on CUDA; ALIGNN held out. Canonical final FixSymmetry/BFGS/ExpCellFilter fmax.01/maxstep.05/cap1600. Three16-quench strategies, actual MD/Hessian extraNFE/time reported, not claimed equalFLOPs.',
        literature=['https://arxiv.org/abs/cond-mat/9803344','https://arxiv.org/abs/1007.2003','https://arxiv.org/abs/2501.06848'])
    (root/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(parents=parents,neutral=sum(c['neutral'] for c in cases),root=str(root))),flush=True)


if __name__=='__main__':main()
