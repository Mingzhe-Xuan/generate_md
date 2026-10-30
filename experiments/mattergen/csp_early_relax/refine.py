"""Uniform tight continuation of all72 coarse endpoints; preserve coarse results."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from comparison import eligible
from run import save,sha,Physics,ARMS


def continuation_pose(row):
    point=row['result']
    if point['status']!='converged':raise ValueError('This verification requires all coarse endpoints to exist.')
    return {k:point[k] for k in ('numbers','cell','frac')}


def main(source,root):
    started=time.perf_counter();root.mkdir(parents=True,exist_ok=False);(root/'inputs').mkdir()
    old=json.loads((source/'complete.json').read_text());oldplan=json.loads((source/'plan.json').read_text())
    oldrows={(r['trajectory'],r['arm']):r for r in old['relaxations']}
    if set(oldrows)!={(i,a) for i in range(24) for a in ARMS} or len(old['relaxations'])!=72:
        raise ValueError('Incomplete coarse cohort.')
    policy=dict(oldplan['policy'],fmax=.001,force_acceptance_fmax=.01,
        refinement_source=str(source),coarse_stage_fmax=.01,
        role='Uniform tighter continuation of ALL72 coarse endpoints, unchanged physical acceptance; no new sampling.')
    for name in ('complete.json','plan.json','report.json'):
        shutil.copyfile(source/name,root/('coarse-'+name))
    save(root/'plan.json',dict(policy=policy,source=str(source),source_sha256={name:sha(source/name) for name in ('complete.json','plan.json','report.json')},
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    physics=Physics(root/'physics');inputs=[];rows=[]
    try:
        for i in range(24):
            for arm in ARMS:
                oldrow=oldrows[i,arm];pose=continuation_pose(oldrow);key=f'{i:02d}-{arm}'
                path=root/'inputs'/f'{key}.npz'
                np.savez_compressed(path,**{k:np.asarray(v) for k,v in pose.items()})
                item=dict(trajectory=i,arm=arm,batch=oldrow['batch'],seed=oldrow['seed'],system=oldrow['system'],
                    path=str(path.relative_to(root)),sha256=sha(path),source_csp_sha256=oldrow['sha256'])
                inputs.append(item)
                task=dict(id=key,action='relax',output=str(root/'relaxations'/key),**pose,fmax=.001,max_steps=1600,cell_relax=True)
                point=physics.request([task])[0]
                rows.append(dict(item,result=point,force_eligible=eligible(point),coarse_result=oldrow['result']))
                save(root/'relaxations.json',rows)
                save(root/'relaxation-progress.json',dict(completed=len(rows),total=72,statuses=dict(Counter(r['result']['status'] for r in rows)),
                    force_eligible=sum(r['force_eligible'] for r in rows),NFE=physics.NFE,elapsed_seconds=time.perf_counter()-started))
                print(json.dumps(dict(trajectory=i,arm=arm,status=point['status'],steps=point.get('steps'),force_eligible=eligible(point))),flush=True)
        save(root/'complete.json',dict(passed=True,inputs=inputs,relaxations=rows,sampling_audit=old['sampling_audit'],
            native_forwards=old['native_forwards'],new_native_forwards=0,physics=physics.metadata,NFE=physics.NFE,requests=physics.requests,
            refinement_elapsed_seconds=time.perf_counter()-started,elapsed_seconds=old['elapsed_seconds']+time.perf_counter()-started,
            note='Original native sampling audit inherited with source hashes. New compute only uniformly refines all72 coarse endpoints.'))
    finally:physics.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(args.source.resolve(),args.run.resolve())
