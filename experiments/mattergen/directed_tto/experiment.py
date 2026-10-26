"""Matched one-gradient/four-step-size trials; every scientific endpoint gets full B."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from geometry import Geometry
from settings import POLICY,FAMILIES,ARMS,SCALES
from runtime import Mechanics
from cf_gradient import GradientScorer
from probe import code_hashes,sha,load,save

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'basin_search'))
from search import point,export
from policy import physical,best


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    from pymatgen.core import Structure
    from ase import Atoms
    from ase.io import write
    root=args.run.resolve();plan=load(root/'plan.json');proof=load(root/'native-probe.json')
    if not proof['passed'] or proof['code_sha256']!=code_hashes() or plan['policy']!=POLICY:raise ValueError('Native gradient gate is missing/stale.')
    if len(plan['cases'])!=16 or [c['parent_id'] for c in proof['operators']]!=plan['parent_ids']:raise ValueError('Cohort changed.')
    rpc=root/'search-rpc';rpc.mkdir(exist_ok=False);scorer=GradientScorer();client=Mechanics(rpc)
    baselines=[];seeds=[];attempts=[];selected={f+'-'+a:[] for f in FAMILIES for a in ARMS};selected['seed']=[]
    try:
        for index,(case,operator) in enumerate(zip(plan['cases'],proof['operators'])):
            parent=case['parent_id'];stem=Path(parent).stem
            if sha(operator['operator_path'])!=operator['operator_sha256'] or sha(case['B']['relaxed_path'])!=operator['source_sha256']:raise ValueError('Operator/source SHA changed.')
            arrays=np.load(operator['operator_path']);s=Structure.from_file(case['B']['relaxed_path']);geom=Geometry(s.atomic_numbers,s.lattice.matrix,s.frac_coords)
            base=point(parent,case['raw_path'],case['B'],scorer);baselines.append(base);volume=base['B']['volume_A3'];energy=base['B']['final_energy_eV']/geom.n
            original=case['predictions']['after']
            if max(abs(base['gap']-original['crystalformer_gap']),abs(base['eform']-original['crystalformer_eform']))>1e-5:raise ValueError('Fresh baseline scorer differs.')
            input_dir=root/'inputs'/stem;input_dir.mkdir(parents=True,exist_ok=False)
            def write_q(q,path):
                h,t=geom.pose(q);write(path,Atoms(numbers=geom.numbers,cell=h,scaled_positions=t,pbc=True),format='vasp',direct=True,sort=False)
            seed_path=input_dir/'seed.vasp';write_q(arrays['qseed'],seed_path)
            b=client.request([dict(id=stem+'-seed',action='relax',source=str(seed_path),output=str(root/'B'/stem/'seed'))])[0]
            seedpoint=point(stem+'-seed',seed_path,b,scorer,volume) if b['status']=='converged' else base
            seeds.append(dict(parent_id=parent,B=b,point=seedpoint));selected['seed'].append(best([base,seedpoint],energy))
            pools={f+'-'+a:[base,seedpoint] for f in FAMILIES for a in ARMS};tasks=[];pending=[]
            for f,family in enumerate(FAMILIES):
                for arm in ARMS:
                    direction=arrays['descent'][f] if arm=='descent' else -arrays['descent'][f] if arm=='reverse' else arrays['random'][f]
                    for k,scale in enumerate(SCALES):
                        identifier=f'{stem}-{family}-{arm}-s{k}';q=arrays['qseed']+scale*direction
                        record=dict(parent_id=parent,family=family,arm=arm,scale=scale,id=identifier,actual_direction_norm=float(np.linalg.norm(direction)),q=q.tolist())
                        if np.linalg.norm(direction)<1e-10 or not geom.feasible(q):
                            record.update(status='stationary_or_geometric_rejection',eligible=False);attempts.append(record);continue
                        path=input_dir/(identifier+'.vasp');write_q(q,path)
                        record.update(input=str(path),input_sha256=sha(path));pending.append(record)
                        tasks.append(dict(id=identifier,action='relax',source=str(path),output=str(root/'B'/stem/identifier)))
            results=client.request(tasks)
            for record,B in zip(pending,results):
                record.update(status=B['status'],B=B,eligible=False)
                if B['status']=='converged':
                    try:pnt=point(record['id'],record['input'],B,scorer,volume)
                    except ValueError as error:record.update(status='CF_prediction_failed',error=repr(error));attempts.append(record);continue
                    record.update(point=pnt,eligible=bool(physical(pnt['B'],energy) and pnt['eform']<-.5))
                    pools[record['family']+'-'+record['arm']].append(pnt)
                attempts.append(record)
            for arm,pool in pools.items():selected[arm].append(best(pool,energy))
            save(root/'direction-results.json',dict(baseline=baselines,seeds=seeds,attempts=attempts,selected=selected,CF=scorer.metadata,NFE=client.NFE,mechanics_seconds=client.seconds,completed_parents=index+1))
            print(json.dumps(dict(parent=parent,completed=index+1,attempts=len(attempts),NFE=client.NFE,
                errors={name:abs(points[-1]['gap']-2.5) for name,points in selected.items()})),flush=True)
        if len(attempts)!=16*3*3*4:raise ValueError('Attempt budget changed.')
        for arm,choices in selected.items():export(root,arm,choices,plan,scorer)
        save(root/'direction.complete.json',dict(completed=True,parents=16,attempts=len(attempts),CF=scorer.metadata,NFE=client.NFE))
    finally:client.close()


if __name__=='__main__':main()
