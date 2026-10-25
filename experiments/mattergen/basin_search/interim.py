"""Read-only CPU materiality check of completed search methods; never final proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from policy import ACCEPTANCE,physical
from report import interval


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Batch geometry inspection must use Slurm.')
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    root=args.run.resolve();plan=load(root/'plan.json');snapshot=(root/'search-results.json').read_bytes();data=json.loads(snapshot)
    parents=plan['parent_ids']
    if len(parents)!=16 or len(set(parents))!=16 or [c['parent_id'] for c in plan['cases']]!=parents or len(data['baseline'])!=16:raise ValueError('Full16-parent cohort required.')
    if plan['acceptance']!=ACCEPTANCE:raise ValueError('Frozen acceptance criteria changed.')
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    results={}
    for method in data['pools']:
        points=data['selected'][method]
        if len(points)!=16:raise ValueError('Completed method has missing parents.')
        rows=[]
        for case,point in zip(plan['cases'],points):
            if point['id'].split('.')[0].split('-'+method+'-')[0]!=Path(case['parent_id']).stem:raise ValueError('Selected point belongs to another parent.')
            B=point['B'];old=case['predictions']['after'];baseline=case['B'];base_energy=baseline['final_energy_eV']/baseline['num_atoms']
            if sha(point['pose'])!=B['relaxed_sha256'] or sha(baseline['relaxed_path'])!=baseline['relaxed_sha256']:raise ValueError('Terminal source hash changed.')
            first=Structure.from_file(baseline['relaxed_path']);second=Structure.from_file(point['pose']);metrics=measure(first,second)
            match=bool(matcher.fit(second,first));valid=physical(B,base_energy) and point['eform']<-.5
            gain=abs(old['crystalformer_gap']-2.5)-abs(point['gap']-2.5)
            changed=not match and metrics['species_pair_distance_RMS_A']>=plan['acceptance']['min_pair_distance_RMS_A']
            rows.append(dict(parent_id=case['parent_id'],point_id=point['id'],CF_error_gain=gain,geometry=metrics,author_match=match,
                physical=valid,material_changed=changed,material_CF_improved=bool(valid and changed and gain>=plan['acceptance']['min_CF_error_improvement_eV'])))
        results[method]=dict(material_changed=sum(r['material_changed'] for r in rows),material_CF_improved=sum(r['material_CF_improved'] for r in rows),
            CF_MAE_before=float(np.mean([abs(c['predictions']['after']['crystalformer_gap']-2.5) for c in plan['cases']])),
            CF_MAE_after=float(np.mean([abs(p['gap']-2.5) for p in points])),CF_delta_CI95=interval([-r['CF_error_gain'] for r in rows]),cases=rows)
    out=dict(provisional=True,final_verification_complete=False,parent_count=16,source_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
        methods=results,note='Original post-quench search CF scores only; independent final frozenCF/ALIGNN evaluation pending. This file cannot trigger confirmation and is not goal completion.')
    destination=root/f'interim-{os.environ["SLURM_JOB_ID"]}.json';destination.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({m:{k:v for k,v in r.items() if k!='cases'} for m,r in results.items()}),flush=True)


if __name__=='__main__':main()
