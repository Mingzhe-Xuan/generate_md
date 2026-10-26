"""Calibrate a CF axis using post-quench box-smoothed directional derivatives."""
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
from feedback import validate_source
from settings import FAMILIES,SEED,interval
from probe import load,save,sha
from policy import best,physical

CALIBRATION_RADIUS=.5
VALIDATION_RADIUS=.8
UTILITY_DEADBAND=.005


def oracle_loss(record,base_energy):
    if 'point' not in record:return 1.
    p=record['point']
    if not math.isfinite(p['gap']) or not math.isfinite(p['eform']):raise ValueError('Nonfinite cached CF prediction.')
    if not physical(p['B'],base_energy) or p['eform']>=-.5:return 1.
    error=abs(p['gap']-2.5)
    return error/(1+error)


def box_gradient(plus,minus,radius):
    if not radius>0 or not all(math.isfinite(v) for v in (plus,minus)):raise ValueError('Invalid box derivative inputs.')
    return (plus-minus)/(2*radius)


def calibrated_sign(plus,minus,radius=CALIBRATION_RADIUS):
    gradient=box_gradient(plus,minus,radius)
    return 0 if abs(plus-minus)<=UTILITY_DEADBAND else -1 if gradient>0 else 1


def axis_case(base,seed,records,parent_index):
    energy=base['B']['final_energy_eV']/base['B']['num_atoms']
    keyed={(r['arm'],r['scale']):r for r in records}
    needed={(a,s) for a in ('descent','reverse') for s in (CALIBRATION_RADIUS,VALIDATION_RADIUS)}
    if not needed.issubset(keyed):raise ValueError('Missing calibration/heldout direction record.')
    plus=keyed['descent',CALIBRATION_RADIUS];minus=keyed['reverse',CALIBRATION_RADIUS]
    hp,hm=oracle_loss(plus,energy),oracle_loss(minus,energy);sign=calibrated_sign(hp,hm)
    pool=[base,seed]+[r['point'] for r in (plus,minus) if 'point' in r];query_best=best(pool,energy)
    random_sign=int(np.random.default_rng([SEED,parent_index,61234]).choice([-1,1]))
    signs=dict(calibrated=sign,reversed_calibration=-sign,raw_CF=1,random_sign=random_sign)
    results={}
    for name,choice in signs.items():
        row=keyed['descent' if choice>0 else 'reverse',VALIDATION_RADIUS] if choice else None
        candidate=[row['point']] if row is not None and 'point' in row else []
        selected=best(pool+candidate,energy)
        results[name]=dict(chosen_sign=choice,selected=selected,error_eV=abs(selected['gap']-2.5),
            retained_gain_beyond_queries_eV=abs(query_best['gap']-2.5)-abs(selected['gap']-2.5),
            extra_attempt_status='stationary_budget_consumed' if choice==0 else row.get('status','unknown'))
    return dict(calibration_radius=CALIBRATION_RADIUS,validation_radius=VALIDATION_RADIUS,calibration_losses=[hp,hm],
        smoothed_axis_derivative=box_gradient(hp,hm,CALIBRATION_RADIUS),calibrated_sign=sign,query_best=query_best,policies=results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Batch analysis must use Slurm.')
    source=a.source.resolve();plan=load(source/'plan.json');data=load(source/'direction-results.json');proof=load(source/'native-probe.json')
    validate_source(plan,data,proof)
    if not (source/'report.json').is_file() or not load(source/'direction.complete.json')['completed']:raise ValueError('Need complete source final evaluation.')
    output=source/('sign-calibration-'+os.environ['SLURM_JOB_ID']);output.mkdir(exist_ok=False)
    checked=set();families={}
    for family in FAMILIES:
        rows=[]
        for i,case in enumerate(plan['cases']):
            parent=case['parent_id'];records=[r for r in data['attempts'] if r['parent_id']==parent and r['family']==family]
            used=[r for r in records if r['arm'] in ('descent','reverse') and r['scale'] in (CALIBRATION_RADIUS,VALIDATION_RADIUS)]
            for record in used:
                if 'point' not in record:continue
                q=record['point'];path=q['pose']
                if path not in checked:
                    if sha(path)!=q['B']['relaxed_sha256'] or sha(q['input'])!=q['B']['input_sha256']:raise ValueError('Cached input/terminal provenance mismatch.')
                    checked.add(path)
            row=axis_case(data['baseline'][i],data['seeds'][i]['point'],records,int(Path(parent).stem.split('-')[1]));row['parent_id']=parent;rows.append(row)
        values={name:np.array([r['policies'][name]['error_eV'] for r in rows]) for name in rows[0]['policies']}
        calibrated=values['calibrated'];summary=dict(denominator=16,calibratable_parents=sum(r['calibrated_sign']!=0 for r in rows),
            MAE={name:float(v.mean()) for name,v in values.items()},
            gradient_proposal_gains_ge_0p1_beyond_queries=sum(r['policies']['calibrated']['retained_gain_beyond_queries_eV']>=.1 for r in rows),
            contrasts={name:dict(mean_delta=float((calibrated-v).mean()),CI95=interval(calibrated-v)) for name,v in values.items() if name!='calibrated'})
        families[family]=dict(summary=summary,cases=rows)
    result=dict(role='Exploratory heldout-amplitude sign audit, not a new optimization experiment or independent confirmation.',
        source_hashes={n:sha(source/n) for n in ('plan.json','direction-results.json','native-probe.json','report.json')},
        protocol=dict(calibration_radius_A=CALIBRATION_RADIUS,heldout_radius_A=VALIDATION_RADIUS,utility_deadband=UTILITY_DEADBAND,
            oracle='MAE/(1+MAE) for fullB physically eligible/Ef<-.5, else1',
            budget='Two common calibration queries andone heldout sign query per policy/parent; zero gradient consumes attempt. Common original/seed/probe fallback.',
            limitation='CFaxis fixed locally; derivative belongs to uniform-smoothed postB objective, not unsmoothed fullR. No random-axis negative-ray data or newALIGNN evaluations; positive discovery needs fresh native trials.'),families=families)
    save(output/'audit.json',result);print(json.dumps({name:r['summary'] for name,r in families.items()}),flush=True)


if __name__=='__main__':main()
