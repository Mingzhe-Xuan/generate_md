"""Frozen conditional cohort and seven matched physical-retention comparisons."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anneal_policy import PARENTS

STAGES=('original','seed','zero','reverse','random','fixed','selected')
POLICY=dict(fmax=.001,maxstep=.05,steps=1600,preserve_symmetry=False,trials=4,
    atomic_RMS_A=.02,strain=.003,seed=20261028,energy_ceiling_eV_atom=.05,volume_bounds=[.65,1.4],
    selection='ALL full16 selection-eligible CF gains>=.1 beyond the shared zero-CF pool, without ALIGNN or global-flag filtering.',
    stages=list(STAGES),pairing='Trial0 unchanged; the same parent/trial seed across all seven endpoints. Feasibility resampling can differ; actual inputs are saved. Trials are correlated within parents.',
    role='Conditional physical retention; does not replace population directionality or unseen-parent confirmation.')


def material_cases(plan,data,report):
    if plan['count']!=16 or plan['parent_ids']!=list(PARENTS):raise ValueError('Complete original16 required; pilot is not population evidence.')
    parents=plan['parent_ids'];result=[]
    for m in ('full','zero'):
        if len(data['selected'][m])!=16 or [r['parent_id'] for r in report['arms'][m]['cases']]!=parents:raise ValueError('Source coverage changed.')
    for i,parent in enumerate(parents):
        a,b=(report['arms'][m]['cases'][i] for m in ('full','zero'))
        for m,row in (('full',a),('zero',b)):
            q=data['selected'][m][i]
            if row['point_id']!=q['id'] or abs(row['prediction']['crystalformer_gap']-q['gap'])>1e-5:raise ValueError('Source point/score link changed.')
        valid=a['physical_eligible'] and data['selected']['full'][i]['eform']<-.5
        if a['selection_eligible']!=valid:raise ValueError('Source eligibility changed.')
        gain=abs(b['prediction']['crystalformer_gap']-2.5)-abs(a['prediction']['crystalformer_gap']-2.5)
        if valid and gain>=.1:result.append(dict(parent_id=parent,index=i,source_gain_eV=gain))
    if len(result)!=report['assessment']['material_parents_beyond_zero_bias_pool']:raise ValueError('Source material count changed.')
    return result


def endpoints(data,native,case):
    i=case['index']
    return dict(original=native['baseline'][i],selected=data['selected']['full'][i],
        **{m:data['selected'][m][i] for m in ('seed','zero','reverse','random','fixed')})


def validate_trials(cases,records):
    parents=[c['parent_id'] for c in cases]
    expected={(p,s,t) for p in parents for s in STAGES for t in range(4)}
    if (len(parents)!=len(set(parents)) or len(records)!=28*len(cases)
        or {(r['parent_id'],r['stage'],r['trial']) for r in records}!=expected
        or len({r['candidate_id'] for r in records})!=len(records)):
        raise ValueError('Missing/duplicate seven-stage trial ledger.')


def eligible(row):
    if row.get('status')!='converged' or row.get('converged') is not True or row.get('separated') is not True:return False
    keys=('max_raw_atomic_force','max_filter_force','final_energy_eV','num_atoms','original_energy_eV_atom','original_volume_ratio')
    if any(row.get(k) is None for k in keys) or not np.isfinite([row[k] for k in keys]).all() or row['num_atoms']<=0:return False
    return bool(max(row['max_raw_atomic_force'],row['max_filter_force'])<=.001 and .65<=row['original_volume_ratio']<=1.4
        and row['final_energy_eV']/row['num_atoms']<=row['original_energy_eV_atom']+.05)


def summarize(cases,records,predictions):
    validate_trials(cases,records)
    if set(predictions)!={r['candidate_id'] for r in records}:raise ValueError('Prediction coverage changed.')
    lookup={(r['parent_id'],r['stage'],r['trial']):r for r in records};result=[]
    for case in cases:
        models={}
        for model in ('crystalformer','alignn'):
            models[model]={}
            for stage in STAGES[:-1]:
                pairs=[]
                for trial in range(4):
                    a=lookup[case['parent_id'],stage,trial];b=lookup[case['parent_id'],'selected',trial]
                    x,y=predictions[a['candidate_id']],predictions[b['candidate_id']]
                    known=all(z.get(model+'_gap') is not None and np.isfinite(z[model+'_gap']) for z in (x,y))
                    valid=all(eligible(r) and z.get('crystalformer_eform') is not None and z['crystalformer_eform']<-.5 for r,z in ((a,x),(b,y)))
                    pairs.append(dict(trial=trial,known=known,eligible=valid,gain_eV=abs(x[model+'_gap']-2.5)-abs(y[model+'_gap']-2.5) if known else None))
                values=[r['gain_eV'] for r in pairs if r['known'] and r['eligible']]
                models[model][stage]=dict(trials=4,known_pairs=sum(r['known'] for r in pairs),eligible_known_pairs=len(values),pairs=pairs,
                    median_gain_eV=float(np.median(values)) if values else None,min_gain_eV=min(values) if values else None,
                    all_four_gain_ge_0p1=len(values)==4 and min(values)>=.1,all_four_gain_gt_1meV=len(values)==4 and min(values)>.001)
        result.append(dict(**case,models=models))
    return result
