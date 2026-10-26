"""Paired development evidence separates sampled probes, gradient updates and CF prior."""
import argparse
import json
from pathlib import Path
import numpy as np
from guided_common import load,save,sha,validate_plan,implementation_hashes,physical
from guided_settings import POLICY,FAMILIES,OUTPUTS
from guided_validate import validate_study
from settings import interval


def contrast(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if x.shape!=(16,) or y.shape!=(16,):raise ValueError('Require all16 paired development parents.')
    d=x-y;return dict(mean_delta=float(d.mean()),CI95=interval(d))


def assessment(errors,eligible):
    masks=eligible if isinstance(eligible,dict) else {f:eligible for f in FAMILIES}
    if (set(masks)!=set(FAMILIES) or any(np.asarray(v).shape!=(16,) for v in errors.values())
        or any(np.asarray(v).shape!=(16,) for v in masks.values())):raise ValueError('Incomplete16 development evidence.')
    family={}
    for name in FAMILIES:
        g=errors[name+'-descent'];pool=errors[name+'-probes'];controls={a:contrast(g,errors[name+'-'+a]) for a in ('reverse','random','raw_cf')}
        material=int(sum((pool-g>=.1)&np.asarray(masks[name],dtype=bool)))
        update=contrast(g,pool);signal=bool(material>=4 and update['mean_delta']<=-.05 and update['CI95'][1]<0
            and all(controls[a]['mean_delta']<=-.05 and controls[a]['CI95'][1]<0 for a in ('reverse','random')))
        family[name]=dict(material_parents_beyond_probe_pool=material,gradient_vs_controls=controls,gradient_vs_probe_best=update,development_directional_evidence=signal)
    prior={};g=errors['cf_guided-descent'];increment=g-errors['cf_guided-probes']
    for name in ('random_guided','isotropic'):
        prior[name]=dict(final=contrast(g,errors[name+'-descent']),incremental=contrast(increment,errors[name+'-descent']-errors[name+'-probes']))
    importance=family['cf_guided']['development_directional_evidence'] and all(v[k]['mean_delta']<=-.05 and v[k]['CI95'][1]<0 for v in prior.values() for k in ('final','incremental'))
    return dict(families=family,CF_prior_ablations=prior,development_CF_gradient_importance=bool(importance),independent_confirmation_required=True,goal_completion_proven=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    from pymatgen.core import Structure
    plan=load(root/'plan.json');proof=load(root/'native.json');data=load(root/'results.json');done=load(root/'complete.json');validate_plan(plan);validate_study(plan,proof,data)
    if (not done['completed'] or done['results_sha256']!=sha(root/'results.json') or done['native_sha256']!=sha(root/'native.json')
        or proof['plan_sha256']!=sha(root/'plan.json') or done['plan_sha256']!=sha(root/'plan.json') or proof['code_sha256']!=implementation_hashes()):raise ValueError('Final guided source/data proof changed.')
    reports={};errors={};baselines={p:Structure.from_file(q['pose']) for p,q in zip(plan['parent_ids'],data['baseline'])}
    for method in OUTPUTS:
        folder=root/f'evaluation-{method}/cross-prediction';metadata=load(folder/'metadata.json')
        if not metadata['crystalformer_control']['control_passed'] or len(metadata['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in metadata['controls']):raise ValueError('Independent model controls failed.')
        records=[r for r in load(folder/'predictions.json') if r['stage']=='after'];byid={r['candidate_id']:r for r in records}
        if len(records)!=16 or set(byid)!=set(plan['parent_ids']):raise ValueError('Final16 prediction coverage changed.')
        rows=[]
        for i,(parent,q) in enumerate(zip(plan['parent_ids'],data['selected'][method])):
            pred=byid[parent];s=Structure.from_file(q['pose']);reference=baselines[parent];base=data['baseline'][i];energy=base['B']['final_energy_eV']/base['B']['num_atoms']
            if sha(q['pose'])!=q['B']['relaxed_sha256'] or pred['geometry_sha256']!=q['B']['relaxed_sha256'] or tuple(s.atomic_numbers)!=tuple(reference.atomic_numbers):raise ValueError('Pose hash/ordered species changed.')
            if max(abs(pred['crystalformer_gap']-q['gap']),abs(pred['crystalformer_eform']-q['eform']))>1e-5 or abs(q['B']['volume_ratio']-s.volume/reference.volume)>1e-6:raise ValueError('Frozen CF score/original volume differs.')
            valid=bool(physical(q['B'],energy) and q['eform']<-.5)
            rows.append(dict(parent_id=parent,point_id=q['id'],prediction=pred,physical_eligible=valid,B=q['B'],pose=q['pose']))
        summary={}
        for model in ('crystalformer','alignn'):
            values=[abs(r['prediction'][model+'_gap']-2.5) for r in rows if r['prediction'].get(model+'_gap') is not None]
            summary[model]=dict(denominator=16,known=len(values),MAE=float(np.mean(values)) if values else None,
                gap_pass=sum(r['prediction'][model+'_flags']['gap'] is True for r in rows),eform_pass=sum(r['prediction'][model+'_flags']['eform'] is True for r in rows),
                joint_pass=sum(r['prediction'][model+'_flags']['joint'] is True and r['physical_eligible'] for r in rows))
        reports[method]=dict(cases=rows,summary=summary);errors[method]=np.array([abs(r['prediction']['crystalformer_gap']-2.5) for r in rows])
    evidence=assessment(errors,{f:[r['physical_eligible'] for r in reports[f+'-descent']['cases']] for f in FAMILIES})
    for method,entry in reports.items():
        for model in ('crystalformer','alignn'):
            delta=[]
            for r,b in zip(entry['cases'],reports['baseline']['cases']):
                x=r['prediction'].get(model+'_gap');y=b['prediction'].get(model+'_gap')
                if x is not None and y is not None:delta.append(abs(x-2.5)-abs(y-2.5))
            entry['summary'][model].update(paired_baseline_known=len(delta),MAE_vs_baseline_delta=float(np.mean(delta)) if delta else None,MAE_vs_baseline_CI95=interval(delta) if delta else None)
    costs={}
    for name in FAMILIES:
        rows=[r for r in data['attempts'] if r['family']==name]
        costs[name]=dict(attempts=len(rows),probes=sum(r['kind']=='probe' for r in rows),rays=sum(r['kind']=='ray' for r in rows),
            B_NFE=sum(r.get('B',{}).get('NFE',0) for r in rows),failed_or_geometric=sum('point' not in r for r in rows))
    result=dict(role='Exploratory method development on original16, not independent confirmation.',policy=POLICY,parent_ids=plan['parent_ids'],
        assessment=evidence,arms=reports,costs=dict(NFE_total=data['NFE_total'],NFE_native=data['NFE_native'],NFE_search=data['NFE_search'],families=costs),
        plan_sha256=sha(root/'plan.json'),report_sha256=sha(__file__),limitations='CF guide specifies a covariance axis; antithetic full-B utility supplies its update sign. Estimated gradient is preconditioned/smoothed, not exact backpropagation through B. Probe-only gains do not establish gradient-update value. Equal query counts are not equal NFE/FLOPs; ALIGNN is held out, and no SUN/DFT stability claim is made.')
    save(root/'report.json',result);print(json.dumps(dict(assessment=evidence,summary={k:v['summary'] for k,v in reports.items()})),flush=True)


if __name__=='__main__':main()
