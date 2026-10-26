"""Separate denoising-only gains, CF direction value and decoder-Jacobian value."""
import argparse
import json
from pathlib import Path
import numpy as np
from denoised_policy import ARMS, OUTPUTS, POLICY
from denoised_search import load, save, sha
from denoised_validate import validate
from denoised_shared import interval,physical


def contrast(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=(16,) or b.shape!=(16,):
        raise ValueError('Require all16 paired parents.')
    delta=a-b
    return dict(mean_delta=float(delta.mean()),CI95=interval(delta))


def assessment(errors,eligible):
    if set(errors)!=set(OUTPUTS) or any(np.asarray(v).shape!=(16,) for v in errors.values()) or np.asarray(eligible).shape!=(16,):
        raise ValueError('Complete matched16 outputs required.')
    full=errors['full'];seed=errors['seeds']
    effects={a:contrast(full,errors[a]) for a in ('reverse','random','stop','direct','seeds','baseline')}
    material=int(sum((np.asarray(seed)-full>=.1)&np.asarray(eligible,dtype=bool)))
    directional=bool(material>=4 and all(effects[a]['mean_delta']<=-.05 and effects[a]['CI95'][1]<0 for a in ('seeds','reverse','random')))
    jacobian=bool(directional and all(effects[a]['mean_delta']<=-.05 and effects[a]['CI95'][1]<0 for a in ('stop','direct')))
    return dict(material_parents_beyond_shared_seed_pool=material,contrasts=effects,development_directional_evidence=directional,
        development_decoder_Jacobian_value=jacobian,unseen_parent_confirmation_required=True,strict_matched_retention_required=True,goal_completion_proven=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    plan,native,data,done=[load(root/n) for n in ('plan.json','native.json','results.json','complete.json')]
    validate(root,plan,native,data)
    if not done['completed'] or any(done[key]!=sha(root/name) for key,name in (('plan_sha256','plan.json'),('native_sha256','native.json'),('results_sha256','results.json'))):
        raise ValueError('Complete study artifact changed.')
    arms={};errors={}
    for method in OUTPUTS:
        folder=root/f'evaluation-{method}/cross-prediction';meta=load(folder/'metadata.json')
        if not meta['crystalformer_control']['control_passed'] or len(meta['controls'])<2 or any(max(q['max_errors'])>1e-5 for q in meta['controls']):
            raise ValueError('Frozen CF/ALIGNN native controls failed.')
        records=[q for q in load(folder/'predictions.json') if q['stage']=='after'];byid={q['candidate_id']:q for q in records}
        if len(records)!=16 or set(byid)!=set(plan['parent_ids']):
            raise ValueError('Official all16 prediction coverage differs.')
        cases=[]
        for i,(parent,q) in enumerate(zip(plan['parent_ids'],data['selected'][method])):
            pred=byid[parent];base=native['baseline'][i];energy=base['B']['final_energy_eV']/base['B']['num_atoms']
            if pred['geometry_sha256']!=q['B']['relaxed_sha256'] or sha(q['pose'])!=q['B']['relaxed_sha256'] or max(abs(pred['crystalformer_gap']-q['gap']),abs(pred['crystalformer_eform']-q['eform']))>1e-5:
                raise ValueError('Final saved pose/CF score linkage changed.')
            valid=bool(physical(q['B'],energy))
            cases.append(dict(parent_id=parent,point_id=q['id'],prediction=pred,physical_eligible=valid,selection_eligible=valid and q['eform']<-.5,B=q['B'],pose=q['pose']))
        summary={}
        for model in ('crystalformer','alignn'):
            known=[c for c in cases if c['prediction'].get(model+'_gap') is not None]
            values=[abs(c['prediction'][model+'_gap']-2.5) for c in known]
            summary[model]=dict(denominator=16,known=len(known),MAE=float(np.mean(values)) if values else None,
                gap_pass=sum(c['physical_eligible'] and c['prediction'][model+'_flags']['gap'] is True for c in cases),
                eform_pass=sum(c['physical_eligible'] and c['prediction'][model+'_flags']['eform'] is True for c in cases),
                joint_pass=sum(c['physical_eligible'] and c['prediction'][model+'_flags']['joint'] is True for c in cases))
        arms[method]=dict(cases=cases,summary=summary)
        errors[method]=np.array([abs(c['prediction']['crystalformer_gap']-2.5) for c in cases])
    for method,arm in arms.items():
        for model in ('crystalformer','alignn'):
            deltas=[]
            for row,base in zip(arm['cases'],arms['baseline']['cases']):
                x,y=row['prediction'].get(model+'_gap'),base['prediction'].get(model+'_gap')
                if x is not None and y is not None:
                    deltas.append(abs(x-2.5)-abs(y-2.5))
            arm['summary'][model].update(paired_baseline_known=len(deltas),MAE_vs_baseline_delta=float(np.mean(deltas)) if deltas else None,MAE_vs_baseline_CI95=interval(deltas) if deltas else None)
    costs={}
    for name in ARMS+('seeds',):
        rows=[r for r in data['attempts'] if (r['kind']=='seed' if name=='seeds' else r.get('arm')==name)]
        costs[name]=dict(attempts=len(rows),evaluated_points=sum('point' in r for r in rows),B_NFE=sum(r.get('B',{}).get('NFE',0) for r in rows),
            rejected_or_failed=sum('point' not in r for r in rows))
    result=dict(policy=POLICY,parent_ids=plan['parent_ids'],assessment=assessment(errors,[c['selection_eligible'] for c in arms['full']['cases']]),arms=arms,
        costs=dict(NFE_native=data['NFE_native'],NFE_search=data['NFE_search'],NFE_total=data['NFE_total'],arms=costs),
        plan_sha256=sha(root/'plan.json'),report_sha256=sha(__file__),
        limitations='One gradient direction/four scales at each of two noisy seeds. All arms share BOTH denoised-seed-B outcomes and originalB. DirectCF uses its original RMS/strain metric at the common clean seed; decoded arms use symmetric-cell latent coordinates. Equal ray attempts are not equal NFE, neural calls, decoded displacement or FLOPs. ALIGNN never ranks. Actual-F64 proposal arithmetic is distinct from failed F32-proxy proofs. No SUN/DFT stability certification.')
    save(root/'report.json',result)
    print(json.dumps(dict(assessment=result['assessment'],summary={k:v['summary'] for k,v in arms.items()})),flush=True)


if __name__=='__main__':
    main()
