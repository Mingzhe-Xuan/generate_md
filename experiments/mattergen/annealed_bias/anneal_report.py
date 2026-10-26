"""Native-gated pilot/full results; integration1 cannot establish population value."""
import argparse
from pathlib import Path
import numpy as np
from anneal_experiment import OUTPUTS,validate
from anneal_native import save
from anneal_policy import POLICY,ARMS
from anneal_shared import load,sha,physical,interval


def contrast(a,b):
    delta=np.asarray(a)-np.asarray(b)
    return dict(mean_delta=float(delta.mean()),CI95=interval(delta))


def assessment(errors,eligible,count):
    if set(errors)!=set(OUTPUTS) or any(np.asarray(v).shape!=(count,) for v in errors.values()) or len(eligible)!=count:raise ValueError('Complete paired scope required.')
    if count!=16:return dict(role='Integration1 only; not population efficacy.',development_directional_evidence=False,annealing_value=False,goal_completion_proven=False)
    effects={a:contrast(errors['full'],errors[a]) for a in ('reverse','random','zero','fixed','seed','baseline')}
    material=int(sum((np.asarray(errors['zero'])-errors['full']>=.1)&np.asarray(eligible,dtype=bool)))
    direction=bool(material>=4 and all(effects[a]['mean_delta']<=-.05 and effects[a]['CI95'][1]<0 for a in ('reverse','random','zero')))
    return dict(material_parents_beyond_zero_bias_pool=material,contrasts=effects,development_directional_evidence=direction,
        annealing_value=bool(direction and effects['fixed']['mean_delta']<=-.05 and effects['fixed']['CI95'][1]<0),
        unseen_parent_confirmation_required=True,matched_tight_retention_required=True,goal_completion_proven=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    plan,native,data,done=[load(root/n) for n in ('plan.json','native.json','results.json','complete.json')];validate(root,plan,native,data);count=plan['count']
    if not done['completed'] or any(done[k]!=sha(root/f) for k,f in (('plan_sha256','plan.json'),('native_sha256','native.json'),('results_sha256','results.json'))):raise ValueError('Completed artifacts changed.')
    arms={};errors={};full_eligible=[]
    for method in OUTPUTS:
        folder=root/f'evaluation-{method}/cross-prediction';meta=load(folder/'metadata.json')
        if not meta['crystalformer_control']['control_passed'] or len(meta['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in meta['controls']):raise ValueError('Official model controls failed.')
        rows=[r for r in load(folder/'predictions.json') if r['stage']=='after'];byid={r['candidate_id']:r for r in rows}
        if len(rows)!=count or set(byid)!=set(plan['parent_ids']):raise ValueError('Prediction population changed.')
        cases=[]
        for i,(parent,q) in enumerate(zip(plan['parent_ids'],data['selected'][method])):
            pred=byid[parent];base=native['baseline'][i];valid=physical(q['B'],base['B']['final_energy_eV']/base['B']['num_atoms'])
            if sha(q['pose'])!=pred['geometry_sha256'] or pred['geometry_sha256']!=q['B']['relaxed_sha256'] or max(abs(pred['crystalformer_gap']-q['gap']),abs(pred['crystalformer_eform']-q['eform']))>1e-5:raise ValueError('Final model/pose linkage changed.')
            cases.append(dict(parent_id=parent,point_id=q['id'],prediction=pred,physical_eligible=valid,selection_eligible=valid and q['eform']<-.5,B=q['B'],pose=q['pose']))
            if method=='full':full_eligible.append(valid and q['eform']<-.5)
        summary={}
        for model in ('crystalformer','alignn'):
            values=[abs(r['prediction'][model+'_gap']-2.5) for r in cases if r['prediction'].get(model+'_gap') is not None]
            summary[model]=dict(denominator=count,known=len(values),MAE=float(np.mean(values)) if values else None,
                **{k+'_pass':sum(r['physical_eligible'] and r['prediction'][model+'_flags'][k] is True for r in cases) for k in ('gap','eform','joint')})
        arms[method]=dict(cases=cases,summary=summary);errors[method]=np.array([abs(byid[p]['crystalformer_gap']-2.5) for p in plan['parent_ids']])
    costs={}
    for arm in ARMS:
        ts=[t for t in data['trajectories'] if t['arm']==arm];rs=[r for r in data['attempts'] if r.get('arm')==arm]
        costs[arm]=dict(trajectories=len(ts),fork_attempts=len(rs),steering_NFE=sum(t['NFE_steering'] for t in ts),quench_NFE=sum(r.get('B',{}).get('NFE',0) for r in rs),
            FIRE_updates=sum(s['updates'] for t in ts for s in t['stages']),converged_stages=sum(s['converged'] for t in ts for s in t['stages']),steering_failures=sum(t['status']!='completed' for t in ts))
    result=dict(policy=POLICY,role=plan['role'],parent_ids=plan['parent_ids'],assessment=assessment(errors,full_eligible,count),arms=arms,
        costs=dict(actual_NFE=data['NFE_total'],native_preflight_NFE=native['NFE'],steering_NFE=data['NFE_steering'],quench_NFE=data['NFE_quench'],reused_native_proof_NFE=data['reused_native_proof_NFE'],arms=costs),
        plan_sha256=sha(root/'plan.json'),report_sha256=sha(__file__),
        limitations='ActualF64 CF/per-atomF32 physicalenergy bias; conservative random initialnorm-matched tilt. Intermediate biased stationarity is not physical force convergence. All9zeroCF endpoints+seedB+originalB are shared fallback. Stage subsets have3additionalqueries; fullarms9. Equalmaxsteps/querycounts are not equalNFE/neuralcalls/FLOPs. Integration1 is not population efficacy. CFformation/force guards do not certifySUN/DFT stability.')
    save(root/'report.json',result);print(__import__('json').dumps(dict(assessment=result['assessment'],summary={k:v['summary'] for k,v in arms.items()})),flush=True)


if __name__=='__main__':main()
