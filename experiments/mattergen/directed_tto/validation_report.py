"""Independent48-parent, same-budget gradient direction validation report."""
import argparse
import json
from pathlib import Path
import numpy as np
from validation_run import validate_manifest,validate_results,core_hashes,OUTPUTS
from validation_policy import VALIDATION_POLICY
from settings import ARMS,interval
from probe import load,save,sha
from policy import physical


def assess(errors,baseline,seed,eligible):
    arrays=[np.asarray(v) for v in (errors['descent'],errors['reverse'],errors['random'],baseline,seed,eligible)]
    if any(v.shape!=(48,) for v in arrays):raise ValueError('Require all48 paired parents.')
    g,reverse,random,base,zero,valid=arrays;criterion=VALIDATION_POLICY['directional_evidence']
    contrasts={name:dict(mean_delta=float((g-v).mean()),CI95=interval(g-v)) for name,v in (('reverse',reverse),('random',random))}
    material=int(sum((np.minimum(base,zero)-g>=.1)&valid.astype(bool)))
    success=material>=criterion['min_material_parents'] and all(v['mean_delta']<=-criterion['min_mean_advantage_vs_each_control_eV'] and v['CI95'][1]<0 for v in contrasts.values())
    return dict(material_parents=material,gradient_vs_controls=contrasts,MAE_vs_baseline_delta=float((g-base).mean()),
        MAE_vs_baseline_CI95=interval(g-base),independent_directional_evidence=bool(success),
        physical_retention_checks_still_required=bool(material>0),goal_completion_proven=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    from pymatgen.core import Structure
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    plan=load(root/'validation-plan.json');data=load(root/'validation-results.json');proof=load(root/'validation-native.json');complete=load(root/'validation.complete.json')
    validate_manifest(plan,sha(root/'validation-plan.json'));validate_results(plan,data)
    if not complete['completed'] or complete['data_sha256']!=sha(root/'validation-results.json') or proof['code_sha256']!=core_hashes():raise ValueError('Validation data/source proof changed.')
    if not proof['passed'] or len(proof['cases'])!=48 or not all(c['passed'] for c in proof['cases']):raise ValueError('Native48 gates incomplete.')
    reports={};predictions={};checked={};base_structures={}
    for case in plan['cases']:base_structures[case['parent_id']]=Structure.from_file(case['B']['relaxed_path'])
    for method in OUTPUTS:
        folder=root/f'evaluation-{method}/cross-prediction';metadata=load(folder/'metadata.json')
        if not metadata['crystalformer_control']['control_passed'] or len(metadata['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in metadata['controls']):raise ValueError('Frozen evaluator controls failed.')
        records=[r for r in load(folder/'predictions.json') if r['stage']=='after'];byid={r['candidate_id']:r for r in records}
        if len(records)!=48 or set(byid)!=set(plan['parent_ids']):raise ValueError('Final48 evaluation coverage changed.')
        predictions[method]=byid;rows=[]
        for i,(case,q) in enumerate(zip(plan['cases'],data['selected'][method])):
            parent=case['parent_id'];pred=byid[parent]
            if pred.get('crystalformer_gap') is None or max(abs(pred['crystalformer_gap']-q['gap']),abs(pred['crystalformer_eform']-q['eform']))>1e-5:raise ValueError('Independent CF score differs from selected score.')
            if q['pose'] not in checked:
                s=Structure.from_file(q['pose'])
                if sha(q['pose'])!=q['B']['relaxed_sha256'] or tuple(s.atomic_numbers)!=tuple(base_structures[parent].atomic_numbers):raise ValueError('Terminal SHA/ordered species changed.')
                checked[q['pose']]=s
            s=checked[q['pose']]
            if pred['geometry_sha256']!=q['B']['relaxed_sha256'] or abs(q['B']['volume_ratio']-s.volume/base_structures[parent].volume)>1e-6:raise ValueError('Terminal geometry/original volume linkage changed.')
            energy=case['B']['final_energy_eV']/case['B']['num_atoms'];structure_valid=physical(q['B'],energy);valid=structure_valid and q['eform']<-.5
            comparator=min(abs(data['baseline'][i]['gap']-2.5),abs(data['seeds'][i]['control']['gap']-2.5))
            gain=comparator-abs(pred['crystalformer_gap']-2.5)
            material=bool(gain>=.1 and valid)
            rows.append(dict(parent_id=parent,point_id=q['id'],prediction=pred,structure_eligible=bool(structure_valid),physical_eligible=bool(valid),
                retained_CF_gain_eV=gain,material_CF_gain=material,geometry=measure(base_structures[parent],s) if material else None,
                B=q['B'],selected_path=q['pose']))
        reports[method]=dict(cases=rows,summary={})
    baseline_controls=[]
    for i,case in enumerate(plan['cases']):
        fresh=predictions['baseline'][case['parent_id']];old=case['predictions']['after']
        for model in ('crystalformer','alignn'):
            for prop in ('gap','eform'):
                key=model+'_'+prop
                known=fresh.get(key) is not None and old.get(key) is not None
                if (model=='crystalformer' and not known) or (known and abs(fresh[key]-old[key])>1e-5):raise ValueError('Fresh original-geometry baseline predictor control failed: '+str((case['parent_id'],key,fresh.get(key),old.get(key))))
                baseline_controls.append(dict(parent_id=case['parent_id'],quantity=key,known=known,error=abs(fresh[key]-old[key]) if known else None))
    for method,result in reports.items():
        for model in ('crystalformer','alignn'):
            rows=result['cases'];known=[r for r in rows if r['prediction'].get(model+'_gap') is not None]
            deltas=[abs(r['prediction'][model+'_gap']-2.5)-abs(predictions['baseline'][r['parent_id']][model+'_gap']-2.5) for r in known if predictions['baseline'][r['parent_id']].get(model+'_gap') is not None]
            result['summary'][model]=dict(denominator=48,known=len(known),MAE=float(np.mean([abs(r['prediction'][model+'_gap']-2.5) for r in known])) if known else None,
                gap_pass=sum(r['prediction'][model+'_flags']['gap'] is True for r in rows),eform_pass=sum(r['prediction'][model+'_flags']['eform'] is True for r in rows),
                property_joint_pass=sum(r['prediction'][model+'_flags']['joint'] is True for r in rows),
                joint_pass=sum(r['prediction'][model+'_flags']['joint'] is True and r['physical_eligible'] for r in rows),
                paired_baseline_parents=len(deltas),MAE_vs_baseline_delta=float(np.mean(deltas)) if deltas else None,MAE_vs_baseline_CI95=interval(deltas) if deltas else None)
    base=np.array([abs(q['gap']-2.5) for q in data['baseline']]);seed=np.array([abs(s['control']['gap']-2.5) for s in data['seeds']]);policies={}
    for policy in ('retained','ordinary'):
        errors={arm:np.array([abs(r['prediction']['crystalformer_gap']-2.5) for r in reports[policy+'-'+arm]['cases']]) for arm in ARMS}
        policies[policy]=assess(errors,base,seed,[r['physical_eligible'] for r in reports[policy+'-descent']['cases']])
    result=dict(role='Independent48-parent/search-seed directional validation; original16 is development only.',policy=VALIDATION_POLICY,parent_ids=plan['parent_ids'],
        primary=policies['retained'],secondary=policies['ordinary'],arms=reports,baseline_predictor_controls=baseline_controls,report_sha256=sha(__file__),plan_sha256=sha(root/'validation-plan.json'),
        costs=dict(NFE_total=data['NFE_total'],NFE_native=data['NFE_native'],NFE_search=data['NFE_search'],
            rays={arm:dict(attempts=sum(r['arm']==arm for r in data['attempts']),eligible=sum(r['arm']==arm and r['eligible'] for r in data['attempts']),
                B_NFE=sum(r.get('B',{}).get('NFE',0) for r in data['attempts'] if r['arm']==arm)) for arm in ARMS}),
        limitations='All48 denominators andfailedattempts retained. Policies share rays andcannot be treated as independent repetitions. CF-only acceptance never uses ALIGNN. Directional evidence is not goal completion without independent physical retention and honest cross-model interpretation; no SUN/DFT/phonon claims.')
    save(root/'validation-report.json',result);print(json.dumps(dict(primary=result['primary'],secondary=result['secondary'],MAE={k:v['summary'] for k,v in reports.items()})),flush=True)


if __name__=='__main__':main()
