"""Independent final scoring of three adaptive CF-feedback direction policies."""
import argparse
import json
from pathlib import Path
import numpy as np
from feedback import FEEDBACK_POLICY,validate_source,validate_feedback_results
from settings import ARMS,assess_gradient,interval
from probe import load,save,sha
from policy import physical


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    specification=load(root/'feedback-plan.json');plan=specification['source_plan'];parents=plan['parent_ids'];source=Path(specification['source_root'])
    if specification['feedback_sha256']!=sha(Path(__file__).with_name('feedback.py')) or specification['report_sha256']!=sha(__file__):raise ValueError('Feedback implementation changed after run freeze.')
    if specification['policy']!=FEEDBACK_POLICY or not load(root/'feedback.complete.json')['completed']:raise ValueError('Incomplete/frozen feedback policy changed.')
    if any(sha(source/n)!=digest for n,digest in specification['source_hashes'].items()):raise ValueError('Source stage-one evidence changed.')
    original=load(source/'direction-results.json');initial=validate_source(plan,original,load(source/'native-probe.json'));data=load(root/'feedback-results.json')
    if data['initial']!=initial:raise ValueError('First-round exact cached choices changed.')
    if data['baseline']!=original['baseline']:raise ValueError('Immutable baseline reference changed.')
    validate_feedback_results(plan,data)
    reports={};errors={};matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    for arm in ARMS:
        folder=root/f'evaluation-{arm}/cross-prediction';metadata=load(folder/'metadata.json')
        if not metadata['crystalformer_control']['control_passed'] or len(metadata['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in metadata['controls']):raise ValueError('Frozen final evaluator control failed.')
        predictions={r['candidate_id']:r for r in load(folder/'predictions.json') if r['stage']=='after'}
        if set(predictions)!=set(parents) or len(data['selected'][arm])!=16:raise ValueError('Final16 denominator changed.')
        rows=[]
        for i,(case,pnt) in enumerate(zip(plan['cases'],data['selected'][arm])):
            pred=predictions[case['parent_id']];first=Structure.from_file(case['B']['relaxed_path']);second=Structure.from_file(pnt['pose'])
            if sha(pnt['pose'])!=pnt['B']['relaxed_sha256'] or pred['geometry_sha256']!=pnt['B']['relaxed_sha256'] or tuple(first.atomic_numbers)!=tuple(second.atomic_numbers):raise ValueError('Final pose/species provenance changed.')
            if max(abs(pred['crystalformer_gap']-pnt['gap']),abs(pred['crystalformer_eform']-pnt['eform']))>1e-5:raise ValueError('Independent CF selected score mismatch.')
            energy=case['B']['final_energy_eV']/case['B']['num_atoms'];valid=physical(pnt['B'],energy) and pnt['eform']<-.5
            rows.append(dict(parent_id=case['parent_id'],point_id=pnt['id'],prediction=pred,physical_eligible=bool(valid),
                geometry=measure(first,second),author_match=bool(matcher.fit(second,first)),B=pnt['B'],selected_path=pnt['pose'],
                first_round_error_eV=abs(data['initial'][arm][i]['gap']-2.5)))
        summary={}
        for model in ('crystalformer','alignn'):
            known=[r for r in rows if r['prediction'].get(model+'_gap') is not None]
            summary[model]=dict(denominator=16,known=len(known),MAE=float(np.mean([abs(r['prediction'][model+'_gap']-2.5) for r in known])) if known else None,
                gap_pass=sum(r['prediction'][model+'_flags']['gap'] is True for r in rows),eform_pass=sum(r['prediction'][model+'_flags']['eform'] is True for r in rows),
                joint_pass=sum(r['prediction'][model+'_flags']['joint'] is True and r['physical_eligible'] for r in rows))
            paired=[(r,case) for r,case in zip(rows,plan['cases']) if r['prediction'].get(model+'_gap') is not None and case['predictions']['after'].get(model+'_gap') is not None]
            if paired:
                base=np.array([abs(c['predictions']['after'][model+'_gap']-2.5) for r,c in paired]);final=np.array([abs(r['prediction'][model+'_gap']-2.5) for r,c in paired])
                summary[model].update(paired_baseline_parents=len(paired),paired_baseline_MAE=float(base.mean()),MAE_delta_vs_baseline=float((final-base).mean()),MAE_delta_vs_baseline_CI95=interval(final-base))
        errors[arm]=np.array([abs(r['prediction']['crystalformer_gap']-2.5) for r in rows]);first=np.array([r['first_round_error_eV'] for r in rows])
        reports[arm]=dict(summary=summary,cases=rows,third_vs_first_MAE_delta=float((errors[arm]-first).mean()),third_vs_first_CI95=interval(errors[arm]-first))
    baseline=np.array([abs(c['predictions']['after']['crystalformer_gap']-2.5) for c in plan['cases']]);seed=np.array([abs(p['gap']-2.5) for p in original['selected']['seed']])
    assessment=assess_gradient(errors['descent'],errors['random'],errors['reverse'],baseline,seed,[r['physical_eligible'] for r in reports['descent']['cases']])
    costs=dict(source_all_methods_NFE=original['NFE'],cached_plain_ray_NFE=sum(r.get('B',{}).get('NFE',0) for r in original['attempts'] if r['family']=='plain'),
        cached_shared_seed_NFE=sum(s['B']['NFE'] for s in original['seeds']),fresh_feedback_NFE=data['NFE'],
        note='Round1 reuses exact completed stage-one plain family; two additional rounds have384rayattempts+96zero-step seedquenches. No hidden discarded failed attempts.')
    save(root/'feedback-report.json',dict(policy=FEEDBACK_POLICY,parents=parents,arms=reports,assessment=assessment,costs=costs,
        interpretation='Three-round adaptive gradient TTO; directions recomputed from each own fully relaxed incumbent, same seed schedule and proposal budgets. Compare descent with reverse/random, not merely round3 with round1. Exploratory development requires independent confirmation if positive; no SUN/DFT claims.'))
    print(json.dumps(dict(assessment=assessment,MAE={a:reports[a]['summary']['crystalformer']['MAE'] for a in ARMS})),flush=True)


if __name__=='__main__':main()
