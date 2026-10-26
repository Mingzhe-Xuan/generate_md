"""Assess genuine gradient contribution after full relaxation against matched controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from settings import FAMILIES,ARMS,POLICY,assess_gradient


def load(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'basin_search'))
    from policy import physical
    plan=load(root/'plan.json');data=load(root/'direction-results.json');parents=plan['parent_ids']
    if plan['policy']!=POLICY or len(parents)!=16 or data['completed_parents']!=16 or len(data['attempts'])!=576:raise ValueError('Incomplete/frozen cohort violation.')
    expected={(c,f,a,k) for c in parents for f in FAMILIES for a in ARMS for k in (.05,.20,.50,.80)}
    if {(r['parent_id'],r['family'],r['arm'],r['scale']) for r in data['attempts']}!=expected:raise ValueError('Missing/duplicated attempts.')
    old=np.array([abs(c['predictions']['after']['crystalformer_gap']-2.5) for c in plan['cases']]);reports={};arrays={};matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    for arm,points in data['selected'].items():
        folder=root/f'evaluation-{arm}/cross-prediction';metadata=load(folder/'metadata.json')
        if not metadata['crystalformer_control']['control_passed'] or len(metadata['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in metadata['controls']):raise ValueError('Independent frozen evaluator controls failed.')
        predictions={r['candidate_id']:r for r in load(folder/'predictions.json') if r['stage']=='after'}
        if set(predictions)!=set(parents) or len(points)!=16:raise ValueError('Final denominator changed.')
        rows=[]
        for case,pnt in zip(plan['cases'],points):
            pred=predictions[case['parent_id']]
            if pred['geometry_sha256']!=pnt['B']['relaxed_sha256'] or max(abs(pred['crystalformer_gap']-pnt['gap']),abs(pred['crystalformer_eform']-pnt['eform']))>1e-5:raise ValueError('Selected score changed after independent evaluation.')
            first=Structure.from_file(case['B']['relaxed_path']);second=Structure.from_file(pnt['pose'])
            valid=physical(pnt['B'],case['B']['final_energy_eV']/case['B']['num_atoms']) and pnt['eform']<-.5
            rows.append(dict(parent_id=case['parent_id'],point_id=pnt['id'],prediction=pred,physical_eligible=bool(valid),geometry=measure(first,second),author_match=bool(matcher.fit(second,first)),B=pnt['B'],selected_path=pnt['pose']))
        summary=dict(denominator=16)
        for model in ('crystalformer','alignn'):
            known=[r for r in rows if r['prediction'].get(model+'_gap') is not None]
            summary[model]=dict(known=len(known),MAE=float(np.mean([abs(r['prediction'][model+'_gap']-2.5) for r in known])) if known else None,
                gap_pass=sum(r['prediction'][model+'_flags']['gap'] is True for r in rows),eform_pass=sum(r['prediction'][model+'_flags']['eform'] is True for r in rows),joint_pass=sum(r['prediction'][model+'_flags']['joint'] is True and r['physical_eligible'] for r in rows))
        arrays[arm]=np.array([abs(r['prediction']['crystalformer_gap']-2.5) for r in rows]);reports[arm]=dict(summary=summary,cases=rows)
    families={}
    for family in FAMILIES:
        g=arrays[family+'-descent'];valid=[r['physical_eligible'] for r in reports[family+'-descent']['cases']]
        families[family]=assess_gradient(g,arrays[family+'-random'],arrays[family+'-reverse'],old,arrays['seed'],valid)
        families[family]['attempts']={a:dict(total=sum(r['family']==family and r['arm']==a for r in data['attempts']),
            eligible=sum(r['family']==family and r['arm']==a and r['eligible'] for r in data['attempts']),
            B_NFE=sum(r.get('B',{}).get('NFE',0) for r in data['attempts'] if r['family']==family and r['arm']==a)) for a in ARMS}
    result=dict(plan=plan,arms=reports,families=families,interpretation='Stage-one one-gradient/four-step-size benchmark. All16neutral parents, matched reverse/random, shared symmetry seed and zero-step control; every scored endpoint fullcanonicalB. Best4 controls equalbudget; sharedseedbenefit is not gradient contribution. Nativefinite8-step derivative is not fullR derivative. Success requires actualadvantage overmatchedcontrols andretainedgain, followedbyindependentconfirmation/perturbationchecks beforegoalcompletion.')
    result['cost']=dict(probe_and_curvature_NFE=load(root/'native-probe.json')['NFE'],scientific_seed_and_proposal_NFE=data['NFE'],
        note='Probe includes all baseline replays/FD verification/Hessian/lookahead construction. Per-arm B_NFE excludes these common preparation costs. Equal proposal budget is not equal totalFLOPs.')
    (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(families),flush=True)


if __name__=='__main__':main()
