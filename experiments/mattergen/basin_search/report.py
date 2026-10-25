"""Evaluate full16-parent post-quench property gains and material changes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from policy import METHODS,ROUNDS,WIDTH,physical


def load(p):return json.loads(Path(p).read_text())


def audit_attempts(history,parents):
    ledger={}
    for event in history:
        stage='B' if 'B' in event else 'MD' if 'MD' in event else 'proposal'
        row=event.get(stage,event);identifier=row.get('candidate_id',row.get('id'))
        if identifier is None:raise ValueError('Unidentified search attempt.')
        key=(event['method'],identifier);ledger.setdefault(key,{})[stage]=row
    for method in METHODS:
        expected={Path(p).stem+f'-{method}-g{g:02}-c{k}' for p in parents for g in range(ROUNDS) for k in range(WIDTH)}
        actual={identifier for m,identifier in ledger if m==method}
        if actual!=expected:raise ValueError('Attempt budget changed or failures were dropped.')
        for identifier in expected:
            stages=ledger[(method,identifier)]
            if 'MD' in stages and stages['MD']['status']=='escaped' and 'B' not in stages:raise ValueError('Successful escape has no quench outcome.')
    return ledger


def interval(values):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(42)
    return np.quantile(values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1),[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    plan=load(root/'plan.json');search=load(root/'search-results.json');acceptance=plan['acceptance'];parents=plan['parent_ids']
    if len(parents)!=16 or not load(root/'native-controls.json')['passed']:raise ValueError('Native baseline controls missing.')
    ledger=audit_attempts(search['history'],parents);reports={}
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    for method,points in search['selected'].items():
        evaluation=root/f'evaluation-{method}';metadata=load(evaluation/'cross-prediction/metadata.json')
        if not metadata['crystalformer_control']['control_passed'] or len(metadata['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in metadata['controls']):raise ValueError('Independent frozen evaluator controls failed.')
        predictions={r['candidate_id']:r for r in load(evaluation/'cross-prediction/predictions.json') if r['stage']=='after'}
        if set(predictions)!=set(parents) or len(points)!=16:raise ValueError('Final16-parent coverage changed.')
        rows=[]
        for case,point,basepoint in zip(plan['cases'],points,search['baseline']):
            parent=case['parent_id'];row=predictions[parent];old=case['predictions']['after'];B=point['B'];base_energy=case['B']['final_energy_eV']/case['B']['num_atoms']
            if row['geometry_sha256']!=B['relaxed_sha256'] or max(abs(row['crystalformer_gap']-point['gap']),abs(row['crystalformer_eform']-point['eform']))>1e-5:raise ValueError('Selected score did not survive frozen final evaluation.')
            first=Structure.from_file(case['B']['relaxed_path']);second=Structure.from_file(point['pose']);metrics=measure(first,second);match=bool(matcher.fit(second,first))
            valid=physical(B,base_energy) and point['eform']<-.5
            improvement=abs(old['crystalformer_gap']-2.5)-abs(row['crystalformer_gap']-2.5)
            changed=not match and metrics['species_pair_distance_RMS_A']>=acceptance['min_pair_distance_RMS_A']
            before_physical=physical(basepoint['B'],base_energy)
            rows.append(dict(parent_id=parent,point_id=point['id'],baseline_prediction=old,prediction=row,physical_eligible=valid,
                baseline_physical_eligible=before_physical,author_match=match,geometry=metrics,material_change=changed,
                material_CF_improvement=valid and changed and improvement>=acceptance['min_CF_error_improvement_eV'],
                CF_error_improvement_eV=improvement,energy_delta_eV_atom=B['final_energy_eV']/B['num_atoms']-base_energy,
                baseline_path=case['B']['relaxed_path'],selected_path=point['pose']))
        summary=dict(denominator=16,material_changed=sum(r['material_change'] for r in rows),material_CF_improved=sum(r['material_CF_improvement'] for r in rows),physical_selected=sum(r['physical_eligible'] for r in rows))
        for model in ('crystalformer','alignn'):
            known=[r for r in rows if r['prediction'].get(model+'_gap') is not None]
            b=np.array([abs(r['baseline_prediction'][model+'_gap']-2.5) for r in known]);a=np.array([abs(r['prediction'][model+'_gap']-2.5) for r in known])
            joint_before=sum(r['baseline_prediction'][model+'_flags']['joint'] and r['baseline_physical_eligible'] for r in rows)
            joint_after=sum(r['prediction'][model+'_flags']['joint'] and r['physical_eligible'] for r in rows)
            summary[model]=dict(known=len(known),MAE_before=float(b.mean()) if len(b) else None,MAE_after=float(a.mean()) if len(a) else None,
                delta_CI95=interval(a-b) if len(a) else None,joint_before=joint_before,joint_after=joint_after,joint_net_gain=joint_after-joint_before)
        attempts=[stages for (m,_),stages in ledger.items() if m==method]
        summary['cost']=dict(proposal_attempts=len(attempts),B_calls=sum('B' in s for s in attempts),MD_calls=sum('MD' in s for s in attempts),
            MatterSim_NFE=sum(r.get('NFE',0) for stages in attempts for r in stages.values()),mechanics_seconds=sum(r.get('elapsed_seconds',0) for stages in attempts for r in stages.values()))
        cf=summary['crystalformer'];success=(summary['material_CF_improved']>=acceptance['min_changed_improved_parents'] and
            (cf['joint_net_gain']>=acceptance['min_joint_net_gain'] or cf['delta_CI95'][1]<0))
        summary['discovery_gate_passed']=bool(success)
        summary['confirmation_needed']=bool(success and search['search_seed']!=acceptance['confirmation_seed'])
        reports[method]=dict(summary=summary,cases=rows)
    result=dict(plan=plan,search_seed=search['search_seed'],methods=reports,
        interpretation='All16 parents retained. CF ranks post-quench poses and is the optimization objective; ALIGNN held out. Material change requires author matcher failure plus species-pair distance RMS>=0.1A. Acceptance additionally requires>=4 material CF gains>=0.1eV and either netjointgain>=3 or negative pairedCFMAECI95. Discovery is exploratory; an independent search seed must confirm before goal completion. No SUN/DFT certification.')
    (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['#16-parent basin-search discovery','',result['interpretation'],'',
        '| Method | Physical /16 | Material changed | Material CF improved | CF MAE before -> after | CF joint before -> after | ALIGNN MAE before -> after | Discovery gate |',
        '|---|---:|---:|---:|---|---|---|---|']
    for method,r in reports.items():
        s=r['summary'];cf=s['crystalformer'];al=s['alignn']
        lines.append(f"| {method} | {s['physical_selected']} | {s['material_changed']} | {s['material_CF_improved']} | {cf['MAE_before']:.5f} -> {cf['MAE_after']:.5f} | {cf['joint_before']} -> {cf['joint_after']} | {al['MAE_before']:.5f} -> {al['MAE_after']:.5f} | {s['discovery_gate_passed']} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({m:r['summary'] for m,r in reports.items()}),flush=True)


if __name__=='__main__':main()
