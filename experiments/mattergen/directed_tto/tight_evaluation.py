"""Frozen GPU CF/ALIGNN evaluation of paired tighter-force diagnostic endpoints."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from tight_relaxation import TIGHT_POLICY,validate_trials,tight_eligible
from probe import load,save,sha
from cf_gradient import GradientScorer


def specification(root):
    spec=load(root/'tight-plan.json');source=Path(spec['source'])
    if spec['policy']!=TIGHT_POLICY or spec['evaluation_sha256']!=sha(__file__) or spec['mechanics_sha256']!=sha(Path(__file__).with_name('tight_relaxation.py')):raise ValueError('Tight diagnostic implementation changed.')
    if any(sha(source/n)!=digest for n,digest in spec['source_hashes'].items()):raise ValueError('Source experiment changed.')
    if not load(root/'tight-physics.complete.json')['completed']:raise ValueError('Tight mechanics incomplete.')
    physics=load(root/'tight-physics.json');validate_trials(spec['parent_ids'],physics['records'])
    if len(physics['controls'])!=2*len(spec['parent_ids']) or not all(c['passed'] for c in physics['controls']):raise ValueError('CPU/GPU controls incomplete.')
    return spec,source,physics


def prepare(root,spec,source,physics):
    scorer=GradientScorer();plan=load(source/'plan.json');data=load(source/'direction-results.json');cases={c['parent_id']:c for c in plan['cases']};controls=[]
    for i in spec['indices']:
        case=plan['cases'][i];parent=case['parent_id'];selected=data['selected']['plain-descent'][i]
        for stage,path,expected,digest in (('direct',case['B']['relaxed_path'],case['predictions']['after'],case['B']['relaxed_sha256']),('selected',selected['pose'],dict(crystalformer_gap=selected['gap'],crystalformer_eform=selected['eform']),selected['B']['relaxed_sha256'])):
            if sha(path)!=digest:raise ValueError('Original source pose changed before scoring.')
            gap,ef=scorer.predict(Path(path));error=max(abs(gap-expected['crystalformer_gap']),abs(ef-expected['crystalformer_eform']))
            if error>1e-5:raise ValueError('Original frozen CF control failed.')
            controls.append(dict(parent_id=parent,stage=stage,error=error))
    destination=root/'evaluation-tight';destination.mkdir(exist_ok=False);inventory=destination/'inventory-249';inventory.mkdir();rows=[];items=[]
    for r in physics['records']:
        case=cases[r['parent_id']];identifier=r['candidate_id'];gap,ef=scorer.predict(Path(r['input']));B=dict(r)
        if sha(r['input'])!=r['input_sha256']:raise ValueError('Diagnostic input changed.')
        folder=destination/'relaxation'/Path(identifier).stem;folder.mkdir(parents=True,exist_ok=False)
        if r.get('relaxed_path'):
            if sha(r['relaxed_path'])!=r['relaxed_sha256']:raise ValueError('Tighter terminal pose changed.')
            shutil.copy2(r['relaxed_path'],folder/'relaxed.vasp');B['relaxed_path']=str((folder/'relaxed.vasp').resolve())
        save(folder/'result.json',B)
        items.append(dict(case['inventory'],candidate_id=identifier,source=r['input'],sha256=r['input_sha256'],original_gap=gap,original_eform=ef,original_valid=str(case['neutral'])))
        rows.append(dict(original_fnames=identifier,bandgap_onehot=gap,ef_onehot=ef,is_neutral_cmn=str(case['neutral']),**{'minbond_less_than_0.5':'True'},valid_structure=str(case['neutral'])))
    with (destination/'result.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(inventory/'inventory.json',dict(role='Paired tight-force diagnostic, correlated trials within parent',denominator=len(items),candidates=items,source_csv_sha256=sha(destination/'result.csv')))
    save(root/'tight-CF-controls.json',dict(passed=True,cases=controls,CF=scorer.metadata))


def report(root,spec,source,physics):
    controls=load(root/'tight-CF-controls.json')
    if not controls['passed'] or len(controls['cases'])!=2*len(spec['parent_ids']) or any(c['error']>1e-5 for c in controls['cases']):raise ValueError('Original CF controls missing/failed.')
    folder=root/'evaluation-tight/cross-prediction';metadata=load(folder/'metadata.json')
    if not metadata['crystalformer_control']['control_passed'] or len(metadata['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in metadata['controls']):raise ValueError('Independent frozen evaluator control failed.')
    predictions={r['candidate_id']:r for r in load(folder/'predictions.json') if r['stage']=='after'}
    if set(predictions)!={r['candidate_id'] for r in physics['records']}:raise ValueError('Tight scoring coverage changed.')
    records={(r['parent_id'],r['stage'],r['trial']):r for r in physics['records']};parents=[]
    for parent in spec['parent_ids']:
        pairs=[]
        for trial in range(4):
            direct=records[parent,'direct',trial];selected=records[parent,'selected',trial];entry=dict(trial=trial,direct=direct,selected=selected,models={})
            for r in (direct,selected):
                pred=predictions[r['candidate_id']]
                if r.get('relaxed_path') and (sha(r['relaxed_path'])!=r['relaxed_sha256'] or pred.get('geometry_sha256') not in (None,r['relaxed_sha256'])):raise ValueError('Tight final geometry linkage changed.')
            for model in ('crystalformer','alignn'):
                x=predictions[direct['candidate_id']].get(model+'_gap');y=predictions[selected['candidate_id']].get(model+'_gap')
                known=x is not None and y is not None
                entry['models'][model]=dict(direct_gap=x,selected_gap=y,error_gain_eV=abs(x-2.5)-abs(y-2.5) if known else None)
            forms=[predictions[r['candidate_id']].get('crystalformer_eform') for r in (direct,selected)]
            entry['both_force_converged']=tight_eligible(direct) and tight_eligible(selected)
            entry['both_tight_eligible']=entry['both_force_converged'] and all(v is not None and v<-.5 for v in forms);pairs.append(entry)
        summary={}
        for model in ('crystalformer','alignn'):
            gains=[r['models'][model]['error_gain_eV'] for r in pairs if r['both_tight_eligible'] and r['models'][model]['error_gain_eV'] is not None]
            summary[model]=dict(known_tight_pairs=len(gains),median_gain_eV=float(np.median(gains)) if gains else None,min_gain_eV=min(gains) if gains else None,
                all_four_gains_ge_0p1=len(gains)==4 and min(gains)>=.1)
        parents.append(dict(parent_id=parent,summary=summary,trials=pairs))
    save(root/'tight-report.json',dict(policy=TIGHT_POLICY,parent_count=len(parents),parents=parents,
        note='Correlated perturbation trials within each selected parent; no16-IID inference. Original .01-force experiment remains unchanged. This .001-force unconstrained free-cell diagnostic is not DFT/phonon or global thermodynamic stability.'))
    print(json.dumps({r['parent_id']:r['summary'] for r in parents}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--mode',choices=('prepare','report'),required=True);a=p.parse_args()
    import os
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm; CF preparation requires GPU allocation.')
    spec,source,physics=specification(a.run)
    (prepare if a.mode=='prepare' else report)(a.run,spec,source,physics)


if __name__=='__main__':main()
