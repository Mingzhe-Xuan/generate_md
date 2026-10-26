"""Stricter, matched physical checks of all material post-quench gradient updates."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from guided_common import load,save,sha,validate_plan,implementation_hashes,GradientScorer
from guided_settings import FAMILIES
from guided_validate import validate_study
from tight_relaxation import perturb
from validation_retention import export_trials,retention_eligible
from attractors import validate_control

BASE_STAGES=('original','probes','reverse','random','raw_cf','selected')
POLICY=dict(fmax=.001,maxstep=.05,cap=1600,preserve_symmetry=False,trials=4,atomic_RMS_A=.02,strain=.003,seed=20261028,
    energy_ceiling_eV_atom=.05,original_volume_bounds=[.65,1.4],selection='ALL material>=.1eV updates beyond own probe-best across ALL three families; no ALIGNN filtering.',
    pairing='Same parent/trial random seed across matched endpoints. Each family-parent has eight stages/four trials; trials and families are correlated.',
    role='Conditional physical retention, not a replacement for full16 directionality or independent confirmation.')
SOURCE_FILES=('plan.json','native.json','results.json','complete.json','report.json')


def stages(family):return BASE_STAGES+tuple('other_'+f for f in FAMILIES if f!=family)


def material_cases(plan,data,report):
    parents=plan['parent_ids'];cases=[]
    if len(parents)!=16:raise ValueError('Require original16 population.')
    for family in FAMILIES:
        selected=report['arms'][family+'-descent']['cases'];probes=report['arms'][family+'-probes']['cases'];count=0
        if [r['parent_id'] for r in selected]!=parents or [r['parent_id'] for r in probes]!=parents:raise ValueError('Source family coverage changed.')
        for i,(a,b) in enumerate(zip(selected,probes)):
            q=data['selected'][family+'-descent'][i];p=data['selected'][family+'-probes'][i]
            if a['point_id']!=q['id'] or b['point_id']!=p['id'] or max(abs(a['prediction']['crystalformer_gap']-q['gap']),abs(b['prediction']['crystalformer_gap']-p['gap']))>1e-5:raise ValueError('Source point/CF linkage changed.')
            gain=abs(b['prediction']['crystalformer_gap']-2.5)-abs(a['prediction']['crystalformer_gap']-2.5)
            if a['physical_eligible'] and gain>=.1:cases.append(dict(family=family,parent_id=parents[i],index=i,source_gain_eV=gain));count+=1
        if count!=report['assessment']['families'][family]['material_parents_beyond_probe_pool']:raise ValueError('Source material count changed.')
    return cases


def source_evidence(source):
    plan=load(source/'plan.json');proof=load(source/'native.json');data=load(source/'results.json');done=load(source/'complete.json');report=load(source/'report.json')
    validate_plan(plan);validate_study(plan,proof,data)
    if (not done['completed'] or done['results_sha256']!=sha(source/'results.json') or done['native_sha256']!=sha(source/'native.json')
        or proof['code_sha256']!=implementation_hashes() or report['plan_sha256']!=sha(source/'plan.json')):raise ValueError('Complete guided source changed.')
    return plan,data,report,material_cases(plan,data,report)


def endpoints(data,case):
    i=case['index'];f=case['family'];result=dict(original=data['baseline'][i],probes=data['selected'][f+'-probes'][i],selected=data['selected'][f+'-descent'][i])
    result.update({a:data['selected'][f+'-'+a][i] for a in ('reverse','random','raw_cf')})
    result.update({'other_'+g:data['selected'][g+'-descent'][i] for g in FAMILIES if g!=f});return result


def validate_trials(cases,records):
    expected={(c['family'],c['parent_id'],s,t) for c in cases for s in stages(c['family']) for t in range(4)}
    actual={(r['family'],r['parent_id'],r['stage'],r['trial']) for r in records}
    if len(expected)!=32*len(cases) or len(records)!=len(expected) or actual!=expected or len({r['candidate_id'] for r in records})!=len(records):raise ValueError('Missing/duplicate eight-stage trials.')


def helpers():
    d=Path(__file__).resolve().parents[1]/'directed_tto'
    return {n:sha(d/n) for n in ('tight_relaxation.py','validation_retention.py')}|{'proposals':sha(d.parent/'basin_search/proposals.py'),'controls':sha(d.parent/'basin_search/attractors.py')}


def specification(root):
    spec=load(root/'retention-plan.json');source=Path(spec['source']);work=load(root/'physics.json');done=load(root/'physics.complete.json')
    if spec['policy']!=POLICY or spec['implementation_sha256']!=sha(__file__) or spec['helper_sha256']!=helpers():raise ValueError('Retention implementation changed.')
    if any(sha(source/n)!=h for n,h in spec['source_hashes'].items()):raise ValueError('Guided source artifact changed.')
    plan,data,report,cases=source_evidence(source)
    if cases!=spec['cases'] or not done['completed'] or done['physics_sha256']!=sha(root/'physics.json'):raise ValueError('Retention cohort/complete proof changed.')
    validate_trials(cases,work['records']);expected={(c['family'],c['parent_id'],s) for c in cases for s in stages(c['family'])}
    if len(work['controls'])!=len(expected) or {(r['family'],r['parent_id'],r['stage']) for r in work['controls']}!=expected or not all(r['passed'] for r in work['controls']):raise ValueError('Source potential controls missing/failed.')
    return spec,plan,data,report,work


def physics(source,root):
    plan,data,report,cases=source_evidence(source);root.mkdir(parents=True,exist_ok=False)
    save(root/'retention-plan.json',dict(policy=POLICY,source=str(source),cases=cases,source_hashes={n:sha(source/n) for n in SOURCE_FILES},implementation_sha256=sha(__file__),helper_sha256=helpers(),source_assessment=report['assessment']))
    records=[];controls=[];calls=[0];begun=time.perf_counter()
    if cases:
        import torch
        import ase.constraints
        from ase.filters import Filter,ExpCellFilter
        if not hasattr(ase.constraints,'Filter'):ase.constraints.Filter=Filter
        from ase.io import read,write
        from ase.neighborlist import neighbor_list
        from ase.optimize import BFGS
        from mattersim.forcefield import MatterSimCalculator,potential
        smoacs=Path(__file__).resolve().parents[2]/'smoacs';sys.path[:0]=[str(smoacs),str(smoacs/'postprocess')]
        from alignn_checkpoint import TrustedCheckpointTorch
        from mattersim_support import model_max_z,validate_atomic_numbers
        expected=load(smoacs/'postprocess/sources.json')['mattersim']['checkpoint_sha256'];checkpoint=Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve()
        if sha(checkpoint)!=expected:raise ValueError('Potential changed.')
        old=potential.torch;potential.torch=TrustedCheckpointTorch(torch,[expected]);torch.set_num_threads(2)
        try:
            with redirect_stdout(sys.stderr):calc=MatterSimCalculator(load_path=str(checkpoint),device='cpu',load_training_state=False)
        finally:potential.torch=old
        if any(p.is_cuda for p in calc.potential.model.parameters()):raise RuntimeError('CPU retention accessed GPU.')
        calc.potential.model.eval();maximum=model_max_z(calc.potential.model);calculate=calc.calculate;templates={}
        def counted(*a,**kw):calls[0]+=1;return calculate(*a,**kw)
        calc.calculate=counted
        for case in cases:
            f=case['family'];p=case['parent_id']
            for stage,q in endpoints(data,case).items():
                if sha(q['pose'])!=q['B']['relaxed_sha256']:raise ValueError('Source pose changed.')
                a=read(q['pose'],format='vasp');a.set_constraint(None);validate_atomic_numbers(a.numbers,maximum);a.calc=calc
                e=float(a.get_potential_energy());force=float(np.linalg.norm(a.get_forces(apply_constraint=False),axis=1).max())
                controls.append(dict(family=f,parent_id=p,stage=stage,source_sha256=q['B']['relaxed_sha256'],**validate_control(e,force,q['B'],expected)));templates[f,p,stage]=a.copy()
        for case in cases:
            f=case['family'];p=case['parent_id'];original=data['baseline'][case['index']]['B'];energy=original['final_energy_eV']/original['num_atoms'];volume=original['volume_A3'];index=int(p.split('-')[1].split('.')[0])
            for stage in stages(f):
                reference=templates[f,p,stage]
                for trial in range(4):
                    folder=root/f/Path(p).stem/stage/f'trial-{trial}';folder.mkdir(parents=True,exist_ok=False);path=folder/'input.vasp';write(path,reference,format='vasp',direct=True,sort=False)
                    row=dict(candidate_id=f'{f}-{Path(p).stem}-{stage}-t{trial}.vasp',family=f,parent_id=p,stage=stage,trial=trial,status='failed',input=str(path),checkpoint_sha256=expected,original_energy_eV_atom=energy,original_volume_A3=volume)
                    nfe=calls[0];started=time.perf_counter()
                    try:
                        a=perturb(reference,index,trial);write(path,a,format='vasp',direct=True,sort=False);a.calc=calc;filtered=ExpCellFilter(a);opt=BFGS(filtered,logfile=str(folder/'optimizer.log'),maxstep=.05);converged=bool(opt.run(fmax=.001,steps=1600))
                        raw=float(np.linalg.norm(a.get_forces(apply_constraint=False),axis=1).max());force=float(np.linalg.norm(filtered.get_forces(),axis=1).max());e=float(a.get_potential_energy())
                        if not np.array_equal(a.numbers,reference.numbers) or not np.isfinite([raw,force,e]).all():raise ValueError('Species/nonfinite relaxation.')
                        final=folder/'relaxed.vasp';write(final,a,format='vasp',direct=True,sort=False)
                        row.update(status='converged' if converged else 'unconverged',converged=converged,num_atoms=len(a),final_energy_eV=e,max_raw_atomic_force=raw,max_filter_force=force,
                            separated=not len(neighbor_list('d',a,.5-1e-9)),volume_A3=a.get_volume(),volume_ratio=a.get_volume()/reference.get_volume(),original_volume_ratio=a.get_volume()/volume,
                            relaxed_path=str(final),relaxed_sha256=sha(final),optimizer_steps=opt.get_number_of_steps())
                    except Exception as error:row['error']=repr(error)
                    row.update(input_sha256=sha(path),NFE=calls[0]-nfe,elapsed_seconds=time.perf_counter()-started,retention_eligible=retention_eligible(row));records.append(row);save(folder/'result.json',row)
                    save(root/'physics.json',dict(records=records,controls=controls,NFE=calls[0],elapsed_seconds=time.perf_counter()-begun))
                    print(json.dumps(dict(family=f,parent_id=p,stage=stage,trial=trial,status=row['status'],eligible=row['retention_eligible'])),flush=True)
    else:save(root/'physics.json',dict(records=[],controls=[],NFE=0,elapsed_seconds=0.,reason='No material gradient updates beyond queried probe best.'))
    validate_trials(cases,records);save(root/'physics.complete.json',dict(completed=True,cases=len(cases),trials=len(records),physics_sha256=sha(root/'physics.json')))


def prepare(root,spec,plan,data,work):
    if not spec['cases']:return
    scorer=GradientScorer();controls=[]
    for case in spec['cases']:
        for stage,q in endpoints(data,case).items():
            if sha(q['pose'])!=q['B']['relaxed_sha256']:raise ValueError('Source pose changed before CF controls.')
            gap,ef=scorer.predict(Path(q['pose']));error=max(abs(gap-q['gap']),abs(ef-q['eform']))
            if error>1e-5:raise ValueError('Source frozen CF control failed.')
            controls.append(dict(family=case['family'],parent_id=case['parent_id'],stage=stage,error=error))
    export_trials(root,plan,work['records'],scorer)
    inventory=root/'evaluation-retention/inventory-249/inventory.json';items=load(inventory);items['role']='Eight-stage correlated gradient-retention trials';save(inventory,items)
    save(root/'CF-controls.json',dict(passed=True,cases=controls,CF=scorer.metadata))


def summarize(cases,records,predictions):
    validate_trials(cases,records)
    if set(predictions)!={r['candidate_id'] for r in records}:raise ValueError('Prediction coverage changed.')
    bykey={(r['family'],r['parent_id'],r['stage'],r['trial']):r for r in records};result=[]
    for case in cases:
        f=case['family'];p=case['parent_id'];models={}
        for model in ('crystalformer','alignn'):
            models[model]={}
            for control in stages(f):
                if control=='selected':continue
                pairs=[]
                for t in range(4):
                    a=bykey[f,p,control,t];b=bykey[f,p,'selected',t];x=predictions[a['candidate_id']];y=predictions[b['candidate_id']];known=x.get(model+'_gap') is not None and y.get(model+'_gap') is not None
                    valid=all(retention_eligible(r) and predictions[r['candidate_id']].get('crystalformer_eform') is not None and predictions[r['candidate_id']]['crystalformer_eform']<-.5 for r in (a,b))
                    pairs.append(dict(trial=t,known=known,eligible=valid,gain_eV=abs(x[model+'_gap']-2.5)-abs(y[model+'_gap']-2.5) if known else None))
                values=[q['gain_eV'] for q in pairs if q['known'] and q['eligible']]
                models[model][control]=dict(trials=4,known_pairs=sum(q['known'] for q in pairs),eligible_known_pairs=len(values),median_gain_eV=float(np.median(values)) if values else None,
                    min_gain_eV=min(values) if values else None,all_four_gain_ge_0p1=len(values)==4 and min(values)>=.1,all_four_gain_gt_1meV=len(values)==4 and min(values)>.001,pairs=pairs)
        result.append(dict(**case,models=models))
    return result


def report(root,spec,work):
    if spec['cases']:
        cf=load(root/'CF-controls.json');expected={(c['family'],c['parent_id'],s) for c in spec['cases'] for s in stages(c['family'])}
        if not cf['passed'] or len(cf['cases'])!=len(expected) or {(r['family'],r['parent_id'],r['stage']) for r in cf['cases']}!=expected or any(r['error']>1e-5 for r in cf['cases']):raise ValueError('Source CF controls incomplete.')
        folder=root/'evaluation-retention/cross-prediction';meta=load(folder/'metadata.json')
        if not meta['crystalformer_control']['control_passed'] or len(meta['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in meta['controls']):raise ValueError('Independent model controls failed.')
        rows=[r for r in load(folder/'predictions.json') if r['stage']=='after'];pred={r['candidate_id']:r for r in rows}
        if len(pred)!=len(rows):raise ValueError('Duplicate predictions.')
        for r in work['records']:
            if sha(r['input'])!=r['input_sha256']:raise ValueError('Physical input changed.')
            if r.get('relaxed_path') and sha(r['relaxed_path'])!=r['relaxed_sha256']:raise ValueError('Physical terminal pose changed.')
            if r['status']=='converged' and pred[r['candidate_id']].get('geometry_sha256')!=r['relaxed_sha256']:raise ValueError('Terminal prediction SHA changed.')
        cases=summarize(spec['cases'],work['records'],pred)
    else:cases=[]
    counts={}
    for f in FAMILIES:
        selected=[c for c in cases if c['family']==f];counts[f]={}
        for model in ('crystalformer','alignn'):
            robust=[c for c in selected if all(c['models'][model][s]['all_four_gain_ge_0p1'] for s in ('original','probes'))]
            counts[f][model]=dict(original_parent_denominator=16,material_cases=len(selected),retained_vs_original_and_probes=len(robust),
                retained_and_beats_direction_controls=sum(all(c['models'][model][s]['all_four_gain_gt_1meV'] for s in ('reverse','random','raw_cf')) for c in robust),
                retained_and_beats_other_family_descents=sum(all(c['models'][model][s]['all_four_gain_gt_1meV'] for s in stages(f) if s.startswith('other_')) for c in robust))
    result=dict(policy=POLICY,source_assessment=spec['source_assessment'],counts=counts,cases=cases,NFE=work['NFE'],goal_completion_proven=False,
        note='ALL material cases kept. No source criterion changed. Correlated trials/families are never pooled as independent parents; these are conditional retention checks.')
    save(root/'report.json',result);print(json.dumps(counts),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--mode',choices=('physics','prepare','report'),required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm; prepare requires GPU.')
    root=a.run.resolve()
    if a.mode=='physics':
        if a.source is None:p.error('physics requires --source')
        physics(a.source.resolve(),root)
    else:
        spec,plan,data,_,work=specification(root)
        if a.mode=='prepare':prepare(root,spec,plan,data,work)
        else:report(root,spec,work)


if __name__=='__main__':main()
