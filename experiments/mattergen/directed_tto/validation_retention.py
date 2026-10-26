"""Matched unconstrained re-quench of every material heldout48 CF-gradient gain."""
import argparse
from contextlib import redirect_stdout
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from probe import load,save,sha
from validation_run import validate_manifest,validate_results,core_hashes
from baseline_amendment import load_amendment
from tight_relaxation import perturb,tight_eligible
from attractors import validate_control
from policy import physical

STAGES=('original','fresh','selected')
RETENTION_POLICY=dict(fmax=.001,maxstep=.05,steps=1600,preserve_symmetry=False,trials=4,
    perturb_atomic_RMS_A=.02,perturb_strain=.003,seed=20261028,
    original_energy_ceiling_eV_atom=.05,original_volume_ratio_bounds=[.65,1.4],
    selection='ALL48 retained-descent endpoints with physical CF gains>=.1 beyond BOTH originalB and feasible seedB; no fresh-B or ALIGNN filtering.',
    comparison='Each selected endpoint versus BOTH originalB and fresh raw-B, with trial-paired perturbations and identical tighter unconstrained relaxation.',
    role='Physical retention diagnostic within selected parents; does not replace independent48 directionality or certify global/DFT/phonon stability.')
SOURCE_FILES=('validation-plan.json','baseline-amendment.json','validation-native.json','validation-results.json','validation.complete.json','validation-report.json')


def helper_hashes():
    here=Path(__file__).resolve().parent;base=here.parent/'basin_search'
    return dict(tight_helpers=sha(here/'tight_relaxation.py'),shared_proposals=sha(base/'proposals.py'),shared_controls=sha(base/'attractors.py'))


def material_indices(plan,data,report):
    parents=plan['parent_ids'];rows=report['arms']['retained-descent']['cases'];result=[]
    if len(parents)!=48 or [r['parent_id'] for r in rows]!=parents:raise ValueError('Require complete48 material-case coverage.')
    for i,(case,row) in enumerate(zip(plan['cases'],rows)):
        q=data['selected']['retained-descent'][i];base=data['baseline'][i];seed=data['seeds'][i]['control']
        valid=physical(q['B'],case['B']['final_energy_eV']/case['B']['num_atoms']) and q['eform']<-.5
        gain=min(abs(base['gap']-2.5),abs(seed['gap']-2.5))-abs(q['gap']-2.5)
        material=bool(valid and gain>=.1)
        if row['point_id']!=q['id'] or row['physical_eligible']!=valid or row['material_CF_gain']!=material or abs(row['retained_CF_gain_eV']-gain)>1e-8:
            raise ValueError('Source material selection/report linkage changed.')
        if material:result.append(i)
    return result


def source_evidence(source):
    plan=load(source/'validation-plan.json');data=load(source/'validation-results.json');native=load(source/'validation-native.json')
    complete=load(source/'validation.complete.json');report=load(source/'validation-report.json')
    validate_manifest(plan,sha(source/'validation-plan.json'));validate_results(plan,data);load_amendment(source,plan)
    if (not complete['completed'] or complete['data_sha256']!=sha(source/'validation-results.json') or not native['passed']
        or native['code_sha256']!=core_hashes() or len(native['cases'])!=48 or not all(c['passed'] for c in native['cases'])
        or report['plan_sha256']!=sha(source/'validation-plan.json') or data['raw_replay_baseline']!=native['raw_replay_baseline']
        or data['baseline_amendment_sha256']!=sha(source/'baseline-amendment.json')):raise ValueError('Independent48 source evidence incomplete/changed.')
    return plan,data,report,material_indices(plan,data,report)


def source_points(data,index):
    return dict(original=data['baseline'][index],fresh=data['raw_replay_baseline'][index],selected=data['selected']['retained-descent'][index])


def validate_trials(parents,records):
    expected={(p,s,t) for p in parents for s in STAGES for t in range(4)}
    if len(set(parents))!=len(parents) or len(records)!=len(expected) or {(r['parent_id'],r['stage'],r['trial']) for r in records}!=expected or len({r['candidate_id'] for r in records})!=len(records):
        raise ValueError('Missing/duplicate three-stage paired trials.')


def retention_eligible(record):
    if not tight_eligible(record):return False
    keys=('final_energy_eV','num_atoms','original_energy_eV_atom','original_volume_ratio')
    if any(record.get(k) is None for k in keys) or not np.isfinite([record[k] for k in keys]).all() or record['num_atoms']<=0:return False
    return bool(.65<=record['original_volume_ratio']<=1.4 and record['final_energy_eV']/record['num_atoms']<=record['original_energy_eV_atom']+.05)


def specification(root):
    spec=load(root/'retention-plan.json');source=Path(spec['source']);physics=load(root/'retention-physics.json');done=load(root/'retention-physics.complete.json')
    if spec['policy']!=RETENTION_POLICY or spec['implementation_sha256']!=sha(__file__) or spec['helper_sha256']!=helper_hashes():raise ValueError('Retention implementation/policy changed.')
    if any(sha(source/name)!=digest for name,digest in spec['source_hashes'].items()):raise ValueError('Source48 artifacts changed.')
    plan,data,report,indices=source_evidence(source)
    if spec['indices']!=indices or spec['parent_ids']!=[plan['parent_ids'][i] for i in indices]:raise ValueError('Material parents changed.')
    validate_trials(spec['parent_ids'],physics['records'])
    expected={(p,s) for p in spec['parent_ids'] for s in STAGES};controls=physics['controls']
    if (not done['completed'] or done['physics_sha256']!=sha(root/'retention-physics.json') or len(controls)!=len(expected)
        or {(r['parent_id'],r['stage']) for r in controls}!=expected or not all(r['passed'] for r in controls)):raise ValueError('Physics controls/complete evidence invalid.')
    return spec,plan,data,report,physics


def physics(source,root):
    plan,data,report,indices=source_evidence(source);root.mkdir(parents=True,exist_ok=False);parents=[plan['parent_ids'][i] for i in indices]
    save(root/'retention-plan.json',dict(policy=RETENTION_POLICY,source=str(source),parent_ids=parents,indices=indices,
        source_hashes={name:sha(source/name) for name in SOURCE_FILES},implementation_sha256=sha(__file__),helper_sha256=helper_hashes(),
        source_directional_evidence=report['primary']['independent_directional_evidence'],source_fresh_baseline_robustness=report['primary_directional_evidence_survives_fresh_rawB']))
    records=[];controls=[];calls=[0];begun=time.perf_counter()
    if indices:
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
        from mattersim_support import validate_atomic_numbers,model_max_z
        checkpoint=Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve();expected=load(smoacs/'postprocess/sources.json')['mattersim']['checkpoint_sha256']
        if sha(checkpoint)!=expected:raise ValueError('Potential checkpoint changed.')
        old=potential.torch;potential.torch=TrustedCheckpointTorch(torch,[expected]);torch.set_num_threads(2)
        try:
            with redirect_stdout(sys.stderr):calc=MatterSimCalculator(load_path=str(checkpoint),device='cpu',load_training_state=False)
        finally:potential.torch=old
        if any(p.is_cuda for p in calc.potential.model.parameters()):raise RuntimeError('CPU retention diagnostic accessed GPU.')
        calc.potential.model.eval();maximum=model_max_z(calc.potential.model);calculate=calc.calculate;templates={}
        def counted(*args,**kwargs):calls[0]+=1;return calculate(*args,**kwargs)
        calc.calculate=counted
        for i in indices:
            parent=plan['parent_ids'][i]
            for stage,q in source_points(data,i).items():
                if sha(q['pose'])!=q['B']['relaxed_sha256']:raise ValueError('Archived source pose changed.')
                atoms=read(q['pose'],format='vasp');atoms.set_constraint(None);validate_atomic_numbers(atoms.numbers,maximum);atoms.calc=calc
                energy=float(atoms.get_potential_energy());force=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max())
                controls.append(dict(parent_id=parent,stage=stage,source_sha256=q['B']['relaxed_sha256'],**validate_control(energy,force,q['B'],expected)))
                templates[parent,stage]=atoms.copy()
        save(root/'retention-controls.json',dict(passed=True,device='cpu',checkpoint_sha256=expected,cases=controls))
        original_points={plan['parent_ids'][i]:data['baseline'][i] for i in indices}
        for parent in parents:
            parent_index=int(Path(parent).stem.split('-')[1])
            original_B=original_points[parent]['B'];original_energy=original_B['final_energy_eV']/original_B['num_atoms'];original_volume=original_B['volume_A3']
            for stage in STAGES:
                reference=templates[parent,stage]
                for trial in range(4):
                    folder=root/Path(parent).stem/stage/f'trial-{trial}';folder.mkdir(parents=True,exist_ok=False);path=folder/'input.vasp'
                    write(path,reference,format='vasp',direct=True,sort=False);start=time.perf_counter();nfe=calls[0]
                    row=dict(candidate_id=f'{Path(parent).stem}-{stage}-t{trial}.vasp',parent_id=parent,stage=stage,trial=trial,input=str(path),status='failed',checkpoint_sha256=expected,
                        original_energy_eV_atom=original_energy,original_volume_A3=original_volume)
                    try:
                        atoms=perturb(reference,parent_index,trial);write(path,atoms,format='vasp',direct=True,sort=False);atoms.calc=calc
                        filtered=ExpCellFilter(atoms);opt=BFGS(filtered,logfile=str(folder/'optimizer.log'),maxstep=.05);converged=bool(opt.run(fmax=.001,steps=1600))
                        raw=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max());force=float(np.linalg.norm(filtered.get_forces(),axis=1).max());energy=float(atoms.get_potential_energy())
                        if not np.array_equal(atoms.numbers,reference.numbers) or not np.isfinite([raw,force,energy]).all():raise ValueError('Species/nonfinite tight result.')
                        terminal=folder/'relaxed.vasp';write(terminal,atoms,format='vasp',direct=True,sort=False)
                        row.update(status='converged' if converged else 'unconverged',converged=converged,num_atoms=len(atoms),final_energy_eV=energy,
                            max_filter_force=force,max_raw_atomic_force=raw,separated=not len(neighbor_list('d',atoms,.5-1e-9)),optimizer_steps=opt.get_number_of_steps(),
                            volume_A3=atoms.get_volume(),volume_ratio=atoms.get_volume()/reference.get_volume(),original_volume_ratio=atoms.get_volume()/original_volume,
                            relaxed_path=str(terminal),relaxed_sha256=sha(terminal))
                    except Exception as error:row['error']=repr(error)
                    row.update(input_sha256=sha(path),NFE=calls[0]-nfe,elapsed_seconds=time.perf_counter()-start,tight_eligible=tight_eligible(row),retention_eligible=retention_eligible(row))
                    records.append(row);save(folder/'result.json',row)
                    save(root/'retention-physics.json',dict(records=records,controls=controls,NFE=calls[0],elapsed_seconds=time.perf_counter()-begun))
                    print(json.dumps(dict(parent_id=parent,stage=stage,trial=trial,status=row['status'],tight_eligible=row['tight_eligible'])),flush=True)
    if not indices:save(root/'retention-physics.json',dict(records=[],controls=[],NFE=0,elapsed_seconds=0.,reason='No material CF endpoints among the full48 parents.'))
    validate_trials(parents,records);save(root/'retention-physics.complete.json',dict(completed=True,parents=len(parents),trials=len(records),physics_sha256=sha(root/'retention-physics.json')))


def export_trials(root,plan,records,scorer):
    destination=root/'evaluation-retention';destination.mkdir(exist_ok=False);inventory=destination/'inventory-249';inventory.mkdir();cases={c['parent_id']:c for c in plan['cases']};items=[];rows=[]
    for r in records:
        case=cases[r['parent_id']];identifier=r['candidate_id'];path=Path(r['input'])
        if sha(path)!=r['input_sha256']:raise ValueError('Trial input changed.')
        gap,ef=scorer.predict(path);B=dict(r);folder=destination/'relaxation'/Path(identifier).stem;folder.mkdir(parents=True,exist_ok=False)
        if r.get('relaxed_path'):
            if sha(r['relaxed_path'])!=r['relaxed_sha256']:raise ValueError('Trial terminal pose changed.')
            shutil.copy2(r['relaxed_path'],folder/'relaxed.vasp');B['relaxed_path']=str((folder/'relaxed.vasp').resolve())
        save(folder/'result.json',B)
        items.append(dict(case['inventory'],candidate_id=identifier,source=str(path),sha256=r['input_sha256'],original_gap=gap,original_eform=ef,original_valid=str(case['neutral'])))
        rows.append(dict(original_fnames=identifier,bandgap_onehot=gap,ef_onehot=ef,is_neutral_cmn=str(case['neutral']),**{'minbond_less_than_0.5':'True'},valid_structure=str(case['neutral'])))
    with (destination/'result.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['original_fnames','bandgap_onehot','ef_onehot','is_neutral_cmn','minbond_less_than_0.5','valid_structure']);writer.writeheader();writer.writerows(rows)
    save(inventory/'inventory.json',dict(role='Three-stage correlated physical-retention trials',denominator=len(items),candidates=items,source_csv_sha256=sha(destination/'result.csv')))


def prepare(root,spec,plan,data,work):
    if not spec['parent_ids']:return
    from cf_gradient import GradientScorer
    scorer=GradientScorer();controls=[]
    for i in spec['indices']:
        for stage,q in source_points(data,i).items():
            if sha(q['pose'])!=q['B']['relaxed_sha256']:raise ValueError('Original pose changed before CF scoring.')
            gap,ef=scorer.predict(Path(q['pose']));error=max(abs(gap-q['gap']),abs(ef-q['eform']))
            if error>1e-5:raise ValueError('Frozen source CF control failed.')
            controls.append(dict(parent_id=plan['parent_ids'][i],stage=stage,error=error))
    export_trials(root,plan,work['records'],scorer);save(root/'retention-CF-controls.json',dict(passed=True,cases=controls,CF=scorer.metadata))


def paired_summary(parents,records,predictions):
    validate_trials(parents,records)
    if set(predictions)!={r['candidate_id'] for r in records}:raise ValueError('Retention prediction coverage changed.')
    bykey={(r['parent_id'],r['stage'],r['trial']):r for r in records};result=[]
    for parent in parents:
        trials=[]
        for trial in range(4):
            stages={s:bykey[parent,s,trial] for s in STAGES};models={}
            for model in ('crystalformer','alignn'):
                gaps={s:predictions[r['candidate_id']].get(model+'_gap') for s,r in stages.items()};comparisons={}
                for control in ('original','fresh'):
                    pair=[stages[control],stages['selected']]
                    feasible=all(retention_eligible(r) and predictions[r['candidate_id']].get('crystalformer_eform') is not None and predictions[r['candidate_id']]['crystalformer_eform']<-.5 for r in pair)
                    known=gaps[control] is not None and gaps['selected'] is not None
                    comparisons[control]=dict(known=known,both_tight_eligible=feasible,error_gain_eV=abs(gaps[control]-2.5)-abs(gaps['selected']-2.5) if known else None)
                models[model]=dict(gaps=gaps,comparisons=comparisons)
            trials.append(dict(trial=trial,stages=stages,models=models))
        summary={}
        for model in ('crystalformer','alignn'):
            summary[model]={}
            for control in ('original','fresh'):
                entries=[r['models'][model]['comparisons'][control] for r in trials];gains=[r['error_gain_eV'] for r in entries if r['known'] and r['both_tight_eligible']]
                summary[model][control]=dict(trials=4,known_pairs=sum(r['known'] for r in entries),eligible_known_pairs=len(gains),
                    min_gain_eV=min(gains) if gains else None,median_gain_eV=float(np.median(gains)) if gains else None,all_four_gains_ge_0p1=len(gains)==4 and min(gains)>=.1)
        result.append(dict(parent_id=parent,summary=summary,trials=trials))
    return result


def report(root,spec,work):
    if spec['parent_ids']:
        controls=load(root/'retention-CF-controls.json');expected={(p,s) for p in spec['parent_ids'] for s in STAGES}
        if not controls['passed'] or len(controls['cases'])!=len(expected) or {(r['parent_id'],r['stage']) for r in controls['cases']}!=expected or any(r['error']>1e-5 for r in controls['cases']):raise ValueError('Frozen source CF controls missing/failed.')
        folder=root/'evaluation-retention/cross-prediction';metadata=load(folder/'metadata.json')
        if not metadata['crystalformer_control']['control_passed'] or len(metadata['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in metadata['controls']):raise ValueError('Independent evaluator controls failed.')
        rows=[r for r in load(folder/'predictions.json') if r['stage']=='after'];predictions={r['candidate_id']:r for r in rows}
        if len(predictions)!=len(rows):raise ValueError('Duplicate trial predictions.')
        for r in work['records']:
            pred=predictions[r['candidate_id']]
            if r.get('relaxed_path') and sha(r['relaxed_path'])!=r['relaxed_sha256']:raise ValueError('Final trial pose changed.')
            if r['status']=='converged' and pred.get('geometry_sha256')!=r['relaxed_sha256']:raise ValueError('Final prediction/pose linkage changed.')
        parents=paired_summary(spec['parent_ids'],work['records'],predictions)
    else:parents=[]
    robust={model:sum(all(p['summary'][model][s]['all_four_gains_ge_0p1'] for s in ('original','fresh')) for p in parents) for model in ('crystalformer','alignn')}
    save(root/'retention-report.json',dict(policy=RETENTION_POLICY,source_directional_evidence=spec['source_directional_evidence'],
        source_fresh_baseline_robustness=spec['source_fresh_baseline_robustness'],original_population=48,selected_material_parents=len(parents),
        robust_parents_against_both_matched_baselines=robust,parents=parents,NFE=work['NFE'],goal_completion_proven=False,
        note='All material parents retained, including failed/unknown trials. Four correlated perturbations per stage are not independent parents. No material parents is an explicit negative result.'))
    print(json.dumps(dict(material_parents=len(parents),robust_against_both_baselines=robust)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--mode',choices=('physics','prepare','report'),required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm; prepare requires a GPU allocation.')
    root=a.run.resolve()
    if a.mode=='physics':
        if a.source is None:p.error('physics requires --source')
        physics(a.source.resolve(),root)
    else:
        spec,plan,data,_,work=specification(root)
        if a.mode=='prepare':prepare(root,spec,plan,data,work)
        else:report(root,spec,work)


if __name__=='__main__':main()
