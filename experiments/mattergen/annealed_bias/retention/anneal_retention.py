"""Source-pinned CPU re-quenching and frozen GPU scores of conditional CF gains."""
import argparse
from contextlib import redirect_stdout
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from anneal_retention_policy import POLICY,STAGES,material_cases,endpoints,validate_trials,eligible,summarize
from anneal_experiment import validate
from anneal_shared import load,save,sha,GradientScorer,validate_control
from denoised_shared import basin_search

HERE=Path(__file__).resolve().parent
SOURCE_FILES=('plan.json','native.json','results.json','complete.json','report.json')


def helpers():
    base=HERE.parents[1]
    return dict(policy=sha(HERE/'anneal_retention_policy.py'),experiment=sha(HERE.parent/'anneal_experiment.py'),
        report=sha(HERE.parent/'anneal_report.py'),proposals=sha(base/'basin_search/proposals.py'),controls=sha(base/'basin_search/attractors.py'),
        checkpoint_loader=sha(base.parent/'smoacs/alignn_checkpoint.py'),support=sha(base.parent/'smoacs/postprocess/mattersim_support.py'))


def source_evidence(source):
    plan,native,data,done,report=[load(source/n) for n in SOURCE_FILES]
    validate(source,plan,native,data)
    if (not done['completed'] or not done['population_efficacy_eligible'] or plan['count']!=16
        or any(done[k]!=sha(source/f) for k,f in (('plan_sha256','plan.json'),('native_sha256','native.json'),('results_sha256','results.json')))
        or report['plan_sha256']!=sha(source/'plan.json') or report['report_sha256']!=sha(HERE.parent/'anneal_report.py')):
        raise ValueError('Complete full16 source/provenance required.')
    return plan,native,data,report,material_cases(plan,data,report)


def perturb(reference,parent,trial):
    if trial==0:
        a=reference.copy();a.set_constraint(None);return a
    return basin_search.mutate(reference,reference,np.random.default_rng([POLICY['seed'],parent,trial]),.02,strain=.003)


def specification(root):
    spec=load(root/'retention-plan.json');source=Path(spec['source']);work=load(root/'physics.json');done=load(root/'physics.complete.json')
    if spec['policy']!=POLICY or spec['implementation_sha256']!=sha(__file__) or spec['helper_sha256']!=helpers():raise ValueError('Retention implementation changed.')
    if any(sha(source/f)!=h for f,h in spec['source_hashes'].items()):raise ValueError('Source artifacts changed.')
    plan,native,data,report,cases=source_evidence(source)
    if cases!=spec['cases'] or not done['completed'] or done['physics_sha256']!=sha(root/'physics.json'):raise ValueError('Physics completion/cohort changed.')
    validate_trials(cases,work['records']);expected={(c['parent_id'],s) for c in cases for s in STAGES}
    if len(work['controls'])!=len(expected) or {(r['parent_id'],r['stage']) for r in work['controls']}!=expected or not all(r['passed'] for r in work['controls']):raise ValueError('Physical source controls missing/failed.')
    return spec,plan,native,data,report,work


def physics(source,root):
    plan,native,data,report,cases=source_evidence(source);root.mkdir(parents=True,exist_ok=False)
    save(root/'retention-plan.json',dict(policy=POLICY,source=str(source),cases=cases,source_hashes={f:sha(source/f) for f in SOURCE_FILES},
        implementation_sha256=sha(__file__),helper_sha256=helpers(),source_assessment=report['assessment']))
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
        smoacs=HERE.parents[2]/'smoacs';sys.path[:0]=[str(smoacs),str(smoacs/'postprocess')]
        from alignn_checkpoint import TrustedCheckpointTorch
        from mattersim_support import model_max_z,validate_atomic_numbers
        expected=load(smoacs/'postprocess/sources.json')['mattersim']['checkpoint_sha256'];checkpoint=Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve()
        if sha(checkpoint)!=expected:raise ValueError('Potential checkpoint changed.')
        old=potential.torch;potential.torch=TrustedCheckpointTorch(torch,[expected]);torch.set_num_threads(2)
        try:
            with redirect_stdout(sys.stderr):calc=MatterSimCalculator(load_path=str(checkpoint),device='cpu',load_training_state=False)
        finally:potential.torch=old
        if any(p.is_cuda for p in calc.potential.model.parameters()):raise RuntimeError('CPU retention accessed GPU.')
        calc.potential.model.eval();maximum=model_max_z(calc.potential.model);calculate=calc.calculate;templates={}
        def counted(*a,**kw):calls[0]+=1;return calculate(*a,**kw)
        calc.calculate=counted
        for case in cases:
            parent=case['parent_id']
            for stage,q in endpoints(data,native,case).items():
                if sha(q['pose'])!=q['B']['relaxed_sha256']:raise ValueError('Source pose changed.')
                a=read(q['pose'],format='vasp');a.set_constraint(None);validate_atomic_numbers(a.numbers,maximum);a.calc=calc
                e=float(a.get_potential_energy());force=float(np.linalg.norm(a.get_forces(apply_constraint=False),axis=1).max())
                controls.append(dict(parent_id=parent,stage=stage,source_sha256=q['B']['relaxed_sha256'],**validate_control(e,force,q['B'],expected)));templates[parent,stage]=a.copy()
        for case in cases:
            parent=case['parent_id'];original=native['baseline'][case['index']]['B'];energy=original['final_energy_eV']/original['num_atoms'];volume=original['volume_A3'];index=int(Path(parent).stem.split('-')[1])
            for stage in STAGES:
                reference=templates[parent,stage]
                for trial in range(4):
                    folder=root/Path(parent).stem/stage/f'trial-{trial}';folder.mkdir(parents=True,exist_ok=False);path=folder/'input.vasp';write(path,reference,format='vasp',direct=True,sort=False)
                    row=dict(candidate_id=f'{Path(parent).stem}-{stage}-t{trial}.vasp',parent_id=parent,stage=stage,trial=trial,status='failed',input=str(path),checkpoint_sha256=expected,
                        original_energy_eV_atom=energy,original_volume_A3=volume,retention_plan_sha256=sha(root/'retention-plan.json'),
                        source_pose_sha256=endpoints(data,native,case)[stage]['B']['relaxed_sha256'],maxstep=.05,step_limit=1600,preserve_symmetry=False,fmax_target=.001,
                        perturbation_applied=trial!=0,perturbation_seed=[POLICY['seed'],index,trial] if trial else None)
                    nfe=calls[0];started=time.perf_counter()
                    try:
                        a=perturb(reference,index,trial);write(path,a,format='vasp',direct=True,sort=False);a.calc=calc;filtered=ExpCellFilter(a);opt=BFGS(filtered,logfile=str(folder/'optimizer.log'),maxstep=.05);converged=bool(opt.run(fmax=.001,steps=1600))
                        raw=float(np.linalg.norm(a.get_forces(apply_constraint=False),axis=1).max());force=float(np.linalg.norm(filtered.get_forces(),axis=1).max());e=float(a.get_potential_energy())
                        if not np.array_equal(a.numbers,reference.numbers) or not np.isfinite([raw,force,e]).all():raise ValueError('Species/nonfinite relaxation.')
                        final=folder/'relaxed.vasp';write(final,a,format='vasp',direct=True,sort=False)
                        row.update(status='converged' if converged else 'unconverged',converged=converged,num_atoms=len(a),final_energy_eV=e,max_raw_atomic_force=raw,max_filter_force=force,
                            separated=not len(neighbor_list('d',a,.5-1e-9)),volume_A3=a.get_volume(),volume_ratio=a.get_volume()/reference.get_volume(),original_volume_ratio=a.get_volume()/volume,
                            relaxed_path=str(final),relaxed_sha256=sha(final),optimizer_steps=opt.get_number_of_steps())
                    except Exception as error:row.update(error=repr(error),traceback=traceback.format_exc())
                    row.update(input_sha256=sha(path),NFE=calls[0]-nfe,elapsed_seconds=time.perf_counter()-started,retention_eligible=eligible(row));records.append(row);save(folder/'result.json',row)
                    save(root/'physics.json',dict(records=records,controls=controls,NFE=calls[0],elapsed_seconds=time.perf_counter()-begun))
                    print(json.dumps(dict(parent=parent,stage=stage,trial=trial,status=row['status'],eligible=row['retention_eligible'])),flush=True)
    else:save(root/'physics.json',dict(records=[],controls=[],NFE=0,elapsed_seconds=0.,reason='No material full-gradient gains beyond zero-CF pool.'))
    validate_trials(cases,records);save(root/'physics.complete.json',dict(completed=True,cases=len(cases),trials=len(records),physics_sha256=sha(root/'physics.json')))
    if not cases:report_empty(root)


def report_empty(root):
    report(root,load(root/'retention-plan.json'),load(root/'physics.json'))


def export_trials(root,plan,records,scorer):
    folder=root/'evaluation-retention';folder.mkdir(exist_ok=False);(folder/'inventory-249').mkdir();cases={c['parent_id']:c for c in plan['cases']};items=[];rows=[]
    for r in records:
        case=cases[r['parent_id']];path=Path(r['input']);identifier=r['candidate_id']
        if sha(path)!=r['input_sha256']:raise ValueError('Trial input changed.')
        gap,ef=scorer.predict(path);B=dict(r);destination=folder/'relaxation'/Path(identifier).stem;destination.mkdir(parents=True)
        if r.get('relaxed_path'):
            if sha(r['relaxed_path'])!=r['relaxed_sha256']:raise ValueError('Trial final pose changed.')
            shutil.copy2(r['relaxed_path'],destination/'relaxed.vasp');B['relaxed_path']=str((destination/'relaxed.vasp').resolve())
        save(destination/'result.json',B)
        items.append(dict(case['inventory'],candidate_id=identifier,source=str(path),sha256=r['input_sha256'],original_gap=gap,original_eform=ef,original_valid=str(case['neutral'])))
        rows.append(dict(original_fnames=identifier,bandgap_onehot=gap,ef_onehot=ef,is_neutral_cmn=str(case['neutral']),**{'minbond_less_than_0.5':'True'},valid_structure=str(case['neutral'])))
    with (folder/'result.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(folder/'inventory-249/inventory.json',dict(role='Seven-stage correlated annealed-gradient retention trials',denominator=len(items),candidates=items,source_csv_sha256=sha(folder/'result.csv')))


def prepare(root,spec,plan,native,data,work):
    if not spec['cases']:return
    scorer=GradientScorer();controls=[]
    for case in spec['cases']:
        for stage,q in endpoints(data,native,case).items():
            if sha(q['pose'])!=q['B']['relaxed_sha256']:raise ValueError('Source pose changed before CF control.')
            gap,ef=scorer.predict(Path(q['pose']));error=max(abs(gap-q['gap']),abs(ef-q['eform']))
            if error>1e-5:raise ValueError('Source CF control failed.')
            controls.append(dict(parent_id=case['parent_id'],stage=stage,error=error))
    export_trials(root,plan,work['records'],scorer);save(root/'CF-controls.json',dict(passed=True,cases=controls,CF=scorer.metadata))


def report(root,spec,work):
    validate_trials(spec['cases'],work['records'])
    if spec['cases']:
        cf=load(root/'CF-controls.json');expected={(c['parent_id'],s) for c in spec['cases'] for s in STAGES}
        if not cf['passed'] or len(cf['cases'])!=len(expected) or {(r['parent_id'],r['stage']) for r in cf['cases']}!=expected or any(r['error']>1e-5 for r in cf['cases']):raise ValueError('Source CF controls incomplete.')
        folder=root/'evaluation-retention/cross-prediction';meta=load(folder/'metadata.json')
        if not meta['crystalformer_control']['control_passed'] or len(meta['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in meta['controls']):raise ValueError('Official evaluator controls failed.')
        rows=[r for r in load(folder/'predictions.json') if r['stage']=='after'];pred={r['candidate_id']:r for r in rows}
        if len(pred)!=len(rows):raise ValueError('Duplicate predictions.')
        for r in work['records']:
            if sha(r['input'])!=r['input_sha256']:raise ValueError('Actual trial input changed.')
            if r.get('relaxed_path') and sha(r['relaxed_path'])!=r['relaxed_sha256']:raise ValueError('Trial final pose changed.')
            if r['status']=='converged' and pred[r['candidate_id']].get('geometry_sha256')!=r['relaxed_sha256']:raise ValueError('Trial prediction pose changed.')
        cases=summarize(spec['cases'],work['records'],pred)
    else:cases=[]
    counts={}
    for model in ('crystalformer','alignn'):
        robust=[c for c in cases if all(c['models'][model][s]['all_four_gain_ge_0p1'] for s in ('original','seed','zero'))]
        counts[model]=dict(original_parent_denominator=16,material_cases=len(cases),retained_vs_original_seed_and_zero=len(robust),
            retained_and_beats_directions=sum(all(c['models'][model][s]['all_four_gain_gt_1meV'] for s in ('reverse','random')) for c in robust),
            retained_and_beats_directions_and_fixed=sum(all(c['models'][model][s]['all_four_gain_gt_1meV'] for s in ('reverse','random','fixed')) for c in robust))
    save(root/'report.json',dict(policy=POLICY,source_assessment=spec['source_assessment'],counts=counts,cases=cases,NFE=work['NFE'],goal_completion_proven=False,
        note='All material cases retained, including unknown/failed trials. Four trials per parent are correlated. Selected-case retention never rescues failed population directionality.'))
    print(json.dumps(counts),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--mode',choices=('physics','prepare','report'),required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm; prepare requires GPU.')
    root=a.run.resolve()
    if a.mode=='physics':
        if a.source is None:p.error('physics requires --source')
        physics(a.source.resolve(),root)
    else:
        spec,plan,native,data,_,work=specification(root)
        if a.mode=='prepare':prepare(root,spec,plan,native,data,work)
        else:report(root,spec,work)


if __name__=='__main__':main()
