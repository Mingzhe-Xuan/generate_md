"""Continue the cached first plain-CF round with two fully quenched feedback rounds."""
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
from geometry import Geometry
from directions import normalize,matched_random
from settings import POLICY,FAMILIES,ARMS,SCALES
from probe import load,save,sha,code_hashes,common_seed
from runtime import Mechanics
from cf_gradient import GradientScorer
from search import point,export
from policy import best,physical

FEEDBACK_POLICY=dict(rounds=3,additional_rounds=2,arms=list(ARMS),scales=list(SCALES),
    gradient='fresh_float32_CF_gap_squared_loss_at_each_relaxed_incumbent_plus_shared_seed',
    first_round='exact_cached_plain_family_from_complete_stage_one',
    seed_schedule='source_search_seed +1000*(round_number-1), same across arms',
    selection='best_CF_gap_MAE_among_incumbent_seed_and_four_full_B_endpoints_with_original_physical_guards',
    role='exploratory iterative method development, not independent confirmation')


def validate_source(plan,data,proof):
    parents=plan['parent_ids']
    if len(parents)!=16 or len(set(parents))!=16 or [c['parent_id'] for c in plan['cases']]!=parents or plan['policy']!=POLICY:
        raise ValueError('Frozen16 cohort/policy changed.')
    if not proof['passed'] or proof['code_sha256']!=code_hashes() or [c['parent_id'] for c in proof['operators']]!=parents or not all(c.get('passed') is True for c in proof['operators']):
        raise ValueError('Missing/stale native derivative proof.')
    expected={(p,f,a,k) for p in parents for f in FAMILIES for a in ARMS for k in SCALES}
    actual={(r['parent_id'],r['family'],r['arm'],r['scale']) for r in data['attempts']}
    if data['completed_parents']!=16 or len(data['attempts'])!=len(expected) or actual!=expected:
        raise ValueError('Incomplete/missing/duplicate source attempts.')
    if len(data['baseline'])!=16 or [r['parent_id'] for r in data['seeds']]!=parents:
        raise ValueError('Baseline/seed coverage changed.')
    selected={a:copy.deepcopy(data['selected']['plain-'+a]) for a in ARMS}
    if any(len(v)!=16 for v in selected.values()):raise ValueError('Selected coverage changed.')
    for i,parent in enumerate(parents):
        base=data['baseline'][i];seed=data['seeds'][i]['point'];energy=base['B']['final_energy_eV']/base['B']['num_atoms']
        for arm in ARMS:
            pool=[base,seed]+[r['point'] for r in data['attempts'] if r['parent_id']==parent and r['family']=='plain' and r['arm']==arm and 'point' in r]
            expected_point=best(pool,energy);actual_point=selected[arm][i]
            if expected_point!=actual_point:raise ValueError('Cached first-round selection changed.')
    return selected


def direction(geom,gradient,arm,seed,parent_index,round_number):
    gradient=geom.project(gradient)
    if np.linalg.norm(gradient)<1e-10:return np.zeros_like(gradient)
    descent=normalize(-gradient)
    if arm=='descent':return descent
    if arm=='reverse':return -descent
    if arm=='random':return matched_random(descent,np.random.default_rng([seed,parent_index,555,round_number]),geom.project)
    raise ValueError('Unknown direction arm.')


def proposal_identifier(parent,arm,round_number,scale_index=None):
    return f'{Path(parent).stem}-{arm}-r{round_number}-'+('seed' if scale_index is None else f's{scale_index}')


def validate_feedback_results(plan,data):
    parents=plan['parent_ids']
    expected={(p,r,a,k) for p in parents for r in (2,3) for a in ARMS for k in SCALES}
    actual={(r['parent_id'],r['round'],r['arm'],r['scale']) for r in data['attempts']}
    if len(data['attempts'])!=384 or actual!=expected:raise ValueError('Missing/duplicate feedback attempts.')
    seed_expected={(p,r,a) for p in parents for r in (2,3) for a in ARMS}
    if len(data['seeds'])!=96 or {(r['parent_id'],r['round'],r['arm']) for r in data['seeds']}!=seed_expected:
        raise ValueError('Missing/duplicate zero-step attempts.')
    groups={(r['parent_id'],r['round']):r for r in data['history']}
    if len(data['history'])!=32 or set(groups)!={(p,r) for p in parents for r in (2,3)}:raise ValueError('Missing/duplicate round history.')
    for i,case in enumerate(plan['cases']):
        parent=case['parent_id'];volume=data['baseline'][i]['B']['volume_A3'];energy=case['B']['final_energy_eV']/case['B']['num_atoms']
        for arm in ARMS:
            current=data['initial'][arm][i]
            for round_number in (2,3):
                group=groups[parent,round_number]
                if group['starts'][arm]!=current:raise ValueError('Adaptive incumbent history changed.')
                candidates=[r for r in data['attempts']+data['seeds'] if r['parent_id']==parent and r['round']==round_number and r['arm']==arm]
                for r in candidates:
                    if r['original_volume_A3']!=volume:raise ValueError('Original volume reference changed.')
                    if 'point' in r and abs(r['point']['B']['volume_ratio']-r['point']['B']['volume_A3']/volume)>1e-6:
                        raise ValueError('Rolling volume ratio replaced original reference.')
                current=best([current]+[r['point'] for r in candidates if 'point' in r],energy)
                if group['selected'][arm]!=current:raise ValueError('Adaptive selection violated physical/CF rule.')
            if data['selected'][arm][i]!=current:raise ValueError('Final feedback incumbent changed.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    from pymatgen.core import Structure
    from ase import Atoms
    from ase.io import write
    source=a.source.resolve();root=a.run.resolve();plan=load(source/'plan.json');data=load(source/'direction-results.json');proof=load(source/'native-probe.json')
    if not load(source/'direction.complete.json')['completed'] or not (source/'report.json').is_file():raise ValueError('Source final evaluation missing.')
    selected=validate_source(plan,data,proof);initial=copy.deepcopy(selected)
    root.mkdir(parents=True,exist_ok=False);rpc=root/'feedback-rpc';rpc.mkdir();scorer=GradientScorer();client=Mechanics(rpc)
    specification=dict(policy=FEEDBACK_POLICY,source_plan=plan,source_root=str(source),
        source_hashes={n:sha(source/n) for n in ('plan.json','direction-results.json','native-probe.json','direction.complete.json','report.json')},
        code_sha256=code_hashes(),feedback_sha256=sha(__file__),report_sha256=sha(Path(__file__).with_name('feedback_report.py')))
    save(root/'feedback-plan.json',specification);records=[];seeds=[];history=[];controls=[]
    try:
        for i,case in enumerate(plan['cases']):
            parent=case['parent_id'];base=data['baseline'][i];original=Structure.from_file(case['B']['relaxed_path'])
            checked=set()
            for pnt in [base]+[initial[arm][i] for arm in ARMS]:
                if pnt['pose'] in checked:continue
                checked.add(pnt['pose']);s=Structure.from_file(pnt['pose'])
                if sha(pnt['pose'])!=pnt['B']['relaxed_sha256'] or tuple(s.atomic_numbers)!=tuple(original.atomic_numbers):raise ValueError('Cached species/source SHA changed.')
                gap,ef=scorer.predict(Path(pnt['pose']));error=max(abs(gap-pnt['gap']),abs(ef-pnt['eform']))
                if error>1e-5:raise ValueError('Cached first round CF control failed.')
                controls.append(dict(parent_id=parent,pose=pnt['pose'],CF_error=error))
        save(root/'feedback-native-controls.json',dict(passed=True,cases=controls,CF=scorer.metadata,source_native_sha256=sha(source/'native-probe.json')))
        for round_number in (2,3):
            for i,case in enumerate(plan['cases']):
                parent=case['parent_id'];index=int(Path(parent).stem.split('-')[1]);base=data['baseline'][i]
                original_volume=base['B']['volume_A3'];energy=base['B']['final_energy_eV']/base['B']['num_atoms']
                pools={arm:[selected[arm][i]] for arm in ARMS};starts={arm:copy.deepcopy(selected[arm][i]) for arm in ARMS};tasks=[];pending=[]
                folder=root/'inputs'/Path(parent).stem/f'round-{round_number}';folder.mkdir(parents=True,exist_ok=False)
                for arm in ARMS:
                    current=selected[arm][i];s=Structure.from_file(current['pose']);geom=Geometry(s.atomic_numbers,s.lattice.matrix,s.frac_coords)
                    qseed,draw=common_seed(geom,index,plan['search_seed']+1000*(round_number-1));score=scorer.evaluate(geom,qseed,True)
                    d=direction(geom,score['gradient'],arm,plan['search_seed'],index,round_number)
                    for k,scale in [(None,0.)]+list(enumerate(SCALES)):
                        identifier=proposal_identifier(parent,arm,round_number,k);q=qseed+scale*d
                        record=dict(parent_id=parent,arm=arm,round=round_number,scale=scale,id=identifier,role='seed' if k is None else 'proposal',
                            incumbent_id=current['id'],incumbent_sha256=current['B']['relaxed_sha256'],original_volume_A3=original_volume,
                            seed_draw=draw,gradient_norm=float(np.linalg.norm(score['gradient'])),actual_direction_norm=float(np.linalg.norm(d)),eligible=False)
                        if k is not None and (np.linalg.norm(d)<1e-10 or not geom.feasible(q)):
                            record.update(status='stationary_or_geometric_rejection');records.append(record);continue
                        cell,frac=geom.pose(q);path=folder/(identifier+'.vasp');write(path,Atoms(numbers=geom.numbers,cell=cell,scaled_positions=frac,pbc=True),format='vasp',direct=True,sort=False)
                        record.update(input=str(path),input_sha256=sha(path));pending.append(record)
                        tasks.append(dict(id=identifier,action='relax',source=str(path),output=str(root/'B'/Path(parent).stem/identifier)))
                results=client.request(tasks)
                for record,B in zip(pending,results):
                    record.update(status=B['status'],B=B)
                    if B['status']=='converged':
                        try:pnt=point(record['id'],record['input'],B,scorer,original_volume)
                        except ValueError as error:record.update(status='CF_prediction_failed',error=repr(error))
                        else:
                            record.update(point=pnt,eligible=bool(physical(pnt['B'],energy) and pnt['eform']<-.5));pools[record['arm']].append(pnt)
                    (seeds if record['role']=='seed' else records).append(record)
                for arm in ARMS:selected[arm][i]=best(pools[arm],energy)
                history.append(dict(parent_id=parent,round=round_number,starts=starts,selected={arm:copy.deepcopy(selected[arm][i]) for arm in ARMS}))
                save(root/'feedback-results.json',dict(baseline=data['baseline'],initial=initial,selected=selected,history=history,attempts=records,seeds=seeds,NFE=client.NFE,mechanics_seconds=client.seconds,CF=scorer.metadata))
                print(json.dumps(dict(parent_id=parent,round=round_number,groups=len(history),attempts=len(records),NFE=client.NFE,
                    errors={arm:abs(selected[arm][i]['gap']-2.5) for arm in ARMS})),flush=True)
        if len(records)!=384 or len(seeds)!=96 or len(history)!=32:raise ValueError('Feedback attempt budget changed.')
        validate_feedback_results(plan,load(root/'feedback-results.json'))
        for arm in ARMS:export(root,arm,selected[arm],plan,scorer)
        save(root/'feedback.complete.json',dict(completed=True,fresh_ray_attempts=384,cached_first_ray_attempts=192,seed_B_attempts=96,NFE=client.NFE,CF=scorer.metadata))
    finally:client.close()


if __name__=='__main__':main()
