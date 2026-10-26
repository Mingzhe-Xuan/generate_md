"""Native-gated48-parent CF rays with frozen conservative and ordinary selections."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from validation_policy import VALIDATION_POLICY,retained_choice
from settings import POLICY,ARMS,SCALES
from geometry import Geometry
from directions import normalize,matched_random
from probe import load,save,sha,code_hashes,common_seed,close
from numerics import promote,score as precision_score
from cf_gradient import GradientScorer
from runtime import Mechanics
from search import point
from policy import best,physical
from evaluation_export import export_evaluation
from baseline_amendment import load_amendment,fixed_pose_control

PLAN_SHA='0838a373d8da0f530fa39240474b42f53aae8a462530c362b3b1b7134d47b47d'
OUTPUTS=tuple(f'{policy}-{arm}' for policy in ('retained','ordinary') for arm in ARMS)+('baseline',)
EXTRA_BASELINE='raw-replay-baseline'


def core_hashes():
    result=code_hashes()
    for name in ('validation_run.py','validation_policy.py','evaluation_export.py','baseline_amendment.py'):result[name]=sha(Path(__file__).with_name(name))
    base=Path(__file__).resolve().parents[1]/'basin_search'
    for name in ('scorer.py','search.py','policy.py'):result['shared_'+name]=sha(base/name)
    return result


def validate_manifest(plan,digest):
    parents=plan['parent_ids']
    if digest!=PLAN_SHA or plan['policy']!=VALIDATION_POLICY or plan['search_seed']!=20261027:raise ValueError('Preregistered plan changed.')
    if len(parents)!=48 or len(set(parents))!=48 or set(parents)&set(plan['pilot_ids']) or [c['parent_id'] for c in plan['cases']]!=parents:raise ValueError('Heldout48 cohort changed.')
    if plan['native_math_sha256']!=code_hashes():raise ValueError('Native mathematical source changed.')


def six_checks_pass(records):
    expected={(b,e) for b in ('atomic','strain','both') for e in POLICY['CF_FD_steps_A']}
    return bool(len(records)==6 and {(r['block'],r['epsilon']) for r in records}==expected and all(r['passed'] for r in records))


def native_gate(root,plan,scorer,double_scorer,client):
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    strict=StructureMatcher(ltol=.02,stol=.05,angle_tol=1,primitive_cell=False,scale=False,attempt_supercell=False)
    amendment,diagnostic,replays=load_amendment(root,plan)
    records=[];baselines=[];fresh_baselines=[]
    for i,(case,replay) in enumerate(zip(plan['cases'],replays)):
        parent=case['parent_id'];reference=Structure.from_file(case['B']['relaxed_path']);geom=Geometry(reference.atomic_numbers,reference.lattice.matrix,reference.frac_coords)
        if sha(case['raw_path'])!=case['B']['input_sha256'] or sha(case['B']['relaxed_path'])!=case['B']['relaxed_sha256']:raise ValueError('Heldout source provenance changed.')
        base=point(parent,case['raw_path'],case['B'],scorer);zero=scorer.evaluate(geom,np.zeros(geom.size));expected=case['predictions']['after'];baselines.append(base)
        cf_error=max(abs(base['gap']-expected['crystalformer_gap']),abs(base['eform']-expected['crystalformer_eform']),abs(zero['gap']-base['gap']),abs(zero['eform']-base['eform']))
        energy_error=abs(replay['final_energy_eV']-case['B']['final_energy_eV'])/geom.n if replay['status']=='converged' else None
        fit=replay['status']=='converged' and bool(strict.fit(reference,Structure.from_file(replay['relaxed_path'])))
        potential_control=fixed_pose_control(case,client.points(geom,np.zeros((1,geom.size)))[0])
        fresh_structure=Structure.from_file(replay['relaxed_path'])
        if tuple(fresh_structure.atomic_numbers)!=tuple(reference.atomic_numbers):raise ValueError('Fresh raw-B ordered species changed.')
        fresh_baselines.append(point(parent,case['raw_path'],replay,scorer,base['B']['volume_A3']))
        global_index=int(Path(parent).stem.split('-')[1]);q,draw=common_seed(geom,global_index,plan['search_seed']);score=scorer.evaluate(geom,q,True);frozen=scorer.evaluate(geom,q)
        g=geom.project(score['gradient']);double=precision_score(double_scorer,geom,q,scorer.torch.float64,True);native=precision_score(double_scorer,geom,q,scorer.torch.float64)
        adapter=max(abs(score[k]-frozen[k]) for k in ('gap','eform'));forward_error=max(abs(score[k]-double[k]) for k in ('gap','eform'))
        gradient_error=float(np.linalg.norm(g-geom.project(double['gradient']))/max(np.linalg.norm(double['gradient']),1e-10));double_adapter=max(abs(double[k]-native[k]) for k in ('gap','eform'))
        checks=[]
        if i<4:
            rng=np.random.default_rng([20261029,global_index,777])
            for block in ('atomic','strain','both'):
                v=geom.project(rng.normal(size=geom.size))
                if block=='atomic':v[-6:]=0
                if block=='strain':v[:-6]=0
                v=normalize(v)
                for eps in POLICY['CF_FD_steps_A']:
                    plus=precision_score(double_scorer,geom,q+eps*v,scorer.torch.float64);minus=precision_score(double_scorer,geom,q-eps*v,scorer.torch.float64)
                    fd=(plus['loss']-minus['loss'])/(2*eps);ad=float(g@v)
                    checks.append(dict(block=block,epsilon=eps,FD=fd,predicted=ad,passed=close(fd,ad,.001,.02)))
        descent=np.zeros_like(g) if np.linalg.norm(g)<1e-10 else normalize(-g)
        random=matched_random(descent,np.random.default_rng([plan['search_seed'],global_index,0,555]),geom.project)
        passed=bool(cf_error<=1e-5 and potential_control['passed'] and adapter<=1e-5 and forward_error<=1e-5 and gradient_error<=.001 and double_adapter<=1e-10 and (six_checks_pass(checks) if i<4 else True))
        entry=dict(parent_id=parent,passed=passed,baseline_CF_error=cf_error,baseline_energy_error_eV_atom=energy_error,baseline_match=fit,
            fixed_pose_control=potential_control,raw_path_reproduced=bool(energy_error is not None and energy_error<=1e-4 and fit),
            adapter_forward_error=adapter,precision_forward_error=forward_error,precision_gradient_relative_error=gradient_error,double_adapter_error=double_adapter,
            derivative_checks=checks,qseed=q.tolist(),descent=descent.tolist(),random=random.tolist(),gradient_norm=float(np.linalg.norm(g)),seed_draw=draw,source_sha256=case['B']['relaxed_sha256'])
        records.append(entry);save(root/'validation-native-progress.json',dict(cases=records,CF=scorer.metadata,NFE=client.NFE))
        print(json.dumps(dict(stage='native',parent_id=parent,completed=i+1,passed=passed,precision_gradient_error=gradient_error)),flush=True)
        if not passed:raise ValueError('Heldout native gate failed before any scientific proposal; inspect validation-native-progress.json.')
    proof=dict(passed=True,cases=records,baseline=baselines,plan_sha256=sha(root/'validation-plan.json'),code_sha256=core_hashes(),
        CF=scorer.metadata,CF_double_diagnostic=double_scorer.metadata,mechanics=client.ready,
        baseline_amendment_sha256=sha(root/'baseline-amendment.json'),raw_replay_baseline=fresh_baselines,
        raw_path_mismatches=diagnostic['raw_path_mismatches'],NFE_cached_replay=amendment['cached_raw_B_NFE'],NFE_current_execution=client.NFE,
        NFE=client.NFE+amendment['cached_raw_B_NFE'],mechanics_seconds=client.seconds)
    save(root/'validation-native.json',proof);return proof


def validate_results(plan,data):
    parents=plan['parent_ids']
    if len(parents)!=48 or data['completed_parents']!=48 or len(data['baseline'])!=48 or [s['parent_id'] for s in data['seeds']]!=parents:raise ValueError('Incomplete heldout48 denominator.')
    expected={(p,a,k) for p in parents for a in ARMS for k in SCALES}
    if len(data['attempts'])!=576 or {(r['parent_id'],r['arm'],r['scale']) for r in data['attempts']}!=expected:raise ValueError('Missing/duplicate heldout ray attempts.')
    if set(data['selected'])!=set(OUTPUTS) or any(len(v)!=48 for v in data['selected'].values()):raise ValueError('Selected output coverage changed.')
    if 'baseline_amendment_sha256' in data:
        fresh=data['raw_replay_baseline']
        if len(fresh)!=48 or [q['id'] for q in fresh]!=parents:raise ValueError('Fresh raw-B coverage changed.')
        if data['NFE_total']!=data['NFE_native']+data['NFE_search'] or data['NFE_cached_replay']>data['NFE_native']:raise ValueError('Cached/native/search cost ledger changed.')
    for i,parent in enumerate(parents):
        base=data['baseline'][i];seed=data['seeds'][i];energy=base['B']['final_energy_eV']/base['B']['num_atoms'];control=best([base,seed['point']],energy)
        original=plan['cases'][i]
        if base['B']['final_energy_eV']!=original['B']['final_energy_eV'] or base['B']['num_atoms']!=original['B']['num_atoms'] or base['B']['relaxed_sha256']!=original['B']['relaxed_sha256']:
            raise ValueError('Immutable original energy/geometry reference changed.')
        if base['id']!=parent or seed['control']!=control or data['selected']['baseline'][i]!=base:raise ValueError('Baseline/feasible seed comparator changed.')
        if 'baseline_amendment_sha256' in data:
            q=data['raw_replay_baseline'][i]
            if q['B']['input_sha256']!=original['B']['input_sha256'] or q['B']['checkpoint_sha256']!=original['B']['checkpoint_sha256'] or abs(q['B']['volume_ratio']-q['B']['volume_A3']/base['B']['volume_A3'])>1e-6:raise ValueError('Fresh raw-B provenance/reference changed.')
        for arm in ARMS:
            proposals=[r['point'] for r in data['attempts'] if r['parent_id']==parent and r['arm']==arm and 'point' in r]
            for q in proposals:
                if abs(q['B']['volume_ratio']-q['B']['volume_A3']/base['B']['volume_A3'])>1e-6:raise ValueError('Original volume reference changed.')
            if data['selected']['ordinary-'+arm][i]!=best([base,seed['point']]+proposals,energy):raise ValueError('Ordinary selection changed.')
            if data['selected']['retained-'+arm][i]!=retained_choice(base,control,proposals,energy):raise ValueError('Retained selection changed.')


def search(root,plan,proof,scorer,client):
    from pymatgen.core import Structure
    from ase import Atoms
    from ase.io import write
    baseline=proof['baseline'];seeds=[];attempts=[];selected={k:[] for k in OUTPUTS};start_NFE=client.NFE
    for i,(case,operator) in enumerate(zip(plan['cases'],proof['cases'])):
        parent=case['parent_id'];stem=Path(parent).stem;base=baseline[i];volume=base['B']['volume_A3'];energy=base['B']['final_energy_eV']/base['B']['num_atoms']
        if sha(base['pose'])!=operator['source_sha256']:raise ValueError('Original baseline changed after native gate.')
        s=Structure.from_file(base['pose']);geom=Geometry(s.atomic_numbers,s.lattice.matrix,s.frac_coords);qseed=np.array(operator['qseed']);folder=root/'inputs'/stem;folder.mkdir(parents=True,exist_ok=False)
        def write_q(q,path):
            cell,frac=geom.pose(q);write(path,Atoms(numbers=geom.numbers,cell=cell,scaled_positions=frac,pbc=True),format='vasp',direct=True,sort=False)
        seed_path=folder/'seed.vasp';write_q(qseed,seed_path)
        B=client.request([dict(id=stem+'-seed',action='relax',source=str(seed_path),output=str(root/'B'/stem/'seed'))])[0]
        seedpoint=point(stem+'-seed',seed_path,B,scorer,volume) if B['status']=='converged' else base;seed_control=best([base,seedpoint],energy)
        seeds.append(dict(parent_id=parent,B=B,point=seedpoint,control=seed_control));pending=[];tasks=[]
        for arm in ARMS:
            d=np.array(operator['descent'] if arm=='descent' else operator['random'] if arm=='random' else -np.array(operator['descent']))
            for k,scale in enumerate(SCALES):
                identifier=f'{stem}-{arm}-s{k}';q=qseed+scale*d;record=dict(parent_id=parent,arm=arm,scale=scale,id=identifier,eligible=False,direction_norm=float(np.linalg.norm(d)))
                if np.linalg.norm(d)<1e-10 or not geom.feasible(q):
                    record['status']='stationary_or_geometric_rejection';attempts.append(record);continue
                path=folder/(identifier+'.vasp');write_q(q,path);record.update(input=str(path),input_sha256=sha(path));pending.append(record)
                tasks.append(dict(id=identifier,action='relax',source=str(path),output=str(root/'B'/stem/identifier)))
        results=client.request(tasks)
        for record,B in zip(pending,results):
            record.update(status=B['status'],B=B)
            if B['status']=='converged':
                try:q=point(record['id'],record['input'],B,scorer,volume)
                except ValueError as error:record.update(status='CF_prediction_failed',error=repr(error))
                else:record.update(point=q,eligible=bool(physical(q['B'],energy) and q['eform']<-.5))
            attempts.append(record)
        for arm in ARMS:
            proposals=[r['point'] for r in attempts if r['parent_id']==parent and r['arm']==arm and 'point' in r]
            selected['ordinary-'+arm].append(best([base,seedpoint]+proposals,energy));selected['retained-'+arm].append(retained_choice(base,seed_control,proposals,energy))
        selected['baseline'].append(base)
        save(root/'validation-results.json',dict(baseline=baseline,seeds=seeds,attempts=attempts,selected=selected,completed_parents=i+1,CF=scorer.metadata,
            baseline_amendment_sha256=proof['baseline_amendment_sha256'],raw_replay_baseline=proof['raw_replay_baseline'],NFE_cached_replay=proof['NFE_cached_replay'],
            NFE_total=client.NFE+proof['NFE_cached_replay'],NFE_native=proof['NFE'],NFE_search=client.NFE-start_NFE,mechanics_seconds=client.seconds))
        print(json.dumps(dict(stage='search',parent_id=parent,completed=i+1,attempts=len(attempts),NFE=client.NFE,
            errors={k:abs(v[-1]['gap']-2.5) for k,v in selected.items()})),flush=True)
    return load(root/'validation-results.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--export-only',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('All CF and physics must run on Slurm GPU.')
    root=a.run.resolve();plan=load(root/'validation-plan.json');validate_manifest(plan,sha(root/'validation-plan.json'))
    load_amendment(root,plan)
    if (root/'validation.complete.json').exists():
        done=load(root/'validation.complete.json')
        if done['data_sha256']!=sha(root/'validation-results.json') or load(root/'validation-native.json')['code_sha256']!=core_hashes():raise ValueError('Completed validation source/data changed.')
        validate_results(plan,load(root/'validation-results.json'));print('VALIDATION_SEARCH_ALREADY_COMPLETE',flush=True);return
    scorer=GradientScorer()
    if a.export_only:
        proof=load(root/'validation-native.json')
        if not proof['passed'] or proof['code_sha256']!=core_hashes():raise ValueError('Native proof missing/stale.')
        data=load(root/'validation-results.json')
    else:
        rpc=root/'validation-rpc';rpc.mkdir(exist_ok=False);client=Mechanics(rpc);double=GradientScorer();promote(double,scorer.torch.float64)
        try:proof=native_gate(root,plan,scorer,double,client);data=search(root,plan,proof,scorer,client)
        finally:client.close()
    validate_results(plan,data)
    for method in OUTPUTS:export_evaluation(root,method,data['selected'][method],plan,scorer)
    export_evaluation(root,EXTRA_BASELINE,data['raw_replay_baseline'],plan,scorer)
    save(root/'validation.complete.json',dict(completed=True,parents=48,attempts=576,data_sha256=sha(root/'validation-results.json'),plan_sha256=sha(root/'validation-plan.json'),code_sha256=core_hashes()))


if __name__=='__main__':main()
