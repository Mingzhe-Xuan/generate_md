"""Scope-aware annealed-bias integration and full development16 experiment."""
import argparse
import copy
import os
from pathlib import Path
import traceback
import numpy as np
from anneal_policy import POLICY,PARENTS,ARMS,PEAKS,PROBE_STAGES,weights
from anneal_math import bounded_project
from anneal_fire import minimize
from anneal_objective import Provider
from anneal_native import save,NATIVE_POLICY
from anneal_shared import (Geometry,GradientScorer,Mechanics,promote,precision_score,load,sha,point,best,validate_control,
    implementation_hashes)
from evaluation_export import export_evaluation

OUTPUTS=ARMS+('seed','baseline')+tuple('full_stage'+str(s) for s in PROBE_STAGES)


def core_hashes():
    return implementation_hashes()|{'annealed/anneal_experiment.py':sha(__file__)}


def seed_components(case):
    physical=copy.deepcopy(case['physics_seed']);physical['gradient_E_total']=np.asarray(physical['gradient_E_total'],dtype=float)
    cf=copy.deepcopy(case['CF_seed']);cf['gradient']=np.asarray(cf['gradient'],dtype=float)
    return physical,cf


def specifications(parents):
    rows=[]
    for p in parents:
        stem=Path(p).stem;rows.append(dict(id=stem+'-seed',parent_id=p,kind='seed'))
        for arm in ARMS:
            for k,peak in enumerate(PEAKS):
                for stage in PROBE_STAGES:
                    rows.append(dict(id=f'{stem}-{arm}-p{k}-s{stage}',parent_id=p,kind='fork',arm=arm,peak=peak,stage=stage))
    return rows


def choose(base,rows):
    energy=base['B']['final_energy_eV']/base['B']['num_atoms']
    seed=[r['point'] for r in rows if r['kind']=='seed' and 'point'in r]
    common=[base]+seed+[r['point'] for r in rows if r.get('arm')=='zero' and 'point'in r]
    result=dict(baseline=base,seed=best([base]+seed,energy))
    result.update({a:best(common+[r['point'] for r in rows if r.get('arm')==a and 'point'in r],energy) for a in ARMS})
    for stage in PROBE_STAGES:
        result['full_stage'+str(stage)]=best(common+[r['point'] for r in rows if r.get('arm')=='full' and r.get('stage')==stage and 'point'in r],energy)
    return result


def source_proof(source,native_root):
    n=load(native_root/'native.json')
    if not n['passed'] or n['method']!=POLICY or n['native_policy']!=NATIVE_POLICY or n['implementation_sha256']!=implementation_hashes():
        raise ValueError('Annealed native proof changed/failed.')
    if len(n['cases'])!=16 or [c['parent_id'] for c in n['cases']]!=list(PARENTS) or not all(c['passed'] for c in n['cases']):
        raise ValueError('Complete native16 required.')
    for name,h in n['source_hashes'].items():
        if sha(source/name)!=h:raise ValueError('Original source evidence changed.')
    for c in n['cases']:
        if sha(c['seed_input'])!=c['seed_sha256']:raise ValueError('Native seed pose changed.')
    return n


def trajectory(root,case,base,arm,peak,k,scorer,client):
    geom=Geometry(**case['geometry']);q=np.asarray(case['qseed'],dtype=float);p,cf=seed_components(case)
    provider=Provider(geom,scorer,client,base['B']['final_energy_eV'],q,cf['loss'],case['random_slope'],initial_physics=p,initial_cf=cf)
    stem=f'{Path(case["parent_id"]).stem}-{arm}-p{k}';folder=root/'steering'/stem;folder.mkdir(parents=True,exist_ok=False)
    result=dict(id=stem,parent_id=case['parent_id'],arm=arm,peak=peak,stages=[],probes={},status='running');nfe=client.NFE
    try:
        for stage,strength in enumerate(weights(arm,peak)):
            telemetry=[]
            def objective(x):
                state=provider.evaluate(x,strength,arm)
                telemetry.append({key:state[key] for key in ('raw_physical_fmax','CF_gap','CF_loss','physical_gradient_norm_per_atom','weighted_bias_gradient_norm','cosine_physical_vs_bias')})
                return state
            outcome=minimize(q,objective,lambda x:bounded_project(geom,x),geom.feasible)
            if len(outcome['trace'])!=len(telemetry):raise RuntimeError('FIRE telemetry/solver trace mismatch.')
            q=outcome['q'];result['stages'].append(dict(stage=stage,weight=strength,telemetry=telemetry,**outcome))
            if stage in PROBE_STAGES:result['probes'][str(stage)]=q.copy()
            result.update(NFE_steering=client.NFE-nfe,component_costs=provider.costs());save(folder/'trajectory.json',result)
        result['status']='completed'
    except ValueError as error:
        result.update(status='steering_failed',error=repr(error),traceback=traceback.format_exc())
    result.update(NFE_steering=client.NFE-nfe,component_costs=provider.costs());save(folder/'trajectory.json',result)
    return result


def validate(root,plan,native,data):
    from pymatgen.core import Structure
    if plan['policy']!=POLICY or plan['core_sha256']!=core_hashes() or plan['parent_ids']!=list(PARENTS[:plan['count']]):
        raise ValueError('Frozen experiment scope/core changed.')
    if native['plan_sha256']!=sha(root/'plan.json') or not native['passed'] or data['completed_parents']!=plan['count']:
        raise ValueError('Incomplete integration/search.')
    source_proof(Path(plan['source']),Path(plan['native_root']))
    if sha(Path(plan['native_root'])/'native.json')!=plan['native_source_sha256']:raise ValueError('Native source proof changed.')
    specs=specifications(plan['parent_ids']);expected={r['id']:r for r in specs};actual={r['id']:r for r in data['attempts']}
    if len(actual)!=len(specs) or len(data['attempts'])!=len(specs) or set(actual)!=set(expected):raise ValueError('Missing/duplicate query inventory.')
    cases={c['parent_id']:c for c in native['cases']};traces={t['id']:t for t in data['trajectories']}
    if len(traces)!=plan['count']*len(ARMS)*len(PEAKS):raise ValueError('Trajectory budget changed.')
    for t in traces.values():
        expected_weights=weights(t['arm'],t['peak'])
        for i,s in enumerate(t['stages']):
            if s['stage']!=i or s['weight']!=expected_weights[i] or s['updates']>200 or len(s['trace'])!=len(s['telemetry']):raise ValueError('Stage schedule/trace mismatch.')
            if s['converged']!=(s['status']=='converged') or (s['converged'] and s['final_gradient_inf']>.002):raise ValueError('False optimizer convergence.')
        if t['status']=='completed' and len(t['stages'])!=8:raise ValueError('Completed trajectory missing stage.')
    for id,row in actual.items():
        if any(row[k]!=v for k,v in expected[id].items()):raise ValueError('Query specification changed.')
        case=cases[row['parent_id']];geom=Geometry(**case['geometry'])
        if 'q' not in row:
            if row['kind']!='fork' or row['status']!='steering_failed':raise ValueError('Unknown missing candidate.')
            k=PEAKS.index(row['peak']);t=traces[f'{Path(row["parent_id"]).stem}-{row["arm"]}-p{k}']
            if t['status']=='completed' or str(row['stage']) in t['probes']:raise ValueError('Available candidate silently discarded.')
            continue
        if row['kind']=='seed':q=np.asarray(case['qseed'])
        else:
            k=PEAKS.index(row['peak']);t=traces[f'{Path(row["parent_id"]).stem}-{row["arm"]}-p{k}']
            q=np.asarray(t['probes'][str(row['stage'])])
            if not np.allclose(q,t['stages'][row['stage']]['q'],atol=1e-12,rtol=0):raise ValueError('Probe does not match its recorded stage.')
        if not np.allclose(q,row['q'],atol=1e-12,rtol=0) or not geom.feasible(q):raise ValueError('Candidate state/guard differs.')
        if sha(row['input'])!=row['input_sha256'] or row['B']['input_sha256']!=row['input_sha256']:raise ValueError('Quench input SHA differs.')
        cell,frac=geom.pose(q);s=Structure.from_file(row['input']);delta=np.asarray(s.frac_coords)-frac;delta-=np.rint(delta)
        if tuple(s.atomic_numbers)!=tuple(geom.numbers) or not np.allclose(cell,s.lattice.matrix,atol=1e-12,rtol=0) or np.abs(delta).max()>1e-12:raise ValueError('Saved actual candidate changed.')
        if 'point'in row and sha(row['point']['pose'])!=row['point']['B']['relaxed_sha256']:raise ValueError('Final pose SHA differs.')
    for i,parent in enumerate(plan['parent_ids']):
        base=native['baseline'][i];case=plan['cases'][i]
        if base['id']!=parent or sha(base['pose'])!=case['B']['relaxed_sha256'] or any(base['B'].get(k)!=v for k,v in case['B'].items() if k not in ('volume_A3','volume_ratio','separated')):raise ValueError('Original baseline changed.')
        selected=choose(native['baseline'][i],[r for r in data['attempts'] if r['parent_id']==parent])
        if any(data['selected'][m][i]!=p for m,p in selected.items()):raise ValueError('Common zero-CF pool or selection changed.')
    if data['NFE_steering']!=sum(t['NFE_steering'] for t in data['trajectories']) or data['NFE_quench']!=sum(r.get('B',{}).get('NFE',0) for r in data['attempts']):raise ValueError('NFE subledger differs.')
    if data['NFE_total']!=native['NFE']+data['NFE_steering']+data['NFE_quench']:raise ValueError('NFE total differs.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--count',type=int,choices=(1,16),default=16);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm GPU.')
    from ase import Atoms
    from ase.io import write
    source=a.source.resolve();proof=source_proof(source,a.native.resolve());old=load(source/'plan.json');root=a.run.resolve();root.mkdir(parents=True,exist_ok=False)
    for name in ('rpc','inputs'): (root/name).mkdir()
    plan=dict(policy=POLICY,count=a.count,parent_ids=list(PARENTS[:a.count]),cases=old['cases'][:a.count],source=str(source),native_root=str(a.native.resolve()),
        native_source_sha256=sha(a.native/'native.json'),core_sha256=core_hashes(),role='Integration1 only; no population efficacy inference.' if a.count==1 else 'Full development16; unseen-parent confirmation required.')
    save(root/'plan.json',plan);client=None
    try:
        scorer=GradientScorer();official=GradientScorer();promote(scorer,scorer.torch.float64);client=Mechanics(root/'rpc')
        native=dict(passed=False,cases=[],baseline=[],plan_sha256=sha(root/'plan.json'),mechanics=client.ready)
        for case,initial in zip(plan['cases'],proof['cases']):
            if case['parent_id']!=initial['parent_id'] or sha(case['B']['relaxed_path'])!=case['B']['relaxed_sha256'] or sha(case['raw_path'])!=case['B']['input_sha256']:raise ValueError('Original parent/input changed.')
            base=point(case['parent_id'],case['raw_path'],case['B'],official);geom=Geometry(**initial['geometry']);q=np.asarray(initial['qseed'])
            old=case['predictions']['after']
            if max(abs(base['gap']-old['crystalformer_gap']),abs(base['eform']-old['crystalformer_eform']))>1e-5:raise ValueError('Original frozen CF control changed.')
            point0=client.points(geom,np.zeros((1,geom.size)))[0];validate_control(point0['energy_eV'],float(np.linalg.norm(point0['forces_eV_A'],axis=1).max()),case['B'],point0['checkpoint_sha256'])
            seed_point=client.points(geom,np.array([q]))[0];record=initial['physics_seed']['record']
            if abs(seed_point['energy_eV']-record['energy_eV'])/geom.n>1e-4 or np.abs(np.asarray(seed_point['forces_eV_A'])-record['forces_eV_A']).max()>1e-4 or np.abs(np.asarray(seed_point['stress_eV_A3'])-record['stress_eV_A3']).max()>1e-4:raise ValueError('Seed physical point changed.')
            cf=precision_score(scorer,geom,q,scorer.torch.float64,True);old_cf=initial['CF_seed']
            if max(abs(cf[k]-old_cf[k]) for k in ('gap','eform'))>1e-5 or np.linalg.norm(geom.project(cf['gradient'])-old_cf['gradient'])>1e-6+.001*np.linalg.norm(old_cf['gradient']):raise ValueError('Native CF seed changed.')
            native['cases'].append(initial);native['baseline'].append(base)
        native.update(passed=True,NFE=client.NFE);save(root/'native.json',native)
        rows=[];trajectories=[];selected={m:[] for m in OUTPUTS};steering=0
        for i,(parent,case,base) in enumerate(zip(plan['parent_ids'],native['cases'],native['baseline'])):
            geom=Geometry(**case['geometry']);specs=[s for s in specifications([parent])];byid={s['id']:dict(s) for s in specs}
            seed=byid[Path(parent).stem+'-seed'];seed['q']=case['qseed']
            for arm in ('zero','full','reverse','random','fixed'):
                for k,peak in enumerate(PEAKS):
                    t=trajectory(root,case,base,arm,peak,k,scorer,client);trajectories.append(t);steering+=t['NFE_steering']
                    for stage in PROBE_STAGES:
                        row=byid[f'{Path(parent).stem}-{arm}-p{k}-s{stage}']
                        if str(stage) in t['probes']:row['q']=t['probes'][str(stage)]
                        else:row.update(status='steering_failed',error=t.get('error','Missing stage'))
                    print(__import__('json').dumps(dict(stage='steering',parent=parent,arm=arm,peak=peak,status=t['status'],NFE=client.NFE)),flush=True)
            pending=[];tasks=[]
            for row in byid.values():
                if 'q' not in row:continue
                cell,frac=geom.pose(row['q']);path=root/'inputs'/(row['id']+'.vasp');write(path,Atoms(numbers=geom.numbers,cell=cell,scaled_positions=frac,pbc=True),format='vasp',direct=True,sort=False)
                row.update(input=str(path),input_sha256=sha(path));pending.append(row);tasks.append(dict(id=row['id'],action='relax',source=str(path),output=str(root/'B'/row['id'])))
            for row,B in zip(pending,client.request(tasks)):
                row.update(status=B['status'],B=B)
                if B['status']=='converged':
                    try:row['point']=point(row['id'],row['input'],B,official,base['B']['volume_A3'])
                    except ValueError as error:row.update(status='prediction_failed',error=repr(error))
            current=list(byid.values());rows.extend(current)
            for m,pick in choose(base,current).items():selected[m].append(pick)
            data=dict(completed_parents=i+1,attempts=rows,trajectories=trajectories,selected=selected,NFE_steering=steering,NFE_quench=sum(r.get('B',{}).get('NFE',0) for r in rows),NFE_total=client.NFE,
                reused_native_proof_NFE=proof['NFE'],CF=scorer.metadata,official_CF=official.metadata)
            save(root/'results.json',data);print(__import__('json').dumps(dict(stage='parent_complete',parents=i+1,attempts=len(rows),NFE=client.NFE)),flush=True)
        validate(root,plan,native,data)
        for m in OUTPUTS:
            export_evaluation(root,m,data['selected'][m],plan,official)
            manifest=root/f'evaluation-{m}/inventory-249/inventory.json';inventory=load(manifest)
            inventory['role']=plan['role'];save(manifest,inventory)
        save(root/'complete.json',dict(completed=True,plan_sha256=sha(root/'plan.json'),native_sha256=sha(root/'native.json'),results_sha256=sha(root/'results.json'),population_efficacy_eligible=a.count==16))
    except Exception as error:
        save(root/'failed.json',dict(error=repr(error),traceback=traceback.format_exc()));raise
    finally:
        if client is not None:client.close()


if __name__=='__main__':main()
