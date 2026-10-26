"""Actual-double denoised CF gradients with matched full MatterSim outcomes."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from adapter import Denoiser, sha
from mapping import LatentGeometry
from native import GradientScorer, evaluate, promote, save
from double_native import DOUBLE_POLICY, case_passes
from denoised_policy import POLICY, PARENTS, TIMES, SCALES, ARMS, OUTPUTS
from denoised_math import controlled_directions, feasible_pose, validate_coverage
from denoised_shared import point,best,validate_control

DIRECTED = Path(__file__).resolve().parents[1]/'directed_tto'
sys.path.insert(0, str(DIRECTED))
from geometry import Geometry
from numerics import score as precision_score
from runtime import Mechanics
from evaluation_export import export_evaluation


def load(path):
    return json.loads(Path(path).read_text())


def core_hashes():
    names = ('adapter.py','mapping.py','precision.py','native.py','double_native.py','denoised_policy.py','denoised_math.py','denoised_shared.py','denoised_search.py','denoised_validate.py')
    result = {n:sha(Path(__file__).with_name(n)) for n in names}
    for name in ('geometry.py','numerics.py','cf_gradient.py','runtime.py','mechanics.py','settings.py','evaluation_export.py','probe.py','directions.py'):
        result['directed/'+name] = sha(DIRECTED/name)
    for name in ('policy.py','search.py','scorer.py','attractors.py'):
        result['basin/'+name] = sha(DIRECTED.parent/'basin_search'/name)
    result['Gaussian'] = sha(DIRECTED.parents[1]/'smoacs/block_alternating/gaussian.py')
    return result


def validate_native(source, native_root):
    from probe import code_hashes
    proof = load(native_root/'native.json')
    if proof['policy'] != DOUBLE_POLICY or not proof['passed'] or proof['original_float32_gate_reclassified'] or len(proof['cases']) != 8 or not all(case_passes(c) for c in proof['cases']):
        raise ValueError('Actual-F64 native proof missing/failed.')
    expected = [(p,t) for p in PARENTS[:4] for t in TIMES]
    if [(r['parent_id'],r['time']) for r in proof['cases']] != expected:
        raise ValueError('Native proof cohort changed.')
    for name, expected_sha in proof['implementation_sha256'].items():
        if sha(Path(__file__).with_name(name)) != expected_sha:
            raise ValueError('Native scientific implementation changed: '+name)
    for name, expected_sha in proof['source_hashes'].items():
        if sha(source/name) != expected_sha:
            raise ValueError('Native original evidence changed.')
    if load(source/'native-probe.json')['code_sha256']!=code_hashes():
        raise ValueError('Previously proven shared CF/physics implementation changed.')
    for c in proof['cases']:
        if sha(c['clean_pose']) != c['clean_pose_sha256']:
            raise ValueError('Native saved clean pose changed.')
    return proof


def prepare(root, source, native_root, scorer, official, denoiser, client):
    from pymatgen.core import Structure
    from ase import Atoms
    from ase.io import write
    proof = validate_native(source, native_root)
    torch=scorer.torch
    if any(p.dtype!=torch.float64 or not p.is_cuda for p in denoiser.module.parameters()) or any(p.dtype!=torch.float64 or not p.is_cuda for m in scorer.models for p in m.parameters()):
        raise ValueError('Actual proposal models require double CUDA parameters.')
    old = load(source/'plan.json')
    if tuple(old['parent_ids']) != PARENTS or not all(c['neutral'] for c in old['cases']):
        raise ValueError('Original development16 changed.')
    plan = dict(policy=POLICY, parent_ids=list(PARENTS), cases=old['cases'], source=str(source),
        source_hashes=proof['source_hashes'], native_root=str(native_root), native_sha256=sha(native_root/'native.json'), core_sha256=core_hashes())
    save(root/'plan.json', plan)
    baselines, operators, controls = [], [], []
    for case in plan['cases']:
        parent = case['parent_id']
        if sha(case['B']['relaxed_path']) != case['B']['relaxed_sha256'] or sha(case['raw_path']) != case['B']['input_sha256']:
            raise ValueError('Original input/final changed.')
        base = point(parent, case['raw_path'], case['B'], official)
        expected = case['predictions']['after']
        error = max(abs(base['gap']-expected['crystalformer_gap']), abs(base['eform']-expected['crystalformer_eform']))
        if error > 1e-5:
            raise ValueError('Source frozen CF control failed.')
        s = Structure.from_file(base['pose'])
        physical = client.points(Geometry(s.atomic_numbers,s.lattice.matrix,s.frac_coords), np.zeros((1,3*len(s)+6)))[0]
        control = validate_control(physical['energy_eV'],float(np.linalg.norm(physical['forces_eV_A'],axis=1).max()),case['B'],physical['checkpoint_sha256'])
        if not control['passed']:
            raise ValueError('Source fixed-pose MatterSim control failed.')
        controls.append(dict(parent_id=parent,CF_error=error,physical=control))
        baselines.append(base)
        reference = LatentGeometry(s.atomic_numbers,s.lattice.matrix,s.frac_coords)
        index = int(Path(parent).stem.split('-')[1])
        for time in TIMES:
            with scorer.torch.no_grad():
                graph = denoiser.graph(reference,scorer.torch.zeros(reference.size,device='cuda'))
                noisy,_ = denoiser.forward_noise(graph,time,[POLICY['seed'],index,int(time*1000)])
            identity=dict(parent_id=parent,time=time,reference=dict(numbers=reference.numbers.tolist(),cell=reference.cell.tolist(),frac=reference.frac.tolist()))
            try:
                latent = LatentGeometry(reference.numbers,noisy.cell[0].cpu().double().numpy(),noisy.pos.cpu().double().numpy())
            except ValueError as error:
                if str(error) != 'A right-handed nonsingular cell is required.':
                    raise
                operators.append(dict(identity,status='invalid_noisy_cell',error=repr(error)))
                continue
            zero = np.zeros(latent.size)
            try:
                ad = evaluate(denoiser,scorer,latent,time,zero,scorer.torch.float64,True)
            except ValueError as error:
                if str(error) != 'Denoising produced a nonpositive cell determinant.':
                    raise
                operators.append(dict(identity,status='invalid_decoded_seed',error=repr(error)))
                continue
            stop = evaluate(denoiser,scorer,latent,time,zero,scorer.torch.float64,True,stop=True)
            gradient, stopped = ad.pop('gradient'), stop.pop('gradient')
            if max(abs(ad[k]-stop[k]) for k in ('gap','eform')) > 1e-5:
                raise ValueError('Jacobian ablation changed its forward function.')
            folder = root/'operators'/(Path(parent).stem+'-t'+str(int(time*1000)))
            folder.mkdir(parents=True)
            path = folder/'seed.vasp'
            cell,frac = np.asarray(ad['clean_cell'])[0],np.asarray(ad['clean_frac'])
            write(path,Atoms(numbers=latent.numbers,cell=cell,scaled_positions=frac,pbc=True),format='vasp',direct=True,sort=False)
            gap,ef = official.predict(path)
            if max(abs(gap-ad['gap']),abs(ef-ad['eform'])) > 1e-5:
                raise ValueError('New operator same-pose official CF control failed.')
            direct_geom = Geometry(latent.numbers,cell,frac)
            direct = precision_score(scorer,direct_geom,zero,scorer.torch.float64,True)
            if max(abs(direct[k]-ad[k]) for k in ('gap','eform')) > 1e-5:
                raise ValueError('Direct CF control changed the common clean seed.')
            directions = controlled_directions(gradient,stopped,direct['gradient'],latent.project,np.random.default_rng([POLICY['seed'],index,int(time*1000),731]))
            operators.append(dict(identity,status='ready',
                latent=dict(numbers=latent.numbers.tolist(),cell=latent.cell.tolist(),frac=latent.frac.tolist()), clean=ad,
                gradient=gradient.tolist(),stop_gradient=stopped.tolist(),direct_gradient=direct['gradient'].tolist(),
                directions={k:v.tolist() for k,v in directions.items()},seed_input=str(path),seed_sha256=sha(path),same_pose_CF=dict(gap=gap,eform=ef)))
        save(root/'native-progress.json',dict(controls=controls,operators=operators,baseline=baselines,NFE=client.NFE))
        print(json.dumps(dict(stage='native',parent=parent,completed=len(baselines))),flush=True)
    result = dict(passed=True,controls=controls,operators=operators,baseline=baselines,NFE=client.NFE,core_sha256=core_hashes(),
        plan_sha256=sha(root/'plan.json'),MatterGen=denoiser.metadata,CF=scorer.metadata,official_CF=official.metadata,mechanics=client.ready,
        CUDA_free_and_total_bytes=list(torch.cuda.mem_get_info()))
    save(root/'native.json',result)
    return plan,result


def make_proposal(root, operator, specification, denoiser, scorer):
    from ase import Atoms
    from ase.io import write
    reference = operator['reference']
    if operator['status'] != 'ready':
        return dict(specification,status=operator['status'])
    latent = LatentGeometry(**operator['latent'])
    q=np.zeros(latent.size)
    if specification['kind'] == 'seed':
        cell,frac = np.asarray(operator['clean']['clean_cell'])[0],np.asarray(operator['clean']['clean_frac'])
        path = Path(operator['seed_input'])
    else:
        arm,scale = specification['arm'],specification['scale']
        direction = np.asarray(operator['directions'][arm])
        q=scale*direction
        if np.linalg.norm(direction) < 1e-12:
            return dict(specification,status='stationary_attempt')
        if arm == 'direct':
            geometry = Geometry(latent.numbers,np.asarray(operator['clean']['clean_cell'])[0],operator['clean']['clean_frac'])
            cell,frac = geometry.pose(scale*direction)
        else:
            decoded = evaluate(denoiser,scorer,latent,operator['time'],scale*direction,scorer.torch.float64)
            cell,frac = np.asarray(decoded['clean_cell'])[0],np.asarray(decoded['clean_frac'])
        path = root/'inputs'/(specification['id']+'.vasp')
    evidence=dict(proposal_q=q.tolist(),proposal_cell=np.asarray(cell).tolist(),proposal_frac=np.asarray(frac).tolist())
    if not feasible_pose(latent.numbers,cell,frac,np.asarray(reference['cell']),np.asarray(reference['frac'])):
        return dict(specification,**evidence,status='geometric_rejection')
    if specification['kind']=='ray':
        write(path,Atoms(numbers=latent.numbers,cell=cell,scaled_positions=frac,pbc=True),format='vasp',direct=True,sort=False)
    return dict(specification,**evidence,status='pending_B',input=str(path),input_sha256=sha(path))


def select_parent(base,attempts):
    energy=base['B']['final_energy_eV']/base['B']['num_atoms']
    pool=[base]+[r['point'] for r in attempts if r['kind']=='seed' and 'point' in r]
    selected=dict(baseline=base,seeds=best(pool,energy))
    selected.update({a:best(pool+[r['point'] for r in attempts if r.get('arm')==a and 'point' in r],energy) for a in ARMS})
    return selected


def study(root,plan,native,denoiser,scorer,official,client):
    rows=[];selected={m:[] for m in OUTPUTS};start=client.NFE
    for i,(parent,base) in enumerate(zip(PARENTS,native['baseline'])):
        attempts=[]
        for operator in [q for q in native['operators'] if q['parent_id']==parent]:
            stem=Path(parent).stem+'-t'+str(int(operator['time']*1000))
            specs=[dict(id=stem+'-seed',parent_id=parent,time=operator['time'],kind='seed')]
            specs += [dict(id=f'{stem}-{a}-s{k}',parent_id=parent,time=operator['time'],kind='ray',arm=a,scale=s) for a in ARMS for k,s in enumerate(SCALES)]
            for spec in specs:
                try:
                    row=make_proposal(root,operator,spec,denoiser,scorer)
                except ValueError as error:
                    if str(error) not in ('Denoising produced a nonpositive cell determinant.','Invalid trial cell/positions.'):
                        raise
                    row=dict(spec,status='decoded_geometry_rejection',error=repr(error))
                attempts.append(row)
        pending=[r for r in attempts if r['status']=='pending_B']
        tasks=[dict(id=r['id'],action='relax',source=r['input'],output=str(root/'B'/r['id'])) for r in pending]
        results=client.request(tasks)
        for row,B in zip(pending,results):
            row.update(status=B['status'],B=B)
            if B['status']=='converged':
                try:
                    row['point']=point(row['id'],row['input'],B,official,base['B']['volume_A3'])
                except ValueError as error:
                    row.update(status='prediction_failed',error=repr(error))
        rows.extend(attempts)
        for method,q in select_parent(base,attempts).items():
            selected[method].append(q)
        data=dict(attempts=rows,selected=selected,completed_parents=i+1,NFE_native=start,NFE_search=client.NFE-start,NFE_total=client.NFE,
            CF=scorer.metadata,official_CF=official.metadata,MatterGen=denoiser.metadata)
        save(root/'results.json',data)
        print(json.dumps(dict(stage='search',completed_parents=i+1,attempts=len(rows),NFE=client.NFE)),flush=True)
    validate_coverage(rows)
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--export-only',action='store_true')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm GPU.')
    root=a.run.resolve()
    if a.export_only:
        from denoised_validate import validate
        plan,native,data=load(root/'plan.json'),load(root/'native.json'),load(root/'results.json')
        validate(root,plan,native,data)
        official=GradientScorer()
    else:
        root.mkdir(parents=True,exist_ok=False)
        (root/'inputs').mkdir();(root/'rpc').mkdir()
        client=None
        try:
            denoiser=Denoiser(stable_offsets=True);scorer=GradientScorer();official=GradientScorer()
            denoiser.module.double();promote(scorer,scorer.torch.float64)
            client=Mechanics(root/'rpc')
            plan,native=prepare(root,a.source.resolve(),a.native.resolve(),scorer,official,denoiser,client)
            data=study(root,plan,native,denoiser,scorer,official,client)
        except Exception as error:
            save(root/'failed.json',dict(error=repr(error),traceback=traceback.format_exc()))
            raise
        finally:
            if client is not None:client.close()
        from denoised_validate import validate
        validate(root,plan,native,data)
    for method in OUTPUTS:
        export_evaluation(root,method,data['selected'][method],plan,official)
        inventory=root/f'evaluation-{method}/inventory-249/inventory.json';items=load(inventory)
        items['role']='Development16 denoised-gradient comparison';save(inventory,items)
    save(root/'complete.json',dict(completed=True,plan_sha256=sha(root/'plan.json'),native_sha256=sha(root/'native.json'),results_sha256=sha(root/'results.json')))


if __name__=='__main__':
    main()
