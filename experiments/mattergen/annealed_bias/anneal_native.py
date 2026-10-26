"""Native component derivative proof for conservative annealed CF bias."""
import argparse
import json
import os
from pathlib import Path
import traceback
import numpy as np
from anneal_policy import POLICY,PARENTS,SEED,ARMS,PEAKS,weights
from anneal_math import random_slope,bounded_project
from anneal_fire import minimize
from anneal_objective import Provider
from anneal_shared import (GradientScorer,Geometry,promote,precision_score,common_seed,code_hashes,load,sha,Mechanics,
    validate_control,implementation_hashes)

NATIVE_POLICY=dict(parents=16,finite_difference_parents=4,CF_FD_steps=[1e-5,1e-6],MF_FD_steps=[.004,.002],
    CF_absolute=.001,CF_relative=.02,MF_total_energy_absolute=.005,MF_relative=.01,
    source_CF_tolerance=1e-5,same_pose_CF_tolerance=1e-5,
    micro_stages=[0,7],micro_updates_per_stage=2,micro_parents=4,
    component_derivative_tests_not_joint_coarse_FD=True,post_quench_efficacy_trials=0)


def save(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=convert)+'\n')


def close(fd,predicted,absolute,relative):
    return bool(abs(fd-predicted)<=absolute+relative*max(abs(fd),abs(predicted)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm GPU.')
    root=args.run.resolve();root.mkdir(parents=True,exist_ok=False);(root/'rpc').mkdir();client=None
    result=dict(passed=False,method=POLICY,native_policy=NATIVE_POLICY,cases=[],source=str(args.source.resolve()),implementation_sha256=implementation_hashes())
    try:
        from ase import Atoms
        from ase.io import write
        from pymatgen.core import Structure
        source=args.source.resolve();plan=load(source/'plan.json');proof=load(source/'native-probe.json')
        if tuple(plan['parent_ids'])!=PARENTS or not proof['passed'] or proof['code_sha256']!=code_hashes() or not all(c['neutral'] for c in plan['cases']):
            raise ValueError('Original16 source/proven implementation changed.')
        result['source_hashes']={n:sha(source/n) for n in ('plan.json','native-probe.json','direction-results.json','report.json')}
        official=GradientScorer();scorer=GradientScorer();torch=scorer.torch;promote(scorer,torch.float64)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if any(p.dtype!=torch.float64 or not p.is_cuda for m in scorer.models for p in m.parameters()):raise ValueError('Actual CF proposal requires double CUDA.')
        client=Mechanics(root/'rpc');result.update(CF=scorer.metadata,official_CF=official.metadata,mechanics=client.ready)
        for i,case in enumerate(plan['cases']):
            parent=case['parent_id'];index=int(Path(parent).stem.split('-')[1]);path=Path(case['B']['relaxed_path'])
            if sha(path)!=case['B']['relaxed_sha256']:raise ValueError('Original source pose changed.')
            s=Structure.from_file(path);geom=Geometry(s.atomic_numbers,s.lattice.matrix,s.frac_coords)
            gap,ef=official.predict(path);old=case['predictions']['after'];source_error=max(abs(gap-old['crystalformer_gap']),abs(ef-old['crystalformer_eform']))
            if source_error>1e-5:raise ValueError('Source official CF control failed.')
            value=client.points(geom,np.zeros((1,geom.size)))[0]
            control=validate_control(value['energy_eV'],float(np.linalg.norm(value['forces_eV_A'],axis=1).max()),case['B'],value['checkpoint_sha256'])
            q,draw=common_seed(geom,index,SEED)
            cf=precision_score(scorer,geom,q,torch.float64,True);cf['gradient']=geom.project(cf['gradient'])
            if not np.isfinite([cf['gap'],cf['eform'],cf['loss']]).all() or not np.isfinite(cf['gradient']).all():
                raise ValueError('Nonfinite native CF component.')
            slope=random_slope(cf['gradient'],np.random.default_rng([SEED,index,501]),geom.project)
            folder=root/Path(parent).stem;folder.mkdir();seed=folder/'seed.vasp';cell,frac=geom.pose(q)
            write(seed,Atoms(numbers=geom.numbers,cell=cell,scaled_positions=frac,pbc=True),format='vasp',direct=True,sort=False)
            fg,fe=official.predict(seed);same_error=max(abs(fg-cf['gap']),abs(fe-cf['eform']))
            if same_error>1e-5:raise ValueError('Same-pose double/official CF control failed.')
            provider=Provider(geom,scorer,client,case['B']['final_energy_eV'],q,cf['loss'],slope)
            physical,_=provider.components(q,False);provider.cf[q.tobytes()]=cf
            record=dict(parent_id=parent,source_pose_sha256=sha(path),source_CF_error=source_error,physical_source_control=control,same_pose_CF_error=same_error,
                geometry=dict(numbers=geom.numbers,cell=geom.cell,frac=geom.frac),qseed=q,seed_draw=draw,seed_input=str(seed),seed_sha256=sha(seed),
                CF_seed=cf,physics_seed=physical,random_slope=slope,checks=[],mixing=[],micro=[])
            if i<4:
                rng=np.random.default_rng([SEED,index,777])
                for block in ('atomic','strain','both'):
                    direction=geom.project(rng.normal(size=geom.size))
                    if block=='atomic':direction[-6:]=0
                    elif block=='strain':direction[:-6]=0
                    direction/=np.linalg.norm(direction)
                    for epsilon in NATIVE_POLICY['CF_FD_steps']:
                        plus=precision_score(scorer,geom,q+epsilon*direction,torch.float64)['loss'];minus=precision_score(scorer,geom,q-epsilon*direction,torch.float64)['loss']
                        fd=(plus-minus)/(2*epsilon);predicted=float(cf['gradient']@direction)
                        record['checks'].append(dict(kind='CF',block=block,epsilon=epsilon,FD=fd,predicted=predicted,passed=close(fd,predicted,.001,.02)))
                    for epsilon in NATIVE_POLICY['MF_FD_steps']:
                        points=client.points(geom,np.array([q+epsilon*direction,q-epsilon*direction]));fd=(points[0]['energy_eV']-points[1]['energy_eV'])/(2*epsilon*geom.n);predicted=float(physical['gradient_E_total']@direction/geom.n)
                        record['checks'].append(dict(kind='MatterSim_per_atom',block=block,epsilon=epsilon,FD=fd,predicted=predicted,
                            passed=close(fd,predicted,.005/geom.n,.01)))
                if not all(c['passed'] for c in record['checks']):
                    record['passed']=False;result['cases'].append(record);result['NFE']=client.NFE
                    save(root/'native.json',result)
                    raise ValueError('Annealed-bias component derivative failed before micro-updates.')
                for arm in ARMS:
                    for strength in PEAKS:
                        state=provider.evaluate(q,strength,arm)
                        signed=cf['gradient'] if arm in ('full','fixed') else -cf['gradient'] if arm=='reverse' else slope if arm=='random' else np.zeros(geom.size)
                        expected=geom.project(physical['gradient_E_total']/geom.n+strength*signed)
                        error=float(np.linalg.norm(state['gradient']-expected))
                        record['mixing'].append(dict(arm=arm,weight=strength,gradient_error=error,passed=bool(error<=1e-10)))
                    micro=Provider(geom,scorer,client,case['B']['final_energy_eV'],q,cf['loss'],slope,
                        initial_physics=physical,initial_cf=cf)
                    state=q.copy();stages=[]
                    for slot in NATIVE_POLICY['micro_stages']:
                        strength=weights(arm,.1)[slot]
                        outcome=minimize(state,lambda x:micro.evaluate(x,strength,arm),lambda x:bounded_project(geom,x),geom.feasible,dict(steps=2))
                        state=outcome['q'];stages.append(dict(stage=slot,weight=strength,**outcome))
                        if not np.isfinite(state).all() or not geom.feasible(state):raise ValueError('Native micro-step invalid.')
                    record['micro'].append(dict(arm=arm,stages=stages,costs=micro.costs()))
            record['passed']=bool(source_error<=1e-5 and control['passed'] and same_error<=1e-5 and all(c['passed'] for c in record['checks']) and all(c['passed'] for c in record['mixing']))
            result['cases'].append(record);result['NFE']=client.NFE;save(root/'native.json',result)
            print(json.dumps(dict(parent=parent,completed=i+1,passed=record['passed'],NFE=client.NFE)),flush=True)
            if not record['passed']:raise ValueError('Annealed-bias component native gate failed.')
        if len(result['cases'])!=16 or sum(len(c['checks']) for c in result['cases'])!=48 or sum(len(c['mixing']) for c in result['cases'])!=60:
            raise ValueError('Component native proof incomplete.')
        result['passed']=True
    except Exception as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if client is not None:
            result['NFE']=client.NFE;client.close()
        save(root/'native.json',result)


if __name__=='__main__':main()
