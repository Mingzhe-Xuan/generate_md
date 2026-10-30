"""Paired exact-cutoff pure-CSP replay and identical MatterSim relaxations."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from boundary import Boundary
from comparison import eligible

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'forward_inverse'))
from csp_model import CSP,load_file,sha
from physics_client import Physics
math=load_file(HERE.parent/'random_score_probe/trajectory_math.py','early_domain')

ARMS=('mean800','sample800','mean1000')
POLICY=dict(arms=ARMS,main_comparison=['mean800','mean1000'],control='sample800',
    fmax=.01,max_steps=1600,maxstep=.05,cell_relax=True,preserve_symmetry=False,
    N=1000,cutoff_completed_rounds=800,batch_size=8,num_batches=3,
    same_endpoint_energy_tolerance_eV_atom=.001,
    geometry_guards=dict(minimum_pair_A=.5,minimum_volume_per_atom_A3=2.,minimum_cell_singular_A=.3))


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main(source,root):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from ase.geometry import find_mic
    root.mkdir(parents=True,exist_ok=False);(root/'inputs').mkdir()
    physics=None;started=time.perf_counter()
    try:
        old=json.loads((source/'complete.json').read_text());oldplan=json.loads((source/'plan.json').read_text())
        comp=json.loads((source/'composition_plan.json').read_text())
        if not old['passed'] or len(old['paths'])!=24 or oldplan['seed']!=comp['seed']:
            raise ValueError('Invalid original cohort.')
        model=CSP();del model.double
        from mattergen.generator import CrystalGenerator,seed_all
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.diffusion.sampling import pc_sampler
        validation=load_file(HERE.parent/'csp/validation.py','early_validation')
        compositions=[validation.validate_composition(c['counts']) for c in comp['compositions']]
        info=MatterGenCheckpointInfo(model_path=Path('results/mattergen/csp-assets').resolve(),load_epoch='last')
        generator=CrystalGenerator(checkpoint_info=info,batch_size=8,num_batches=3,target_compositions_dict=compositions,
            sampling_config_name='csp',properties_to_condition_on={},diffusion_guidance_factor=0.,record_trajectories=False,seed=comp['seed'])
        config=generator.sampling_config;effective=OmegaConf.to_container(config,resolve=True)
        validation.validate_sampling(effective)
        if effective!=oldplan['sampling_config']:raise ValueError('Original time grid/config changed.')
        seed_all(comp['seed']);batches=list(generator.get_condition_loader(config,compositions))
        sampler=instantiate(config.sampler_partial)(pl_module=SimpleNamespace(diffusion_module=model.native,device=torch.device('cuda')))
        if len(batches)!=3 or sampler._n_steps_corrector!=1:raise ValueError('PC layout changed.')
        records=[];audits=[];forward_calls=[0];expected=[None]
        def hook(module,inputs,output):
            if not torch.equal(inputs[0].atomic_numbers,expected[0]) or 'dft_band_gap' in inputs[0]:
                raise ValueError('Native species/property input changed.')
            if not all(v.is_cuda and torch.isfinite(v).all() for v in (output.pos,output.cell)):
                raise ValueError('Native output not finite CUDA.')
            forward_calls[0]+=1
        handle=model.native.model.register_forward_hook(hook)
        save(root/'plan.json',dict(policy=POLICY,source=str(source),source_sha256={n:sha(source/n) for n in ('complete.json','plan.json','composition_plan.json')},
            checkpoint=model.metadata,sampling_config=effective,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            code_sha256={str(p):sha(p) for folder in (HERE,HERE.parent/'forward_inverse') for p in folder.glob('*.py')}))
        try:
            for b,(condition,mask) in enumerate(batches):
                seed=comp['seed']+b;seed_all(seed);expected[0]=condition.atomic_numbers.clone().cuda()
                if mask or len(condition.to_data_list())!=8:raise ValueError('Unexpected mask/batch.')
                originals=[p for p in old['paths'] if p['batch']==b]
                if len(originals)!=8:raise ValueError('Original paths incomplete.')
                originals.sort(key=lambda p:p['trajectory'])
                for p,piece in zip(originals,condition.to_data_list()):
                    if sha(source/p['path'])!=p['sha256'] or sha(source/p['terminal_path'])!=p['terminal_sha256']:
                        raise ValueError('Original arrays changed.')
                    with np.load(source/p['path']) as d:
                        if not np.array_equal(d['numbers'],piece.atomic_numbers.cpu().numpy()):raise ValueError('Ordered species changed.')
                calls=[0];snapshots={};times={};boundary_state=[];verified_next=[False]
                original=sampler._score_fn;tic=time.perf_counter()
                def capture(rounds,sample,mean):
                    torch.cuda.synchronize();times[rounds]=time.perf_counter()-tic
                    for final in (sample,mean):
                        if not torch.equal(final.atomic_numbers,expected[0]) or not all(torch.isfinite(v).all() for v in (final.pos,final.cell)):
                            raise ValueError('Boundary finite/species check failed.')
                    if rounds==800:
                        boundary_state[:]=[sample.pos.clone(),sample.cell.clone()]
                    choices=[('mean800',mean),('sample800',sample)] if rounds==800 else [('mean1000',mean)]
                    for arm,final in choices:
                        snapshots[arm]=[]
                        for j,piece in enumerate(final.clone().cpu().to_data_list()):
                            trajectory=b*8+j;path=root/'inputs'/f'{trajectory:02d}-{arm}.npz'
                            np.savez_compressed(path,numbers=piece.atomic_numbers.numpy(),cell=piece.cell[0].double().numpy(),frac=piece.pos.double().numpy())
                            snapshots[arm].append(dict(trajectory=trajectory,batch=b,seed=seed,system=originals[j]['system'],arm=arm,
                                path=str(path.relative_to(root)),sha256=sha(path)))
                def checked(x,t):
                    if not torch.equal(x.atomic_numbers,expected[0]) or not all(v.is_cuda and torch.isfinite(v).all() for v in (x.pos,x.cell)):
                        raise ValueError('PC input species/device/finite failure.')
                    if calls[0]==1600:
                        if not torch.equal(x.pos,boundary_state[0]) or not torch.equal(x.cell,boundary_state[1]):
                            raise ValueError('Captured sample800 is not exact continuation state.')
                        verified_next[0]=True
                    calls[0]+=1
                    return original(x,t)
                sampler._score_fn=checked
                try:
                    with Boundary(pc_sampler,torch,capture) as boundary:
                        sample,mean=sampler.sample(condition,mask)
                finally:sampler._score_fn=original
                if boundary.calls!=2000 or calls[0]!=2000 or not verified_next[0]:raise ValueError('Native schedule/boundary audit failed.')
                replay=[]
                for j,piece in enumerate(mean.clone().cpu().to_data_list()):
                    saved=snapshots['mean1000'][j]
                    with np.load(root/saved['path']) as d:
                        if not np.array_equal(d['cell'],piece.cell[0].numpy()) or not np.array_equal(d['frac'],piece.pos.numpy()):
                            raise ValueError('Full returned mean differs from captured terminal.')
                    with np.load(source/originals[j]['terminal_path']) as d:
                        delta=piece.pos.numpy()-d['frac'];delta-=np.rint(delta)
                        replay.append(dict(trajectory=b*8+j,max_cell_error_A=float(np.max(np.abs(piece.cell[0].numpy()-d['cell']))),
                            fractional_image_RMS=float(np.sqrt(np.mean(delta**2)))))
                for arm in ARMS:records.extend(snapshots[arm])
                audits.append(dict(batch=b,seed=seed,score_calls=calls[0],replacements=boundary.calls,
                    rng_unchanged=True,next_corrector_is_sample800=True,seconds800=times[800],seconds1000=times[1000],old_terminal_differences=replay))
                save(root/'sampling-progress.json',dict(inputs=records,audit=audits,raw_forwards=forward_calls[0]))
                print(json.dumps(dict(stage='sampling',batch=b,seconds800=times[800],seconds1000=times[1000])),flush=True)
        finally:handle.remove()
        if len(records)!=72 or forward_calls[0]!=6000:raise ValueError('Incomplete cohort.')
        physics=Physics(root/'physics');relaxations=[]
        # Parent-interleaved arms keep matched endpoints near each other in execution order.
        for i in range(24):
            for arm in ARMS:
                item=next(r for r in records if r['trajectory']==i and r['arm']==arm)
                with np.load(root/item['path']) as d:
                    numbers=d['numbers'];cell=d['cell'];frac=d['frac']
                reasons=math.eligibility(numbers,cell,physics.metadata['model_max_z'])
                if not reasons:
                    cart=frac@cell;dist=np.linalg.norm(find_mic((cart[:,None]-cart[None,:]).reshape(-1,3),cell,pbc=True)[0],axis=1).reshape(len(numbers),len(numbers))
                    np.fill_diagonal(dist,np.inf)
                    if float(dist.min())<.5:reasons.append('collision_below_0.5A')
                key=f'{i:02d}-{arm}'
                if reasons:point=dict(id=key,status='unknown',reasons=reasons,NFE=0,seconds=0.)
                else:
                    task=dict(id=key,action='relax',output=str(root/'relaxations'/key),numbers=numbers.tolist(),cell=cell.tolist(),frac=frac.tolist(),
                        fmax=.01,max_steps=1600,cell_relax=True)
                    point=physics.request([task])[0]
                row=dict(item,result=point,force_eligible=eligible(point));relaxations.append(row)
                save(root/'relaxation-progress.json',dict(completed=len(relaxations),total=72,
                    statuses=dict(Counter(r['result']['status'] for r in relaxations)),NFE=physics.NFE,elapsed_seconds=time.perf_counter()-started))
                save(root/'relaxations.json',relaxations)
                print(json.dumps(dict(stage='relaxation',trajectory=i,arm=arm,status=point['status'],steps=point.get('steps'),seconds=point['seconds'])),flush=True)
        save(root/'complete.json',dict(passed=True,inputs=records,relaxations=relaxations,sampling_audit=audits,
            native_forwards=forward_calls[0],physics=physics.metadata,NFE=physics.NFE,requests=physics.requests,
            elapsed_seconds=time.perf_counter()-started,note='Execution complete; physical convergence assessed separately for every endpoint.'))
    except Exception as e:
        save(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:
        if physics is not None:physics.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(args.source.resolve(),args.run.resolve())
