"""Dense RNG-neutral observations of native random-prior pure-CSP sampling."""
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

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'forward_inverse'))
from csp_model import CSP,load_file,sha
from physics_client import Physics
observer_module=load_file(HERE.parent/'random_score_probe/observer.py','csp_dense_observer')
math_module=load_file(HERE.parent/'random_score_probe/trajectory_math.py','csp_dense_math')


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def signature(numbers):
    return tuple(sorted(Counter(map(int,numbers)).items()))


def main(root):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from ase.geometry import find_mic
    if (root/'plan.json').exists():
        raise ValueError('Refuse existing sampling run.')
    started=time.perf_counter();physics=None
    try:
        plan=json.loads((root/'composition_plan.json').read_text())
        model=CSP();del model.double
        from mattergen.generator import CrystalGenerator,seed_all
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        validation=load_file(HERE.parent/'csp/validation.py','dense_validation')
        compositions=[validation.validate_composition(r['counts']) for r in plan['compositions']]
        mapping={tuple(sorted((int(z),n) for z,n in r['atomic_counts'].items())):r['system'] for r in plan['compositions']}
        info=MatterGenCheckpointInfo(model_path=Path('results/mattergen/csp-assets').resolve(),load_epoch='last')
        generator=CrystalGenerator(checkpoint_info=info,batch_size=8,num_batches=3,
            target_compositions_dict=compositions,sampling_config_name='csp',properties_to_condition_on={},
            diffusion_guidance_factor=0.,record_trajectories=False,seed=plan['seed'])
        config=generator.sampling_config
        validation.validate_sampling(OmegaConf.to_container(config,resolve=True))
        seed_all(plan['seed'])
        batches=list(generator.get_condition_loader(config,compositions))
        counts=Counter();layout=[]
        for b,(condition,mask) in enumerate(batches):
            if mask or 'dft_band_gap' in condition:
                raise ValueError('Unexpected mask/property condition.')
            pieces=condition.to_data_list()
            if len(pieces)!=8:raise ValueError('Batch size changed.')
            systems=[mapping[signature(p.atomic_numbers.cpu().numpy())] for p in pieces]
            counts.update(systems);layout.append(systems)
        if len(batches)!=3 or counts!={r['system']:2 for r in plan['compositions']}:
            raise ValueError('Composition/path coverage mismatch: '+repr(counts))
        plan.update(sampling_config=OmegaConf.to_container(config,resolve=True),layout=layout,
            checkpoint=model.metadata,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            code_sha256={str(p):sha(p) for folder in (HERE,HERE.parent/'forward_inverse',HERE.parent/'random_score_probe',HERE.parent/'csp')
                for p in folder.glob('*.py')},
            geometry_guards=dict(min_pair_A=.5,min_volume_per_atom_A3=2.,min_cell_singular_A=.3),
            statistical_unit='Three native coupled batches of eight paths; descriptive statistics, no independent-path CI.')
        save(root/'plan.json',plan)
        sampler=instantiate(config.sampler_partial)(pl_module=SimpleNamespace(diffusion_module=model.native,device=torch.device('cuda')))
        raw=[];calls=[0];conversion_errors=[];audit=[];paths=[];expected=[None]
        def hook(module,inputs,output):
            x=inputs[0]
            if not torch.equal(x.atomic_numbers,expected[0]) or 'dft_band_gap' in x:
                raise ValueError('Species/condition changed at native model input.')
            if not all(v.is_cuda and torch.isfinite(v).all() for v in (output.pos,output.cell)):
                raise ValueError('Native output not finite CUDA.')
            raw[:]=[output.pos];calls[0]+=1
        handle=model.native.model.register_forward_hook(hook)
        try:
            for b,(condition,mask) in enumerate(batches):
                seed=plan['seed']+b;seed_all(seed)
                expected[0]=condition.atomic_numbers.clone().to('cuda')
                buffers=[dict(numbers=p.atomic_numbers.cpu().numpy(),cell=[],frac=[],score_fractional=[],t=[],step=[],stage=[])
                         for p in condition.to_data_list()]
                original=sampler._score_fn
                def checked(x,t):
                    if not torch.equal(x.atomic_numbers,expected[0]):raise ValueError('Species changed in PC input.')
                    if not all(v.is_cuda and torch.isfinite(v).all() for v in (x.pos,x.cell)):
                        raise ValueError('Nonfinite/CPU PC input.')
                    return original(x,t)
                sampler._score_fn=checked
                def capture(x,t,score,stage,step):
                    _,std=model.native.corruption.sdes['pos'].marginal_prob(x.pos,t,x.get_batch_idx('pos'),x)
                    error=float((score.pos-raw[0]/std).abs().max());conversion_errors.append(error)
                    if error>1e-5:raise ValueError('Same-forward native score conversion failed.')
                    offset=0
                    for j,piece in enumerate(x.clone().cpu().to_data_list()):
                        n=len(piece.atomic_numbers);buffer=buffers[j]
                        buffer['cell'].append(piece.cell[0].double().numpy())
                        buffer['frac'].append(piece.pos.double().numpy())
                        buffer['score_fractional'].append(score.pos[offset:offset+n].detach().cpu().double().numpy())
                        buffer['t'].append(float(t[j]));buffer['step'].append(step);buffer['stage'].append(stage)
                        offset+=n
                observer=observer_module.Observer(sampler,range(1000),capture,torch)
                sampler._score_fn=observer
                batch_started=time.perf_counter()
                try:
                    sample,mean=sampler.sample(condition,mask)
                finally:
                    sampler._score_fn=original
                if observer.calls!=2000 or observer.snapshots!=1001:
                    raise ValueError('Dense observation coverage changed.')
                for final in (sample,mean):
                    if not torch.equal(final.atomic_numbers,expected[0]) or not all(torch.isfinite(v).all() for v in (final.pos,final.cell)):
                        raise ValueError('Final species/finite audit failed.')
                for j,buffer in enumerate(buffers):
                    trajectory=b*8+j;name=f'trajectory-{trajectory:02d}.npz'
                    np.savez_compressed(root/name,**{k:np.asarray(v) for k,v in buffer.items()})
                    terminal=mean.clone().cpu().to_data_list()[j]
                    np.savez_compressed(root/f'terminal-{trajectory:02d}.npz',numbers=terminal.atomic_numbers.numpy(),
                        cell=terminal.cell[0].numpy(),frac=terminal.pos.numpy())
                    paths.append(dict(trajectory=trajectory,batch=b,seed=seed,system=layout[b][j],path=name,sha256=sha(root/name),
                        terminal_path=f'terminal-{trajectory:02d}.npz',terminal_sha256=sha(root/f'terminal-{trajectory:02d}.npz')))
                audit.append(dict(batch=b,seed=seed,score_calls=observer.calls,snapshots=observer.snapshots,
                    observer_rng_unchanged=True,ordered_species_preserved=True,seconds=time.perf_counter()-batch_started))
                save(root/'sampling-progress.json',dict(paths=paths,audit=audit,calls=calls[0]))
                print(json.dumps(dict(stage='sampling',batch=b,paths=len(paths),seconds=audit[-1]['seconds'])),flush=True)
        finally:
            handle.remove()
        if calls[0]!=6000 or len(paths)!=24:raise ValueError('Incomplete sampling.')
        physics=Physics(root/'physics');rows=[];tasks=[];pending=[]
        def flush():
            if not tasks:return
            result=physics.request(tasks)
            for point,item in zip(result,pending):
                row,score,cell=item
                if point['status']=='completed':
                    row.update(math_module.comparison(score,cell,point['forces_eV_A']))
                    row.update(status='compared' if row['cos_score_force'] is not None else 'stationary',
                        energy_eV=point['energy_eV'],max_raw_force=point['max_raw_force'])
                else:
                    row.update(status='force_failed');row['reasons'].append(point.get('error','force failed'))
            tasks.clear();pending.clear()
        for path in paths:
            with np.load(root/path['path'],allow_pickle=False) as d:
                numbers=d['numbers']
                for k in range(1001):
                    cell=d['cell'][k];frac=d['frac'][k];score=d['score_fractional'][k]
                    row=dict(trajectory=path['trajectory'],batch=path['batch'],system=path['system'],frame=k,
                        stage=str(d['stage'][k]),step=int(d['step'][k]),t=float(d['t'][k]),
                        id=f'{path["trajectory"]}:{k}',status='unknown',reasons=math_module.eligibility(numbers,cell,physics.metadata['model_max_z']),
                        cos_score_force=None,cos_drift_force=None)
                    if not row['reasons']:
                        cart=frac@cell
                        distance=np.linalg.norm(find_mic((cart[:,None]-cart[None,:]).reshape(-1,3),cell,pbc=True)[0],axis=1).reshape(len(numbers),len(numbers))
                        np.fill_diagonal(distance,np.inf);minimum=float(distance.min())
                        row['minimum_pair_distance_A']=minimum
                        if minimum<.5:row['reasons'].append('collision_below_0.5A')
                    if not row['reasons']:
                        tasks.append(dict(id=row['id'],action='point',numbers=numbers.tolist(),cell=cell.tolist(),frac=frac.tolist()))
                        pending.append((row,score,cell))
                    rows.append(row)
                    if len(tasks)>=128:flush()
            flush()
            save(root/'force-progress.json',dict(paths_done=path['trajectory']+1,states=len(rows),requests=physics.requests,NFE=physics.NFE,
                statuses=dict(Counter(r['status'] for r in rows)),elapsed_seconds=time.perf_counter()-started))
            print((root/'force-progress.json').read_text(),flush=True)
        if len(rows)!=24024:raise ValueError('Incomplete state coverage.')
        save(root/'complete.json',dict(passed=True,paths=paths,states=rows,sampling_audit=audit,
            native_forward_calls=calls[0],max_same_forward_conversion_error=max(conversion_errors),
            physics=physics.metadata,requests=physics.requests,NFE=physics.NFE,
            elapsed_seconds=time.perf_counter()-started,device=torch.cuda.get_device_name(),torch=torch.__version__))
    except Exception as e:
        save(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
    finally:
        if physics is not None:physics.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(args.run.resolve())
