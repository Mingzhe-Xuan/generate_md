"""Random-prior, unconditional official MatterGen trajectories vs MatterSim forces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from observer import Observer
from trajectory_math import STEPS,eligibility,comparison,summarize

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'score_probe'))
from probe_models import isolated
from run_probe import save,sha

POLICY=dict(seed=20261050,trajectories=16,batch_size=8,sampling_steps=1000,
            snapshots=STEPS,prior_snapshot=True,conditions={},guidance=0.,
            checkpoint='dft_band_gap: unconditional branch of same checkpoint as previous probe',
            atom_count_prior='ALEX_MP_20',fixed_species=False,fixed_cell=False,
            mask_Z=101,minimum_separation_A=.5,minimum_volume_per_atom_A3=2.,
            minimum_cell_singular_value_A=.3,score_norm_floor=1e-10,force_norm_floor=1e-8,
            final_force_control_trajectories=[0,1,2,3],
            purpose='Sparse observational probe only; no force-guided sampling/relaxation/element backfill.')


def main(args):
    import torch
    from hydra.utils import instantiate
    from ase.geometry import find_mic
    root=args.run.resolve();root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    models=None
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        mapping=isolated(HERE.parent/'denoised_tto/mapping.py','random_probe_mapping')
        adapter=isolated(HERE.parent/'denoised_tto/adapter.py','random_probe_adapter',{'mapping':mapping})
        denoiser=adapter.Denoiser()
        from mattergen.generator import CrystalGenerator,seed_all
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.common.utils.globals import MAX_ATOMIC_NUM
        if MAX_ATOMIC_NUM+1!=POLICY['mask_Z']:
            raise ValueError('Mask token mismatch.')
        # Construct configuration/loaders only; the already verified module supplies weights.
        info=MatterGenCheckpointInfo(model_path=Path('results/mattergen/checkpoints/dft_band_gap').resolve(),load_epoch='last')
        generator=CrystalGenerator(checkpoint_info=info,batch_size=8,num_batches=2,
            properties_to_condition_on={},diffusion_guidance_factor=0.,record_trajectories=False,seed=POLICY['seed'])
        config=generator.sampling_config
        if config.sampler_partial.N!=1000 or config.sampler_partial.guidance_scale!=0:
            raise ValueError('Official schedule/unconditional branch mismatch.')
        seed_all(POLICY['seed'])
        loader=generator.get_condition_loader(config)
        sampler=instantiate(config.sampler_partial)(pl_module=SimpleNamespace(diffusion_module=denoiser.module,device=torch.device('cuda')))
        code={str(p):sha(p) for p in HERE.glob('*') if p.suffix in ('.py','.md','.sbatch')}
        for path in [HERE.parent/'score_probe/probe_models.py',HERE.parent/'score_probe/run_probe.py',
                     HERE.parent/'denoised_tto/adapter.py',HERE.parent/'denoised_tto/mapping.py',
                     HERE.parent/'directed_tto/runtime.py',HERE.parent/'directed_tto/mechanics.py']:
            code[str(path)]=sha(path)
        from omegaconf import OmegaConf
        save(root/'plan.json',dict(policy=POLICY,code_sha256=code,
            sampling_config=OmegaConf.to_container(config,resolve=True),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source=denoiser.spec))
        pending=[];finals=[];sampling_audit=[];raw_output=[];total_calls=[0]
        def hook(module,inputs,output):
            if 'dft_band_gap' in inputs[0]:
                raise ValueError('Unexpected property value in unconditional sampling input.')
            if not all(v.is_cuda for v in (output.pos,output.cell,output.atomic_numbers)):
                raise RuntimeError('CPU score output.')
            if not all(torch.isfinite(v).all() for v in (output.pos,output.cell,output.atomic_numbers)):
                raise ValueError('Nonfinite score output.')
            raw_output[:]=[output.pos]
            total_calls[0]+=1
        handle=denoiser.module.model.register_forward_hook(hook)
        try:
            for batch_index,(condition,mask) in enumerate(loader):
                if mask or 'dft_band_gap' in condition:
                    raise ValueError('Unexpected fixed variables/property conditions.')
                def capture(x,t,score,stage,step):
                    if sampler._guidance_scale!=0:
                        raise ValueError('Guidance changed.')
                    _,std=denoiser.module.corruption.sdes['pos'].marginal_prob(x.pos,t,x.get_batch_idx('pos'),x)
                    error=float((score.pos-raw_output[0]/std).abs().max())
                    if error>1e-5:
                        raise ValueError('Same-output score conversion failed.')
                    pieces=x.clone().to('cpu').to_data_list()
                    scores=score.pos.detach().cpu().double().numpy()
                    offset=0
                    for j,piece in enumerate(pieces):
                        n=int(piece.num_atoms.reshape(-1)[0]);trajectory=batch_index*8+j
                        name=f'traj-{trajectory:02d}-{stage}-{step:04d}.npz'
                        data=dict(numbers=piece.atomic_numbers.numpy(),cell=piece.cell[0].double().numpy(),
                                  frac=piece.pos.double().numpy(),score_fractional=scores[offset:offset+n])
                        path=root/name;np.savez_compressed(path,**data)
                        pending.append(dict(trajectory=trajectory,stage=stage,step=step,t=float(t[j]),
                            num_atoms=n,vector_path=name,vector_sha256=sha(path),score_conversion_error=error))
                        offset+=n
                    print(json.dumps(dict(batch=batch_index,stage=stage,step=step,t=float(t[0]),snapshots=len(pending))),flush=True)
                observer=Observer(sampler,STEPS,capture,torch)
                sampler._score_fn=observer
                try:
                    sample,mean=sampler.sample(condition,mask)
                finally:
                    sampler._score_fn=observer.original
                if observer.calls!=2000 or observer.snapshots!=len(STEPS)+1:
                    raise ValueError('Sampler call/snapshot coverage changed.')
                for j,piece in enumerate(mean.to('cpu').to_data_list()):
                    numbers=piece.atomic_numbers.numpy()
                    if np.any((numbers<1)|(numbers>MAX_ATOMIC_NUM)):
                        raise ValueError('Final sample still contains masked types.')
                    path=root/f'final-{batch_index*8+j:02d}.npz'
                    np.savez_compressed(path,numbers=numbers,cell=piece.cell[0].numpy(),frac=piece.pos.numpy())
                    finals.append(dict(trajectory=batch_index*8+j,path=path.name,sha256=sha(path),num_atoms=len(numbers)))
                sampling_audit.append(dict(batch=batch_index,calls=observer.calls,snapshots=observer.snapshots,
                    observer_rng_unchanged=True,conditioning_keys=list(condition.keys())))
                save(root/'sampling-progress.json',dict(policy=POLICY,states=pending,finals=finals,audit=sampling_audit))
        finally:
            handle.remove()
        if len(pending)!=16*(len(STEPS)+1) or len(finals)!=16 or total_calls[0]!=4000:
            raise ValueError('Incomplete random generation.')
        denoiser.metadata.update(conditioning={},guidance_scale=0.,fixed_species=False,
                                 fixed_cell=False,score_calls=total_calls[0])
        runtime=isolated(HERE.parent/'directed_tto/runtime.py','random_probe_physics')
        rpc=root/'rpc';rpc.mkdir();models=runtime.Mechanics(rpc)
        maximum=min(x['model_max_z'] for x in models.ready)
        tasks=[];rows=[];by_id={}
        for row in pending:
            with np.load(root/row['vector_path'],allow_pickle=False) as d:
                numbers=d['numbers'];cell=d['cell'];frac=d['frac'];score=d['score_fractional']
            cart=frac@cell
            reasons=eligibility(numbers,cell,maximum)
            row=dict(row,masked_atoms=int(np.sum(numbers==101)),unsupported_elements=sorted(set(int(z) for z in numbers if 94<z<101)),
                     status='unknown',reasons=reasons,cos_score_force=None,cos_drift_force=None)
            if not reasons:
                distances=np.linalg.norm(find_mic((cart[:,None]-cart[None,:]).reshape(-1,3),cell,pbc=True)[0],axis=1).reshape(len(numbers),len(numbers))
                np.fill_diagonal(distances,np.inf)
                minimum=float(distances.min()) if len(numbers)>1 else None
                row['minimum_pair_distance_A']=minimum
                if minimum is not None and minimum<.5:
                    row['reasons'].append('collision_below_0.5A')
            if not row['reasons']:
                key=f'traj-{row["trajectory"]}-{row["stage"]}-{row["step"]}'
                tasks.append(dict(id=key,action='point',numbers=numbers.tolist(),cell=cell.tolist(),frac=frac.tolist()))
                by_id[key]=(len(rows),score,cell)
            rows.append(row)
        # Evaluate original states only; no surrogate atom types or relaxation.
        physical=models.request(tasks)
        for point in physical:
            index,score,cell=by_id[point['candidate_id']]
            row=rows[index]
            row['physical']=point
            if point['status']=='completed':
                row.update(comparison(score,cell,point['forces_eV_A']))
                row['status']='compared' if row['cos_score_force'] is not None else 'stationary'
            else:
                row['status']='force_failed';row['reasons'].append(point.get('error','unknown failure'))
        controls=[]
        # Deterministic finite differences on the last sampled input, no added stochastic path.
        for row in rows:
            if row['step']!=999 or row['trajectory'] not in POLICY['final_force_control_trajectories']:
                continue
            if row['status'] not in ('compared','stationary'):
                controls.append(dict(trajectory=row['trajectory'],status='not_evaluable',reasons=row['reasons'],passed=None))
                continue
            with np.load(root/row['vector_path']) as d:
                cell=d['cell'];cart=d['frac']@cell;numbers=d['numbers']
            direction=np.arange(cart.size,dtype=float).reshape(cart.shape)**2
            direction-=direction.mean(0)
            norm=np.linalg.norm(direction)
            if norm<1e-10:
                controls.append(dict(trajectory=row['trajectory'],status='stationary_single_atom',passed=True));continue
            direction/=norm;expected=-float(np.sum(np.asarray(row['physical']['forces_eV_A'])*direction))
            for eps in [.002,.001]:
                test=[]
                for sign in [1,-1]:
                    test.append(dict(id=f'fd-{row["trajectory"]}-{eps}-{sign}',action='point',numbers=numbers.tolist(),
                        cell=cell.tolist(),frac=((cart+sign*eps*direction)@np.linalg.inv(cell)).tolist()))
                plus,minus=models.request(test)
                if plus['status']!='completed' or minus['status']!='completed':raise ValueError('Force finite-difference point failed.')
                finite=(plus['energy_eV']-minus['energy_eV'])/(2*eps)
                passed=abs(finite-expected)<=.005+.02*max(abs(finite),abs(expected))
                controls.append(dict(trajectory=row['trajectory'],epsilon=eps,FD=finite,expected=expected,passed=bool(passed)))
                if not passed:raise ValueError('Native force control failed: '+repr(controls[-1]))
        result=dict(passed=True,policy=POLICY,states=rows,finals=finals,sampling_audit=sampling_audit,
                    raw_model_calls=total_calls[0],force_controls=controls,mechanics=models.ready,NFE=models.NFE,
                    MatterGen=denoiser.metadata,device=torch.cuda.get_device_name(),torch=torch.__version__,
                    elapsed_seconds=time.perf_counter()-started,summary=summarize(rows),
                    note='Unconditional branch of same fine-tuned checkpoint; official full joint sampling, no known crystal input.')
        save(root/'complete.json',result)
        from trajectory_report import write_report
        write_report(root,result)
    except Exception as error:
        save(root/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        if models is not None:models.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('All model compute must run under Slurm.')
    main(args)
