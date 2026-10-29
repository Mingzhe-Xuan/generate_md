"""Paired unconditional sampling +/- available-force projection, then common B."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from projection import force_supported,remove_parallel

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'score_probe'))
from probe_models import isolated
from run_probe import save,sha

POLICY=dict(seed=20261050,trajectories=16,batch_size=8,sampling_steps=1000,conditions={},guidance=0.,
    arms=['baseline','orthogonal'],gate='All real supported atom types and successful MatterSim force; NO cosine threshold.',
    projection='s_new=s-(s.F)/(F.F)*F; s is centered Cartesian score-driven displacement; preserve original score translation.',
    calls='Both corrector and predictor positional score; cell/type scores unchanged.',
    renormalize=False,corrector='Official adaptive step/noise rule retained; reduced scores may increase corrector noise amplitude.',
    relaxation='Unchanged canonical MatterSim5M BFGS/ExpCellFilter/FixSymmetry fmax.01/maxstep.05/1600steps.',
    SUN='Common full chemical-subsystem MP2026-04-13 hull<=.3 references, MP2020 corrections, S<.1eV/atom; U separately within each16.',
    DFT=False,legacy_source='results/mattergen/random-score-probe-20261029-seed50',
    legacy_replay='Diagnostic numerical agreement only; fresh paired baseline is primary. GPU scatter/graph may be nondeterministic.')


def rng_hash(torch):
    return hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()


def execute(args):
    import torch
    from hydra.utils import instantiate
    from ase import Atoms
    from ase.io import write
    from pymatgen.core import Structure
    root=args.run.resolve();root.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();mechanics=None
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        mapping=isolated(HERE.parent/'denoised_tto/mapping.py','orthogonal_mapping')
        adapter=isolated(HERE.parent/'denoised_tto/adapter.py','orthogonal_adapter',{'mapping':mapping})
        denoiser=adapter.Denoiser()
        from mattergen.generator import CrystalGenerator,seed_all
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from omegaconf import OmegaConf
        runtime=isolated(HERE.parent/'directed_tto/runtime.py','orthogonal_mechanics')
        rpc=root/'rpc';rpc.mkdir();mechanics=runtime.Mechanics(rpc)
        maximum=min(x['model_max_z'] for x in mechanics.ready)
        code={str(p):sha(p) for p in HERE.glob('*') if p.suffix in ('.py','.md','.sbatch')}
        for relative in ['score_probe/probe_models.py','score_probe/run_probe.py','denoised_tto/adapter.py','denoised_tto/mapping.py',
                         'directed_tto/runtime.py','directed_tto/mechanics.py']:
            path=HERE.parent/relative;code[str(path)]=sha(path)
        legacy=Path(POLICY['legacy_source']);old=json.loads((legacy/'complete.json').read_text())
        if not old['passed'] or old['policy']['seed']!=POLICY['seed']:raise ValueError('Wrong legacy random probe.')
        save(root/'plan.json',dict(policy=POLICY,code_sha256=code,source=denoiser.spec,
            legacy_complete_sha256=sha(legacy/'complete.json'),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        all_inputs={};all_finals={};sampling={};legacy_checks=[];model_calls=[0]
        def audit(module,inputs,out):
            if 'dft_band_gap' in inputs[0]:raise ValueError('Unexpected property conditioning.')
            if not all(v.is_cuda for v in (out.pos,out.cell,out.atomic_numbers)):raise RuntimeError('CPU score output.')
            if not all(torch.isfinite(v).all() for v in (out.pos,out.cell)):raise ValueError('Nonfinite score.')
            model_calls[0]+=1
        handle=denoiser.module.model.register_forward_hook(audit)
        try:
            for arm in POLICY['arms']:
                folder=root/arm;folder.mkdir();(folder/'structures').mkdir();(folder/'vectors').mkdir()
                info=MatterGenCheckpointInfo(model_path=Path('results/mattergen/checkpoints/dft_band_gap').resolve(),load_epoch='last')
                generator=CrystalGenerator(checkpoint_info=info,batch_size=8,num_batches=2,properties_to_condition_on={},diffusion_guidance_factor=0.)
                config=generator.sampling_config
                if config.sampler_partial.N!=1000 or config.sampler_partial.guidance_scale!=0:raise ValueError('Wrong sampler.')
                seed_all(POLICY['seed']);loader=generator.get_condition_loader(config)
                sampler=instantiate(config.sampler_partial)(pl_module=SimpleNamespace(diffusion_module=denoiser.module,device=torch.device('cuda')))
                inventory=[];finals=[];calls=0;projected=0;force_evals=0;batch_audits=[];force_before=mechanics.NFE
                with (folder/'interventions.jsonl').open('w') as log:
                    for batch_index,(condition,mask) in enumerate(loader):
                        if mask or 'dft_band_gap' in condition:raise ValueError('Unexpected sampler constraints.')
                        original=sampler._score_fn;batch_calls=[0];rng=[];priors=[]
                        def score_fn(x,t):
                            nonlocal calls,projected,force_evals
                            score=original(x,t);call=batch_calls[0];step,within=divmod(call,2)
                            batch_calls[0]+=1;calls+=1
                            if within==0 and step in [0,100,250,500,750,900,950,999]:rng.append(dict(step=step,hash=rng_hash(torch)))
                            if call==0:
                                for j,piece in enumerate(x.clone().to('cpu').to_data_list()):
                                    path=folder/f'prior-{batch_index*8+j:02d}.npz'
                                    np.savez_compressed(path,numbers=piece.atomic_numbers.numpy(),cell=piece.cell.numpy(),frac=piece.pos.numpy())
                                    priors.append(dict(trajectory=batch_index*8+j,path=str(path),sha256=sha(path)))
                            if arm=='baseline':return score
                            before_rng=rng_hash(torch)
                            pieces=x.clone().to('cpu').to_data_list();sf=score.pos.detach().cpu().double().numpy()
                            modified=sf.copy();F=np.zeros_like(sf);flags=[];tasks=[];offset=0;parts=[]
                            for j,piece in enumerate(pieces):
                                z=piece.atomic_numbers.numpy();cell=piece.cell[0].double().numpy();frac=piece.pos.double().numpy();n=len(z)
                                reason=force_supported(z,cell,frac,maximum)
                                row=dict(trajectory=batch_index*8+j,reason=reason,triggered=False)
                                flags.append(row);parts.append((offset,offset+n,cell));offset+=n
                                if reason is None:
                                    tasks.append(dict(id=f'call-{calls}-traj-{j}',action='point',numbers=z.tolist(),cell=cell.tolist(),frac=frac.tolist()))
                            results=mechanics.request(tasks) if tasks else []
                            for task,point in zip(tasks,results):
                                j=int(task['id'].rsplit('-',1)[1]);lo,hi,cell=parts[j];force_evals+=1
                                flags[j]['point_status']=point['status']
                                if point['status']=='completed':
                                    force=np.asarray(point['forces_eV_A']);F[lo:hi]=force
                                    changed,meta=remove_parallel(sf[lo:hi],cell,force)
                                    flags[j].update(meta)
                                    if meta['triggered']:modified[lo:hi]=changed;projected+=1
                                else:flags[j]['reason']='force_prediction_failed: '+point.get('error','unknown')
                            out=score.replace(pos=torch.as_tensor(modified,dtype=score.pos.dtype,device=score.pos.device)) if any(f['triggered'] for f in flags) else score
                            if not torch.equal(score.cell,out.cell) or not torch.equal(score.atomic_numbers,out.atomic_numbers):raise ValueError('Changed non-positional scores.')
                            record=dict(batch=batch_index,call=call,step=step,phase='corrector' if within==0 else 'predictor',t=float(t[0]),states=flags)
                            if within==0:
                                before_mean=float(np.mean([np.linalg.norm(sf[lo:hi]) for lo,hi,_ in parts]))
                                after_mean=float(np.mean([np.linalg.norm(modified[lo:hi]) for lo,hi,_ in parts]))
                                record['fractional_corrector_norms']=dict(before_mean=before_mean,after_mean=after_mean,
                                    unclipped_step_multiplier=(before_mean/after_mean)**2 if after_mean>1e-30 else None,
                                    official_step_cap=1e6)
                            if tasks:
                                path=folder/'vectors'/f'b{batch_index}-call{call:04d}.npz'
                                np.savez_compressed(path,numbers=x.atomic_numbers.cpu().numpy(),cell=x.cell.cpu().double().numpy(),
                                    frac=x.pos.cpu().double().numpy(),counts=x.num_atoms.cpu().numpy(),before=sf,
                                    after=out.pos.cpu().double().numpy(),forces=F)
                                record.update(vector_path=str(path.relative_to(root)),vector_sha256=sha(path))
                            if rng_hash(torch)!=before_rng:raise ValueError('Force projection changed parent RNG state.')
                            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                            if within==1 and step%100==0:print(json.dumps(dict(arm=arm,batch=batch_index,step=step,projected=projected,NFE=mechanics.NFE)),flush=True)
                            return out
                        sampler._score_fn=score_fn
                        try:_,mean=sampler.sample(condition,mask)
                        finally:sampler._score_fn=original
                        if batch_calls[0]!=2000:raise ValueError('Wrong score call count.')
                        batch_audits.append(dict(batch=batch_index,calls=batch_calls[0],rng_checkpoints=rng,prior=priors))
                        for j,piece in enumerate(mean.cpu().to_data_list()):
                            index=batch_index*8+j;z=piece.atomic_numbers.numpy();cell=piece.cell[0].numpy();frac=piece.pos.numpy()
                            if np.any((z<1)|(z>100)):raise ValueError('Masked final structure.')
                            name=f'{arm}-{index:04d}.vasp';path=folder/'structures'/name
                            write(path,Atoms(numbers=z,cell=cell,scaled_positions=frac,pbc=True),format='vasp',direct=True,sort=False)
                            struct=Structure(cell,z,frac)
                            inventory.append(dict(candidate_id=name,source=str(path),sha256=sha(path),num_atoms=len(z),formula=struct.composition.formula,
                                                  elements=sorted(e.symbol for e in struct.composition.elements),trajectory=index))
                            finals.append(dict(trajectory=index,numbers=z.tolist(),cell=cell.tolist(),frac=frac.tolist()))
                            if arm=='baseline':
                                previous=old['finals'][index];old_path=legacy/previous['path']
                                if sha(old_path)!=previous['sha256']:raise ValueError('Legacy endpoint changed.')
                                with np.load(old_path) as d:
                                    same_types=bool(np.array_equal(z,d['numbers']))
                                    legacy_checks.append(dict(trajectory=index,same_types=same_types,
                                        max_cell_error=float(np.abs(cell-d['cell']).max()),
                                        max_fractional_error=float(np.abs(frac-d['frac']).max()) if same_types else None))
                        save(folder/'sampling-progress.json',dict(audit=batch_audits,finals=finals,projected=projected,NFE=mechanics.NFE-force_before))
                if calls!=4000 or len(inventory)!=16:raise ValueError('Incomplete generation.')
                all_inputs[arm]=inventory;all_finals[arm]=finals
                sampling[arm]=dict(audit=batch_audits,score_calls=calls,projected=projected,force_evaluations=force_evals,NFE=mechanics.NFE-force_before,
                                   sampling_config=OmegaConf.to_container(config,resolve=True))
                save(folder/'sampling-complete.json',sampling[arm])
                (folder/'inventory-249').mkdir()
                save(folder/'inventory-249/inventory.json',dict(role=arm,denominator=16,candidates=inventory))
                print(json.dumps(dict(arm=arm,completed=True,projected=projected,NFE=sampling[arm]['NFE'])),flush=True)
        finally:handle.remove()
        for a,b in zip(sampling['baseline']['audit'],sampling['orthogonal']['audit']):
            if a['rng_checkpoints']!=b['rng_checkpoints']:raise ValueError('Paired arms did not use identical RNG sequences.')
            for p,q in zip(a['prior'],b['prior']):
                with np.load(p['path']) as x,np.load(q['path']) as y:
                    if not all(np.array_equal(x[k],y[k]) for k in x.files):raise ValueError('Paired priors differ.')
        save(root/'generation-complete.json',dict(passed=True,policy=POLICY,sampling=sampling,finals=all_finals,legacy_checks=legacy_checks,
            source=denoiser.spec,model_calls=model_calls[0],NFE_projection=mechanics.NFE,seconds=time.perf_counter()-start))
        b_before=mechanics.NFE;B={}
        for arm in POLICY['arms']:
            tasks=[dict(id=c['candidate_id'],action='relax',source=c['source'],output=str(root/arm/'relaxation'/Path(c['candidate_id']).stem)) for c in all_inputs[arm]]
            B[arm]=mechanics.request(tasks)
            save(root/arm/'relaxation-results.json',B[arm])
        combined=root/'combined';(combined/'inventory-249').mkdir(parents=True);(combined/'relaxation').mkdir()
        save(combined/'inventory-249/inventory.json',dict(role='Both paired arms common reference scope',denominator=32,candidates=sum(all_inputs.values(),[])))
        for arm in POLICY['arms']:
            for c in all_inputs[arm]:
                shutil.copytree(root/arm/'relaxation'/Path(c['candidate_id']).stem,combined/'relaxation'/Path(c['candidate_id']).stem)
        save(root/'physics-complete.json',dict(passed=True,policy=POLICY,B=B,sampling=sampling,NFE_total=mechanics.NFE,
            NFE_B=mechanics.NFE-b_before,mechanics=mechanics.ready,seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),torch=torch.__version__))
    except Exception as error:
        save(root/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        if mechanics is not None:mechanics.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm for all compute.')
    execute(args)
