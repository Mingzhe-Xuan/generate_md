"""Test finite descent directions at the archived200-update endpoint in Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import types

from diagnostics import normalized_direction

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Inference must run in Slurm.')
    import numpy as np
    import torch
    sys.path.insert(0,str(ROOT))
    from run_section21 import SOURCE_MEMBER,SOURCE_SHA256
    torch.set_num_threads(4)
    config=json.loads(args.config.read_text())
    if config['model_name']!='Crystalformer' or config['method']!='grad' or config['num_steps']!=200:
        raise ValueError('Require the original CPU206 fixed200-step protocol.')
    archive=Path('results/smoacs/downloads/reproducibility.tar.gz')
    manifest=json.loads((ROOT/'assets.json').read_text())
    if digest(archive)!=manifest['reproducibility.tar.gz']['sha256']:
        raise ValueError('Unverified author archive.')
    with tarfile.open(archive) as tar:original=tar.extractfile(SOURCE_MEMBER).read()
    if hashlib.sha256(original).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Author source mismatch.')
    weights=json.loads((ROOT/'checkpoints.json').read_text())
    for name,metadata in weights.items():
        if digest(args.assets/'models/crystalformer'/name)!=metadata['sha256']:
            raise ValueError('Predictor weights changed.')
    os.environ['SMOACS_ROOT']=str(args.assets.resolve())
    os.environ['MPLBACKEND']='Agg'
    source=Path('third_party/SMOACS').resolve()
    sys.path[:0]=[str(source),str(source/'src')]
    module=types.ModuleType('cpu206_stationarity_author')
    module.__file__=str(args.assets.resolve()/SOURCE_MEMBER)
    sys.modules[module.__name__]=module
    exec(compile(original,module.__file__,'exec',dont_inherit=True),module.__dict__)
    gap_model,ef_model=module.Load_Pretrained_Crystalformers(device='cpu')
    for model in (gap_model,ef_model):
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
    with np.load(args.npz,allow_pickle=False) as z:
        arrays={name:z[name].copy() for name in z.files}
    ids=arrays['original_fnames'].tolist()
    if len(ids)!=128 or len(set(ids))!=128 or set(ids)!=set(config['template_ids']):
        raise ValueError('Endpoint candidate IDs changed.')
    to_tensor=lambda name:torch.tensor(arrays[name])
    size=to_tensor('size').long()
    batch=torch.repeat_interleave(torch.arange(len(size)),size)
    fixed=dict(size=size,batch=batch,atomic_mask=to_tensor('atomic_mask'),
               ox_states_used_mask=to_tensor('ox_states_used_mask'),init_coords=to_tensor('init_coords'))
    scalers=dict(abc_scaler=module.ABC_Scaler(to_tensor('init_abc'),*config['abc_range'],device='cpu'),
                 angle_scaler=module.Angle_Scaler(*config['angle_range']))
    initial=[to_tensor(name).float() for name in ('scaled_batch_abc','scaled_batch_angle','dir_coords',
                       'atomic_distribution','ox_mask_learnable_tensor_per_crystal')]
    def forward(values,temperature):
        gap,ef,out=module.forward_propagation_Crystalformer(optimization_targets=values,fixed_inputs=fixed,
            scalers=scalers,temperature=temperature,bandgap_model=gap_model,e_form_model=ef_model,
            adding_noise_scale=None,limit_coords_displacement=None,
            abc_range=config['abc_range'],angle_range=config['angle_range'])
        gap,ef=gap.reshape(-1),ef.reshape(-1)
        loss=torch.clamp(torch.abs(gap-2.5)-.04,min=0)+ef
        if not all(torch.isfinite(v).all() for v in (gap,ef,loss)):
            raise ValueError('Nonfinite prediction/objective.')
        return gap,ef,loss,out
    args.output.mkdir(parents=True,exist_ok=False)
    with torch.no_grad():gap,ef,loss,out=forward(initial,1e-8)
    control=max(float(np.max(np.abs(gap.numpy()-arrays['bandgap_dist']))),
                float(np.max(np.abs(ef.numpy()-arrays['ef_dist']))))
    if control>1e-3:raise ValueError('Restored endpoint differs from archive by >1e-3.')
    report=dict(job_id=os.environ['SLURM_JOB_ID'],npz_sha256=digest(args.npz),source_sha256=SOURCE_SHA256,
                checkpoint_manifest=weights,device='cpu',candidate_count=128,
                restored_endpoint_max_control_error=control,
                objective='max(abs(gap-2.5)-.04,0)+Ef',
                method='Geometry-only normalized negative-gradient trials at fixed temperature. Atomic/oxidation parameters fixed. No Adam resume; each trial starts from the immutable same endpoint.',
                geometry_variables=['scaled length','scaled angle','fractional coordinates'],probes=[])
    for temperature in (.0001495,1e-8):
        values=[v.clone().requires_grad_(i<3) for i,v in enumerate(initial)]
        gap,ef,baseline,out=forward(values,temperature)
        gradients=torch.autograd.grad(baseline.sum(),values[:3],allow_unused=True)
        gradients=[g if g is not None else torch.zeros_like(v) for g,v in zip(gradients,values[:3])]
        if not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite gradient.')
        rates=[config['lattice_lr'],config['lattice_lr'],config['coords_lr']]
        directions=[torch.from_numpy(normalized_direction(g.detach().numpy(),rate)) for g,rate in zip(gradients,rates)]
        grad_sq=gradients[0].square().sum(dim=1)+gradients[1].square().sum(dim=1)
        grad_sq=grad_sq.index_add(0,batch,gradients[2].square().sum(dim=1))
        norm=torch.sqrt(grad_sq).detach().numpy()
        base=baseline.detach().numpy()
        composition=out['normalized_dist'].detach()
        trials=[]
        per_candidate=[dict(candidate_id=k,baseline_objective=float(base[i]),geometry_gradient_l2=float(norm[i])) for i,k in enumerate(ids)]
        delta_vectors=[]
        for multiplier in (.001,.01,.1,1.):
            trial=[v.clone() for v in initial]
            for i in range(3):trial[i]=trial[i]+multiplier*directions[i]
            with torch.no_grad():g,e,result,trial_out=forward(trial,temperature)
            if not torch.equal(trial_out['normalized_dist'],composition):
                raise ValueError('Geometry trial changed elemental distribution.')
            delta=result.numpy()-base
            delta_vectors.append(delta)
            trials.append(dict(multiplier=multiplier,mean_delta=float(delta.mean()),
                improved_gt_1e_3=int(np.sum(delta < -.001)),worsened_gt_1e_3=int(np.sum(delta > .001))))
            for i,row in enumerate(per_candidate):row['delta_at_'+str(multiplier)]=float(delta[i])
        best=np.min(np.stack(delta_vectors),axis=0)
        for i,row in enumerate(per_candidate):row['best_trial_delta']=float(best[i])
        probe=dict(temperature=temperature,gradient_nonzero_count=int(np.sum(norm>1e-6)),
            gradient_norm_median=float(np.median(norm)),gradient_norm_max=float(np.max(norm)),
            any_trial_improved_gt_1e_3=int(np.sum(best < -.001)),
            any_trial_improved_gt_1e_2=int(np.sum(best < -.01)),trials=trials,candidates=per_candidate)
        report['probes'].append(probe)
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in probe.items() if k!='candidates'}),flush=True)
    print('STATIONARITY_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
