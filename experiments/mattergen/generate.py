"""Official conditional MatterGen sampling with audited CUDA score calls."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from fetch import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=128);parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    if args.count<1 or args.count%args.batch_size:parser.error('Count must be a positive multiple of batch size.')
    import torch
    from torch_scatter import scatter
    from torch_sparse import SparseTensor
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for MatterGen.')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    probe=torch.tensor([1.,2.,3.],device='cuda');index=torch.tensor([0,1,0],device='cuda')
    torch.testing.assert_close(scatter(probe,index),torch.tensor([4.,2.],device='cuda'))
    sparse=SparseTensor(row=index,col=torch.arange(3,device='cuda'),value=probe,sparse_sizes=(2,3))
    if not sparse.sum(1).is_cuda:raise RuntimeError('Sparse extension fell back to CPU.')
    assets=Path('results/mattergen').resolve();spec=json.loads(Path(__file__).with_name('sources.json').read_text())
    if not json.loads((assets/'setup-complete.json').read_text())['source']==spec:raise ValueError('Setup provenance changed.')
    model_path=assets/'checkpoints/dft_band_gap';checkpoint=model_path/'checkpoints/last.ckpt'
    if digest(checkpoint)!=spec['checkpoint_sha256'] or digest(model_path/'config.yaml')!=spec['config_sha256']:raise ValueError('Unverified official checkpoint/config.')
    sys.path.insert(0,str(assets/'source'))
    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
    from mattergen.generator import CrystalGenerator
    from mattergen.diffusion import lightning_module
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'smoacs'))
    from alignn_checkpoint import TrustedCheckpointTorch
    info=MatterGenCheckpointInfo(model_path=model_path,load_epoch='last',strict_checkpoint_loading=True,
        config_overrides=["++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"])
    generator=CrystalGenerator(checkpoint_info=info,batch_size=args.batch_size,num_batches=args.count//args.batch_size,
        properties_to_condition_on={'dft_band_gap':2.5},diffusion_guidance_factor=2.,record_trajectories=False,seed=42)
    if generator.sampling_config.sampler_partial.N!=1000:raise ValueError('Official1000-step schedule changed.')
    args.output.mkdir(parents=True,exist_ok=False);audit=dict(score_calls=0,all_cuda=True,finite_positions_and_cells=True)
    def observe(module,inputs,output):
        tensors=[output.pos,output.cell,output.atomic_numbers]
        if not all(t.is_cuda for t in tensors):audit['all_cuda']=False;raise RuntimeError('MatterGen score output on CPU.')
        if not all(torch.isfinite(t).all() for t in tensors[:2]):audit['finite_positions_and_cells']=False;raise ValueError('Nonfinite score geometry.')
        audit['score_calls']+=1
    original_prepare=generator.prepare;handles=[]
    def prepare_with_audit():
        old=lightning_module.torch;lightning_module.torch=TrustedCheckpointTorch(torch,[spec['checkpoint_sha256']])
        try:original_prepare()
        finally:lightning_module.torch=old
        model=generator._model
        if not all(p.is_cuda for p in model.parameters()):raise RuntimeError('MatterGen model is not entirely on CUDA.')
        if 'dft_band_gap' not in model.diffusion_module.model.cond_fields_model_was_trained_on:raise ValueError('Checkpoint lacks gap conditioning.')
        if not handles:handles.append(model.diffusion_module.model.register_forward_hook(observe))
    generator.prepare=prepare_with_audit;started=time.perf_counter()
    try:structures=generator.generate(output_dir=args.output)
    finally:
        for handle in handles:handle.remove()
        (args.output/'gpu-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if len(structures)!=args.count or audit['score_calls']<1000 or not audit['all_cuda']:raise ValueError('Generation count or CUDA audit failed.')
    (args.output/'structures.json').write_text(json.dumps([s.as_dict() for s in structures])+'\n')
    (args.output/'complete.json').write_text(json.dumps(dict(completed=True,count=len(structures),device=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,seconds=time.perf_counter()-started,source=spec,
        conditioning={'dft_band_gap':2.5},guidance=2.,sampling_steps=1000,batch_size=args.batch_size,seed=42,
        interpretation='Official pretrained checkpoint inference reproduction; formation energy evaluated independently; no retraining.'),indent=2)+'\n')


if __name__=='__main__':main()
