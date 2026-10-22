"""Run the pinned upstream ALIGNN CLI with observational CUDA input audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit model computation through Slurm.')
    source, assets, weights = [p.resolve() for p in (args.source, args.assets, args.weights)]
    import torch
    import dgl
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required.')
    output = Path('results/smoacs') / ('alignn-smoke-' + os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    report = dict(passed=False, torch=torch.__version__, dgl=dgl.__version__,
                  gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability()),
                  forward_calls=0, output_backwards=0,
                  input_backwards=dict(atom_features=0, bondlength=0, angle_features=0),
                  input_gradient_max=dict(atom_features=0., bondlength=0., angle_features=0.),
                  checkpoints={})
    reference = Path(os.environ['ATOMGPTLAB_CACHE'])/'jarvis_data/megnet.json.zip'
    if reference.stat().st_size != 562093824:
        raise RuntimeError('Prepare the complete official MEGNet reference before submitting.')
    with zipfile.ZipFile(reference) as archive:
        if archive.testzip() is not None or 'megnet.json' not in archive.namelist():
            raise RuntimeError('Invalid MEGNet reference ZIP.')
    digest = hashlib.sha256()
    with reference.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    report['megnet_reference'] = dict(bytes=reference.stat().st_size, sha256=digest.hexdigest(),
                                    source='https://ndownloader.figshare.com/files/26724977')
    expected = {'mp_gappbe_alignnn': '8e9309fcb435a88481ccbc945d16cb1343f9ac24d8198607f753ce59b687242e',
                'mp_e_form_alignnn': '848fb8f5844445ab2c751a7a7790ad69d7d3c6c385eb645758c48049e2bd8e71'}
    checkpoint_digests = []
    for name, digest in expected.items():
        actual = hashlib.sha256((weights / (name+'.zip')).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError('Checkpoint hash mismatch: ' + name)
        report['checkpoints'][name] = actual
        with zipfile.ZipFile(weights / (name+'.zip')) as archive:
            for member in archive.namelist():
                if ('checkpoint_' in member and 'pt' in member) or 'best_model.pt' in member:
                    checkpoint_digests.append(hashlib.sha256(archive.read(member)).hexdigest())
    sys.path.insert(0, str(source))
    os.environ['SMOACS_ROOT'] = str(assets)
    os.environ['MPLBACKEND'] = 'Agg'
    from src.models.alignn4inv import ALIGNN4inverse
    from src.models import alignn_loader
    from alignn_checkpoint import TrustedCheckpointTorch
    original_loader_torch = alignn_loader.torch
    alignn_loader.torch = TrustedCheckpointTorch(torch, checkpoint_digests)
    report['legacy_checkpoint_digests'] = checkpoint_digests

    def check(tensor):
        if not tensor.is_cuda or not torch.isfinite(tensor).all().item():
            raise RuntimeError('ALIGNN value/gradient must be finite and on CUDA.')

    def input_backward(name, gradient):
        check(gradient)
        report['input_backwards'][name] += 1
        report['input_gradient_max'][name] = max(report['input_gradient_max'][name], gradient.abs().max().item())

    def output_backward(gradient):
        check(gradient)
        report['output_backwards'] += 1

    def forward(module, inputs, result):
        if not isinstance(module, ALIGNN4inverse):
            return
        check(result)
        for graph in inputs[0]:
            if graph.device.type != 'cuda':
                raise RuntimeError('ALIGNN graph is not on CUDA.')
        report['forward_calls'] += 1
        for name, tensor in zip(report['input_backwards'], inputs[1:]):
            check(tensor)
            if tensor.requires_grad:
                tensor.register_hook(lambda gradient, name=name: input_backward(name, gradient))
        if result.requires_grad:
            result.register_hook(output_backward)

    handle = torch.nn.modules.module.register_module_forward_hook(forward)
    template = assets / 'reproducibility/section_2_3/templates_masks'
    sys.argv = [str(source/'main.py'), '--model_name', 'ALIGNN', '--task', 'bg_and_eform',
                '--target_val', '2.5', '--training_margin', '0.04', '--eval_margin', '0.2',
                '--e_form_coef', '1.0', '--e_form_criteria', '-0.5',
                '--space_group', '-1', '--optimizer', 'gradient',
                '--num_candidate', '2', '--b_size', '2', '--num_steps', '10', '--num_workers', '0',
                '--initial_structures_path', str(template/'initial_mp_nsites2_12__mask_dict_super_common.pkl'),
                '--mask_dict_path', str(template/'masks/mask_dict_super_common.npz'),
                '--alignn_weights_dir', str(weights), '--device', 'cuda', '--seed', '42',
                '--exp_name', 'smoacs/'+output.name]
    report['argv'] = sys.argv.copy()
    try:
        runpy.run_path(str(source/'main.py'), run_name='__main__')
        if report['forward_calls'] < 20 or report['output_backwards'] < 20:
            raise RuntimeError('Missing ten-step two-predictor CUDA execution.')
        if any(n < 20 for n in report['input_backwards'].values()):
            raise RuntimeError('Missing ALIGNN input gradients.')
        if any(x <= 0 for x in report['input_gradient_max'].values()):
            raise RuntimeError('No nonzero input gradients observed.')
        report['passed'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        alignn_loader.torch = original_loader_torch
        handle.remove()
        (output/'cuda-audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
