"""Paired native joint de novo trajectories: bandgap CFG2 versus CFG0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'forward_inverse'))
from csp_model import load_file, sha


def save(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main(root):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    mapping = load_file(HERE.parent/'denoised_tto/mapping.py', 'conditional_mapping')
    adapter = load_file(HERE.parent/'denoised_tto/adapter.py', 'conditional_adapter', {'mapping': mapping})
    model = adapter.Denoiser()
    from mattergen.generator import CrystalGenerator, seed_all
    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
    from mattergen.property_embeddings import get_use_unconditional_embedding
    if 'dft_band_gap' not in model.module.model.cond_fields_model_was_trained_on:
        raise ValueError('Checkpoint has no gap conditioning.')
    observer_module = load_file(HERE.parent/'random_score_probe/observer.py', 'conditional_observer')
    info = MatterGenCheckpointInfo(model_path=Path('results/mattergen/checkpoints/dft_band_gap').absolute(), load_epoch='last')
    generator = CrystalGenerator(checkpoint_info=info, batch_size=4, num_batches=3,
        properties_to_condition_on={'dft_band_gap': 2.5}, diffusion_guidance_factor=2., record_trajectories=False, seed=20261070)
    config = generator.sampling_config
    seed_all(20261070)
    batches = list(generator.get_condition_loader(config, None))
    if len(batches) != 3 or any(len(x.to_data_list()) != 4 or bool(mask) for x, mask in batches):
        raise ValueError('Unexpected native joint batch layout.')
    model.metadata['fixed_species'] = False
    plan = dict(target=2.5, checkpoint=model.metadata, batch_size=4, batches=3, n_steps=1000,
        observation_steps=list(range(0, 1000, 5))+[999],
        config=OmegaConf.to_container(config, resolve=True),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        code_sha256={str(p): sha(p) for p in HERE.glob('*.py')},
        interpretation='ALIGNN target adherence proxy, not DFT accuracy; no relaxation. Native joint composition/geometry generation. Three coupled batches per arm.')
    save(root/'plan.json', plan)
    paths = []; audit = []; priors = {}; raw = []; call_count = [0]; errors = []
    current = dict(arm=None, n=None)
    def hook(module, inputs, output):
        x = inputs[0]; arm = current['arm']; n = current['n']
        flags = get_use_unconditional_embedding(x, 'dft_band_gap').flatten()
        expected = [True]*4+[False]*4 if arm == 'cfg2' else [True]*4
        if flags.tolist() != expected or len(output.pos) != n*(2 if arm == 'cfg2' else 1):
            raise ValueError('Native CFG ordering/embedding audit failed.')
        if not all(v.is_cuda and torch.isfinite(v).all() for v in (output.pos, output.cell)):
            raise ValueError('Nonfinite or CPU native output.')
        if not torch.all(x.dft_band_gap == 2.5): raise ValueError('Condition changed.')
        raw[:] = [output.pos]; call_count[0] += 1
    handle = model.module.model.register_forward_hook(hook)
    try:
        for b, (condition, mask) in enumerate(batches):
            current['n'] = int(condition.num_atoms.sum())
            for arm, guidance in [('cfg2', 2.), ('cfg0', 0.)]:
                current['arm'] = arm
                cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
                cfg.sampler_partial.guidance_scale = guidance
                sampler = instantiate(cfg.sampler_partial)(pl_module=SimpleNamespace(diffusion_module=model.module, device=torch.device('cuda')))
                seed_all(20261070+b)
                buffers = [dict(numbers=[], cell=[], frac=[], step=[], stage=[], t=[]) for _ in range(4)]
                def capture(x, t, score, stage, step):
                    _, std = model.module.corruption.sdes['pos'].marginal_prob(x.pos, t, x.get_batch_idx('pos'), x)
                    n = current['n']
                    expected = raw[0]/std if arm == 'cfg0' else torch.lerp(raw[0][:n]/std, raw[0][n:]/std, guidance)
                    error = float((score.pos-expected).abs().max()); errors.append(error)
                    if error > 1e-5: raise ValueError('Native score conversion audit failed.')
                    for j, piece in enumerate(x.clone().cpu().to_data_list()):
                        buf = buffers[j]
                        for key, value in dict(numbers=piece.atomic_numbers.numpy(), cell=piece.cell[0].double().numpy(),
                            frac=piece.pos.double().numpy(), step=step, stage=stage, t=float(t[j])).items(): buf[key].append(value)
                observer = observer_module.Observer(sampler, plan['observation_steps'], capture, torch)
                sampler._score_fn = observer
                tick = time.perf_counter()
                with torch.no_grad(): sample, mean = sampler.sample(condition.clone(), mask)
                sampler._score_fn = observer.original
                if observer.calls != 2000 or observer.snapshots != 202: raise ValueError('Observer coverage changed.')
                for j, terminal in enumerate(mean.clone().cpu().to_data_list()):
                    buf = buffers[j]; parent = b*4+j
                    if arm == 'cfg2': priors[parent] = {k: np.asarray(buf[k][0]).copy() for k in ('numbers', 'cell', 'frac')}
                    elif not all(np.array_equal(priors[parent][k], buf[k][0]) for k in priors[parent]):
                        raise ValueError('Initial priors differ across paired arms.')
                    for key, value in dict(numbers=terminal.atomic_numbers.numpy(), cell=terminal.cell[0].double().numpy(),
                        frac=terminal.pos.double().numpy(), step=1000, stage='terminal', t=0.).items(): buf[key].append(value)
                    path = root/f'{arm}-{parent:02d}.npz'
                    np.savez_compressed(path, **{k: np.asarray(v) for k, v in buf.items()})
                    paths.append(dict(arm=arm, parent=parent, batch=b, seed=20261070+b, file=path.name, sha256=sha(path)))
                audit.append(dict(arm=arm, batch=b, seed=20261070+b, score_calls=observer.calls, snapshots=observer.snapshots,
                    cuda=True, observer_rng_unchanged=True, seconds=time.perf_counter()-tick))
                save(root/'sampling-progress.json', dict(paths=paths, audit=audit, raw_forward_calls=call_count[0]))
                print(json.dumps(audit[-1]), flush=True)
    finally: handle.remove()
    if len(paths) != 24 or call_count[0] != 12000: raise ValueError('Sampling incomplete.')
    save(root/'sampling.json', dict(paths=paths, audit=audit, raw_forward_calls=call_count[0],
        max_conversion_error=max(errors), exact_shared_priors=True, seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Use Slurm.')
    main(args.run.absolute())
