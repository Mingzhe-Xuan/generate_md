"""Native MatterGen CSP sampling with fixed-species and CUDA audits, then canonical B."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from validation import digest, validate_composition, validate_csp_config, validate_sampling

HERE = Path(__file__).resolve().parent
SMOKE = [{'Si': 8}, {'Sr': 1, 'Ti': 1, 'O': 3}]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(args):
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from pymatgen.core import Element, Structure
    from ase.io import read, write
    from ase import Atoms

    started = time.perf_counter()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    mechanics = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CSP requires CUDA; CPU fallback is forbidden.')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assets = Path('results/mattergen').resolve()
        model_path = assets / 'csp-assets'
        verified = json.loads((model_path / 'verified.json').read_text())
        spec = json.loads((HERE / 'sources.json').read_text())
        if verified['source'] != spec:
            raise ValueError('Asset provenance changed.')
        if digest(model_path / 'checkpoints/last.ckpt') != verified['checkpoint_sha256']:
            raise ValueError('CSP checkpoint changed after verification.')
        if digest(model_path / 'config.yaml') != spec['config_sha256']:
            raise ValueError('CSP config changed after verification.')
        sys.path.insert(0, str(HERE.parent / 'denoised_tto'))
        from adapter import verify_source
        native = json.loads((HERE.parent / 'sources.json').read_text())
        source_hashes = verify_source(assets, native)
        sys.path.insert(0, str(assets / 'source'))
        from mattergen.generator import CrystalGenerator, seed_all
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.diffusion import lightning_module
        sys.path.insert(0, str(HERE.parents[1] / 'smoacs'))
        from alignn_checkpoint import TrustedCheckpointTorch

        info = MatterGenCheckpointInfo(model_path=model_path, load_epoch='last', strict_checkpoint_loading=True,
            config_overrides=[f'lightning_module.diffusion_module.model.gemnet.scale_file={assets}/source/mattergen/common/gemnet/gemnet-dT.json'])
        validate_csp_config(OmegaConf.to_container(info.config, resolve=True))
        compositions = SMOKE if args.compositions is None else json.loads(args.compositions)
        compositions = [validate_composition(c) for c in compositions]
        for c in compositions:
            if any(not 1 <= Element(e).Z <= 94 for e in c):
                raise ValueError('Smoke compositions must be supported by MatterSim.')
        generator = CrystalGenerator(checkpoint_info=info, batch_size=1,
            num_batches=args.samples_per_composition * len(compositions),
            target_compositions_dict=compositions, sampling_config_name='csp',
            properties_to_condition_on={}, diffusion_guidance_factor=0., record_trajectories=False)
        config = generator.sampling_config
        validate_sampling(OmegaConf.to_container(config, resolve=True))
        save(root / 'plan.json', dict(source=verified, native_source=native,
             source_files=source_hashes, compositions=compositions, batch_size=1,
             samples_per_composition=args.samples_per_composition, seed=args.seed,
             sampling=OmegaConf.to_container(config, resolve=True),
             effective_model_config=OmegaConf.to_container(info.config, resolve=True),
             commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             module_files={p.name: digest(p) for p in HERE.iterdir() if p.is_file()},
             force_guidance=False, property_conditioning={}, DFT=False))
        original = lightning_module.torch
        lightning_module.torch = TrustedCheckpointTorch(torch, [verified['checkpoint_sha256']])
        try:
            generator.prepare()
        finally:
            lightning_module.torch = original
        model = generator._model
        if not all(p.is_cuda for p in model.parameters()):
            raise RuntimeError('CSP parameters are not entirely on CUDA.')
        if set(model.diffusion_module.corruption.corruptions) != {'pos', 'cell'}:
            raise ValueError('Loaded model is not CSP.')
        seed_all(args.seed)
        sampler = instantiate(config.sampler_partial)(pl_module=model)
        loader = generator.get_condition_loader(config, compositions)
        rows = []
        generation_start = time.perf_counter()
        for index, (condition, mask) in enumerate(loader):
            folder = root / f'sample-{index:02d}'
            folder.mkdir()
            expected = condition.atomic_numbers.clone().cuda()
            composition = compositions[index // args.samples_per_composition]
            want = sorted(Element(e).Z for e, n in composition.items() for _ in range(n))
            if sorted(expected.cpu().tolist()) != want or mask:
                raise ValueError('Wrong composition loader or unexpected inpainting mask.')
            audit = dict(score_calls=0, denoiser_calls=0, fixed_species=True, all_cuda=True,
                         finite_geometry=True, snapshots=[])
            original_score = sampler._score_fn
            def observe(module, inputs, output):
                if not all(v.is_cuda for v in (output.pos, output.cell)):
                    raise RuntimeError('CPU denoiser output.')
                if not all(torch.isfinite(v).all() for v in (output.pos, output.cell)):
                    raise ValueError('Nonfinite denoiser geometry.')
                audit['denoiser_calls'] += 1
            def score(x, t):
                call = audit['score_calls']
                if not torch.equal(x.atomic_numbers, expected):
                    raise ValueError('Species changed along the CSP trajectory.')
                if not all(v.is_cuda for v in (x.pos, x.cell, x.atomic_numbers)):
                    raise RuntimeError('CPU sampler state.')
                if not all(torch.isfinite(v).all() for v in (x.pos, x.cell)):
                    raise ValueError('Nonfinite sampler geometry.')
                if call in (0, 500, 1000, 1500, 1800, 1998):
                    path = folder / f'state-{call:04d}.npz'
                    np.savez_compressed(path, numbers=x.atomic_numbers.cpu().numpy(),
                        cell=x.cell.cpu().numpy(), frac=x.pos.cpu().numpy(), t=t.cpu().numpy())
                    audit['snapshots'].append(dict(call=call, file=path.name, sha256=digest(path)))
                audit['score_calls'] += 1
                return original_score(x, t)
            handle = model.diffusion_module.model.register_forward_hook(observe)
            sampler._score_fn = score
            tic = time.perf_counter()
            try:
                sample, mean = sampler.sample(condition, mask)
                torch.cuda.synchronize()
            finally:
                sampler._score_fn = original_score
                handle.remove()
                save(folder / 'sampling-audit.json', audit)
            if audit['score_calls'] != 2000 or audit['denoiser_calls'] != 2000:
                raise ValueError('Incomplete native CSP trajectory.')
            for terminal in (sample, mean):
                if not torch.equal(terminal.atomic_numbers, expected):
                    raise ValueError('Species changed at terminal state.')
            cell = mean.cell[0].cpu().double().numpy()
            frac = mean.pos.cpu().double().numpy()
            if not np.isfinite(frac).all() or not np.isfinite(cell).all() or np.linalg.det(cell) <= 1e-6:
                raise ValueError('Invalid generated structure.')
            atoms = Atoms(numbers=expected.cpu().numpy(), cell=cell, scaled_positions=frac, pbc=True)
            path = folder / 'generated.vasp'
            write(path, atoms, format='vasp', direct=True, sort=False)
            parsed = read(path, format='vasp')
            if not np.array_equal(parsed.numbers, expected.cpu().numpy()):
                raise ValueError('Serialization changed atom ordering.')
            Structure(cell, expected.cpu().numpy(), frac).to(filename=str(folder / 'generated.cif'))
            row = dict(id=f'sample-{index:02d}', composition=composition, numbers=expected.cpu().tolist(),
                       generated_path=str(path), generated_sha256=digest(path),
                       sampling_seconds=time.perf_counter() - tic, audit=audit)
            rows.append(row)
            save(root / 'generation-progress.json', rows)
            print('CSP_GENERATED', row['id'], composition, row['sampling_seconds'], flush=True)
        if len(rows) != len(compositions) * args.samples_per_composition:
            raise ValueError('Missing generated structures.')
        sampling_seconds = time.perf_counter() - generation_start
        # Existing canonical B worker uses its own unchanged MatterSim environment.
        path = HERE.parent / 'directed_tto/runtime.py'
        module_spec = importlib.util.spec_from_file_location('csp_physics', path)
        runtime = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runtime)
        rpc = root / 'rpc'
        rpc.mkdir()
        mechanics = runtime.Mechanics(rpc)
        points = mechanics.request([dict(id=r['id'], action='point', numbers=r['numbers'],
            cell=read(r['generated_path']).cell.array.tolist(), frac=read(r['generated_path']).get_scaled_positions().tolist()) for r in rows])
        relaxed = mechanics.request([dict(id=r['id'], action='relax', source=r['generated_path'],
            output=str(root / r['id'] / 'relaxation')) for r in rows])
        for row, point, result in zip(rows, points, relaxed):
            row.update(initial_physics=point, relaxation=result)
            if result['status'] in ('converged', 'unconverged'):
                if read(result['relaxed_path']).numbers.tolist() != row['numbers']:
                    raise ValueError('MatterSim changed the composition.')
        save(root / 'results.json', rows)
        passed = all(r['initial_physics']['status'] == 'completed' and r['relaxation']['status'] == 'converged' for r in rows)
        save(root / 'complete.json', dict(passed=passed, count=len(rows), source=verified,
            sampling_seconds=sampling_seconds, total_seconds=time.perf_counter()-started,
            device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
            numpy=np.__version__, physics_NFE=mechanics.NFE,
            parent_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            note='CUDA memory excludes MatterSim workers; smoke is not a property/SUN benchmark.'))
        if not passed:
            raise RuntimeError('CSP sampling completed, but physical smoke did not fully pass; see results.')
    except Exception:
        save(root / 'failure.json', dict(traceback=traceback.format_exc(), seconds=time.perf_counter()-started))
        raise
    finally:
        if mechanics is not None:
            mechanics.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compositions', help='JSON list of explicit element-count dictionaries')
    parser.add_argument('--samples-per-composition', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261051)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use Slurm.')
    if args.samples_per_composition < 1:
        parser.error('Sample count must be positive.')
    execute(args)


if __name__ == '__main__':
    main()
