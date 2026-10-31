"""Verified official base/conditional checkpoints, shared GPU score interfaces."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from policy import add_lora, sym6, project_cell, Kernel
from prepare import BASE_SHA, BASE_SIZE, sha

HERE = Path(__file__).resolve().parent


def tensor_state_hash(module, frozen_only=False):
    import hashlib
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        if frozen_only and value.requires_grad: continue
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Model:
    def __init__(self, root, kind='base'):
        if not torch.cuda.is_available(): raise RuntimeError('MatterGen policy requires CUDA.')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assets = Path('results/mattergen').resolve()
        spec = json.loads((HERE.parent/'sources.json').read_text())
        sys.path.insert(0, str(HERE.parent/'denoised_tto'))
        from adapter import verify_source
        files = verify_source(assets, spec)
        provenance = None
        if kind == 'base':
            path = root/'assets/base'; expected = BASE_SHA
            recovery = root/'assets/base-recovery.json'
            if recovery.exists():
                from recover_base import MANIFEST_SHA
                provenance = json.loads(recovery.read_text())
                if not provenance['passed'] or provenance['official_base_full_checkpoint_sha256'] != BASE_SHA or provenance['manifest_sha256'] != MANIFEST_SHA:
                    raise ValueError('Unverified recovered base identity.')
                path = root/'assets/base-verified'; expected = provenance['derived_checkpoint_sha256']
                if sha(root/'assets/base-tensor-manifest.json') != MANIFEST_SHA:
                    raise ValueError('Recovered-base manifest identity changed.')
                if sha(path/'config.yaml') != provenance['config_sha256']:
                    raise ValueError('Recovered base config changed.')
            elif (path/'checkpoints/last.ckpt').stat().st_size != BASE_SIZE:
                raise ValueError('Incomplete official base checkpoint.')
        elif kind == 'conditional':
            path = assets/'checkpoints/dft_band_gap'; expected = spec['checkpoint_sha256']
        else: raise ValueError('Unknown checkpoint role.')
        if sha(path/'checkpoints/last.ckpt') != expected:
            raise ValueError('Official checkpoint identity mismatch.')
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.generator import CrystalGenerator
        from mattergen.diffusion import lightning_module
        sys.path.insert(0, str(HERE.parents[1]/'smoacs'))
        from alignn_checkpoint import TrustedCheckpointTorch
        info = MatterGenCheckpointInfo(model_path=path, load_epoch='last', strict_checkpoint_loading=True,
            config_overrides=["++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"])
        self.generator = CrystalGenerator(checkpoint_info=info, batch_size=4, num_batches=1,
            properties_to_condition_on={'dft_band_gap': 2.5} if kind == 'conditional' else {},
            diffusion_guidance_factor=2. if kind == 'conditional' else 0., record_trajectories=False)
        original = lightning_module.torch
        lightning_module.torch = TrustedCheckpointTorch(torch, [expected])
        try: self.generator.prepare()
        finally: lightning_module.torch = original
        self.module = self.generator._model.diffusion_module.eval()
        self.module.requires_grad_(False)
        if not all(p.is_cuda for p in self.module.parameters()):
            raise RuntimeError('Policy model CPU fallback.')
        self.kind = kind
        trained = self.module.model.cond_fields_model_was_trained_on
        if kind == 'base' and trained:
            raise ValueError('Unconditional checkpoint unexpectedly conditions on a property.')
        if kind == 'conditional' and 'dft_band_gap' not in trained:
            raise ValueError('Conditional baseline lacks bandgap embedding.')
        from hydra.utils import instantiate
        cfg = self.generator.sampling_config
        self.sampler = instantiate(cfg.sampler_partial)(pl_module=SimpleNamespace(diffusion_module=self.module, device=torch.device('cuda')))
        self.kernel = Kernel(self.sampler)
        self.metadata = dict(kind=kind, checkpoint_sha256=expected, config_sha256=sha(path/'config.yaml'),
            upstream_commit=spec['upstream_commit'], verified_source_files=len(files),
            guidance=2. if kind == 'conditional' else 0., n_steps=self.sampler.N,
            device=torch.cuda.get_device_name(), torch=torch.__version__)
        if provenance is not None: self.metadata['exact_base_tensor_provenance'] = provenance
        self.lora_names = []

    def lora(self, rank=4):
        if self.lora_names: raise ValueError('Do not install LoRA twice.')
        self.lora_names = add_lora(self.module.model, rank)
        self.metadata.update(lora_rank=rank, lora_layers=len(self.lora_names),
            trainable_parameters=sum(p.numel() for p in self.module.parameters() if p.requires_grad))
        if not self.lora_names: raise ValueError('No policy adaptation layers.')

    def batch(self, counts):
        from torch_geometric.data import Batch
        from mattergen.common.data.chemgraph import ChemGraph
        items = []
        for n in counts:
            fields = dict(atomic_numbers=torch.ones(int(n), dtype=torch.long, device='cuda'),
                cell=torch.eye(3, device='cuda')[None], pos=torch.zeros((int(n), 3), device='cuda'),
                num_atoms=torch.tensor([n], dtype=torch.long, device='cuda'), num_nodes=int(n))
            if self.kind == 'conditional': fields['dft_band_gap'] = torch.tensor([[2.5]], device='cuda')
            items.append(ChemGraph(**fields))
        return Batch.from_data_list(items)

    def initial(self, counts, seed):
        from mattergen.diffusion.sampling.pc_sampler import _sample_prior
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed % 2**32)
        x = _sample_prior(self.module.corruption, self.batch(counts), mask={})
        return x.replace(cell=project_cell(sym6(x.cell)))

    def state(self, x):
        return {k: x[k].detach().cpu().clone() for k in ('pos', 'cell', 'atomic_numbers', 'num_atoms')}

    def restore(self, state):
        x = self.batch(state['num_atoms'].tolist())
        return x.replace(**{k: state[k].to('cuda') for k in ('pos', 'cell', 'atomic_numbers')})

    def adaptation_state(self):
        return {name: value.detach().cpu() for name, value in self.module.named_parameters() if value.requires_grad}

    def load_adaptation(self, state):
        current = {name: p for name, p in self.module.named_parameters() if p.requires_grad}
        if set(current) != set(state): raise ValueError('LoRA checkpoint parameter coverage differs.')
        with torch.no_grad():
            for name, value in state.items(): current[name].copy_(value.to(current[name]))
