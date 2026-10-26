"""Audited differentiable access to the cached official MatterGen model."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
from contextlib import contextmanager
from mapping import clean_estimate, symmetric_noise


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_source(assets, spec):
    archive = assets/'mattergen-source.tar.gz'
    if sha(archive) != spec['source_archive_sha256']:
        raise ValueError('Official source archive changed.')
    result = {}
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if not member.isfile() or not member.name.startswith('mattergen/'):
                continue
            if not member.name.endswith(('.py', '.yaml')):
                continue
            expected = hashlib.sha256(handle.extractfile(member).read()).hexdigest()
            if sha(assets/'source'/member.name) != expected:
                raise ValueError('Cached official source changed: '+member.name)
            result[member.name] = expected
    if len(result) < 50:
        raise ValueError('Incomplete official source archive.')
    return result


def score_with_raw(module, graph, time):
    captured = []
    handle = module.model.register_forward_hook(lambda model, inputs, output: captured.append(output))
    try:
        score = module.score_fn(graph, time)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError('Official score conversion did not perform one model call.')
    return score, captured[0]


class Denoiser:
    def __init__(self):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('MatterGen denoising requires CUDA.')
        self.spec = json.loads(Path(__file__).resolve().parents[1].joinpath('sources.json').read_text())
        assets = Path('results/mattergen').resolve()
        sources = verify_source(assets, self.spec)
        model_path = assets/'checkpoints/dft_band_gap'
        for path, expected in [(model_path/'checkpoints/last.ckpt', self.spec['checkpoint_sha256']),
                               (model_path/'config.yaml', self.spec['config_sha256'])]:
            if sha(path) != expected:
                raise ValueError('MatterGen checkpoint/config changed.')
        sys.path.insert(0, str(assets/'source'))
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.generator import CrystalGenerator
        from mattergen.diffusion import lightning_module
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'smoacs'))
        from alignn_checkpoint import TrustedCheckpointTorch
        info = MatterGenCheckpointInfo(model_path=model_path, load_epoch='last', strict_checkpoint_loading=True,
            config_overrides=["++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"])
        generator = CrystalGenerator(checkpoint_info=info, batch_size=1, num_batches=1,
            properties_to_condition_on={'dft_band_gap': 2.5}, diffusion_guidance_factor=2., seed=42)
        original = lightning_module.torch
        lightning_module.torch = TrustedCheckpointTorch(torch, [self.spec['checkpoint_sha256']])
        try:
            generator.prepare()
        finally:
            lightning_module.torch = original
        self.module = generator._model.diffusion_module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if not all(p.is_cuda for p in self.module.parameters()):
            raise RuntimeError('MatterGen CPU fallback.')
        self.metadata = dict(source=self.spec, source_files=sources, score_calls=0,
            conditioning={'dft_band_gap': 2.5}, guidance_scale=2., fixed_species=True)
        self.precision_helpers = None

    @contextmanager
    def precision_context(self, dtype):
        if dtype == self.torch.float64:
            from precision import helpers, replace_aliases
            if self.precision_helpers is None:
                self.precision_helpers, record = helpers()
                self.metadata['F64_geometry_compatibility'] = record
            with replace_aliases(self.precision_helpers):
                yield
        else:
            yield

    def graph(self, geometry, q):
        from torch_geometric.data import Batch
        from mattergen.common.data.chemgraph import ChemGraph
        torch = self.torch
        cell, frac = geometry.torch_pose(q)
        item = ChemGraph(atomic_numbers=torch.as_tensor(geometry.numbers, dtype=torch.long, device=q.device),
            cell=cell[None], pos=frac.remainder(1.), num_atoms=torch.tensor([geometry.n], device=q.device),
            num_nodes=geometry.n, dft_band_gap=torch.tensor([[2.5]], dtype=q.dtype, device=q.device))
        return Batch.from_data_list([item])

    def forward_noise(self, graph, time, seed):
        torch = self.torch
        t = torch.full((1,), time, dtype=graph.cell.dtype, device=graph.cell.device)
        rng = np.random.default_rng(seed)
        eps_pos = torch.tensor(rng.normal(size=tuple(graph.pos.shape)), dtype=graph.pos.dtype, device=graph.pos.device)
        eps_cell = torch.tensor(symmetric_noise(rng.normal(size=(3, 3)))[None], dtype=graph.cell.dtype, device=graph.cell.device)
        pos_sde, cell_sde = (self.module.corruption.sdes[k] for k in ('pos', 'cell'))
        mean_p, std_p = pos_sde.marginal_prob(graph.pos, t, graph.batch, graph)
        mean_c, std_c = cell_sde.marginal_prob(graph.cell, t, None, graph)
        return graph.replace(pos=(mean_p + std_p*eps_pos).remainder(1.), cell=mean_c+std_c*eps_cell), t

    def scores(self, graph, t):
        from mattergen.property_embeddings import replace_use_unconditional_embedding
        torch = self.torch
        conditional = replace_use_unconditional_embedding(graph, {'dft_band_gap': torch.zeros((1, 1), dtype=torch.bool, device=t.device)})
        unconditional = replace_use_unconditional_embedding(graph, {'dft_band_gap': torch.ones((1, 1), dtype=torch.bool, device=t.device)})
        a = self.module.score_fn(conditional, t)
        b = self.module.score_fn(unconditional, t)
        self.metadata['score_calls'] += 2
        return a.replace(pos=2*a.pos-b.pos, cell=2*a.cell-b.cell)

    def decode(self, graph, t, stop_score_gradient=False):
        scores = self.scores(graph, t)
        if stop_score_gradient:
            scores = scores.replace(pos=scores.pos.detach(), cell=scores.cell.detach())
        pos_sde, cell_sde = (self.module.corruption.sdes[k] for k in ('pos', 'cell'))
        _, sp = pos_sde.marginal_prob(graph.pos, t, graph.batch, graph)
        _, sc = cell_sde.marginal_prob(graph.cell, t, None, graph)
        coefficient = cell_sde._marginal_mean_coeff(t)[:, None, None]
        limit = cell_sde.get_limit_mean(graph.cell, graph)
        clean_pos = clean_estimate(graph.pos, scores.pos, sp).remainder(1.)
        clean_cell = clean_estimate(graph.cell, scores.cell, sc, coefficient, limit)
        return graph.replace(pos=clean_pos, cell=clean_cell)
