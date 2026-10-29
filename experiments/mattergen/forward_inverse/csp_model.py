"""Hash-pinned native CSP fields in FP32 and dtype-controlled FP64."""
from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def load_file(path, name, bindings=None):
    bindings = bindings or {}
    previous = {key: sys.modules.get(key) for key in bindings}
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        sys.modules.update(bindings)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class CSP:
    def __init__(self):
        import torch
        from omegaconf import OmegaConf
        if not torch.cuda.is_available():
            raise RuntimeError('CSP fields require CUDA.')
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assets = Path('results/mattergen').resolve()
        model_path = assets / 'csp-assets'
        spec = json.loads((HERE.parent/'csp/sources.json').read_text())
        verified = json.loads((model_path/'verified.json').read_text())
        expected = '67acb8a009762415cad981333a86528e3e7e38aab9a25ec76d4abf316d246009'
        if verified['source'] != spec or sha(model_path/'checkpoints/last.ckpt') != expected or sha(model_path/'config.yaml') != spec['config_sha256']:
            raise ValueError('CSP assets changed.')
        mapping = load_file(HERE.parent/'denoised_tto/mapping.py', 'fi_mapping')
        adapter = load_file(HERE.parent/'denoised_tto/adapter.py', 'fi_old_adapter', {'mapping': mapping})
        native = json.loads((HERE.parent/'sources.json').read_text())
        source_hashes = adapter.verify_source(assets, native)
        sys.path.insert(0, str(assets/'source'))
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.generator import CrystalGenerator
        from mattergen.diffusion import lightning_module
        validation = load_file(HERE.parent/'csp/validation.py', 'fi_csp_validation')
        sys.path.insert(0, str(HERE.parents[1]/'smoacs'))
        from alignn_checkpoint import TrustedCheckpointTorch
        info = MatterGenCheckpointInfo(model_path=model_path, load_epoch='last', strict_checkpoint_loading=True,
            config_overrides=[f'lightning_module.diffusion_module.model.gemnet.scale_file={assets}/source/mattergen/common/gemnet/gemnet-dT.json'])
        validation.validate_csp_config(OmegaConf.to_container(info.config, resolve=True))
        generator = CrystalGenerator(checkpoint_info=info, batch_size=1, num_batches=1,
            target_compositions_dict=[{'Si': 8}], sampling_config_name='csp', properties_to_condition_on={})
        original = lightning_module.torch
        lightning_module.torch = TrustedCheckpointTorch(torch, [expected])
        try:
            generator.prepare()
        finally:
            lightning_module.torch = original
        self.native = generator._model.diffusion_module.eval()
        self.double = copy.deepcopy(self.native).double().eval()
        for model in (self.native, self.double):
            if set(model.corruption.corruptions) != {'pos', 'cell'} or not all(p.is_cuda for p in model.parameters()):
                raise ValueError('Expected fully CUDA CSP model.')
            for p in model.parameters():
                p.requires_grad_(False)
        self.precision = load_file(HERE.parent/'denoised_tto/precision.py', 'fi_precision')
        self.helpers, edits = self.precision.helpers(stable_offsets=False)
        self.metadata = dict(checkpoint_sha256=expected, source=native, source_files=source_hashes,
            dtype_adapters=edits, fp64='Same weights/architecture; only existing pinned OCP dtype edits, no stable-offset arithmetic rewrite.',
            property_conditioning={}, fixed_species=True, calls_fp32=0, calls_fp64=0)

    def score(self, numbers, cell, cart, time, dtype='double', adjoint=None):
        from torch_geometric.data import Batch
        from mattergen.common.data.chemgraph import ChemGraph
        torch = self.torch
        model = self.double if dtype == 'double' else self.native
        scalar = torch.float64 if dtype == 'double' else torch.float32
        lattice = torch.tensor(cell, dtype=scalar, device='cuda')
        positions = torch.tensor(cart, dtype=scalar, device='cuda', requires_grad=adjoint is not None)
        frac = positions @ torch.linalg.inv(lattice)
        item = ChemGraph(atomic_numbers=torch.tensor(numbers, dtype=torch.long, device='cuda'),
            num_atoms=torch.tensor([len(numbers)], dtype=torch.long, device='cuda'), num_nodes=len(numbers),
            cell=lattice[None], pos=frac)
        graph = Batch.from_data_list([item])
        t = torch.tensor([time], dtype=scalar, device='cuda')
        gemnet = model.model.gemnet
        original_graph = gemnet.generate_interaction_graph
        fingerprints = []
        def observe_graph(*args, **kwargs):
            result = original_graph(*args, **kwargs)
            edges = result[0].T.detach().cpu().numpy()
            offsets = result[8].detach().cpu().numpy()
            topology = np.column_stack([edges, offsets]).astype(np.int64)
            topology = topology[np.lexsort(topology.T[::-1])]
            fingerprints.append(hashlib.sha256(topology.tobytes()).hexdigest())
            return result
        raw = []
        handle = model.model.register_forward_hook(lambda module, inputs, output: raw.append(output))
        gemnet.generate_interaction_graph = observe_graph
        precision = self.precision.replace_aliases(self.helpers) if dtype == 'double' else nullcontext()
        try:
            with precision, torch.enable_grad() if adjoint is not None else torch.no_grad():
                score = model.score_fn(graph, t)
                _, std = model.corruption.sdes['pos'].marginal_prob(graph.pos, t, graph.batch, graph)
                if len(raw) != 1 or len(fingerprints) != 1 or model.model_targets['pos'].value != 'score_times_std':
                    raise ValueError('Unexpected native score conversion or call count.')
                conversion = float((score.pos - raw[0].pos/std).detach().abs().max())
                if conversion > (1e-9 if dtype == 'double' else 1e-5):
                    raise ValueError('Native score conversion failed.')
                sr = score.pos @ torch.linalg.inv(lattice).T
                sr = sr - sr.mean(0)
                if not sr.is_cuda or not torch.isfinite(sr).all() or not torch.equal(graph.atomic_numbers, item.atomic_numbers):
                    raise ValueError('CSP finite/device/species audit failed.')
                gradient = None
                if adjoint is not None:
                    w = torch.tensor(adjoint, dtype=scalar, device='cuda').reshape(sr.shape)
                    gradient = torch.autograd.grad((sr*w).sum(), positions)[0].detach().cpu().numpy().ravel()
            self.metadata['calls_fp64' if dtype == 'double' else 'calls_fp32'] += 1
            return dict(score=sr.detach().cpu().numpy().ravel(), fractional_score=score.pos.detach().cpu().numpy(),
                std=float(std.detach().flatten()[0]), std_range=float((std.max()-std.min()).detach()),
                graph_hash=fingerprints[0], raw_to_score_error=conversion, adjoint=gradient)
        finally:
            gemnet.generate_interaction_graph = original_graph
            handle.remove()
