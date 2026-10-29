"""Scoped reuse of pinned GPU models without changing legacy source or samplers."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from probe_math import cartesian_score, project

HERE = Path(__file__).resolve().parent


def isolated(path, name, bindings=None):
    bindings = bindings or {}
    old_path = list(sys.path)
    old_modules = {key: sys.modules.get(key) for key in bindings}
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
        for key, value in old_modules.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class Models:
    def __init__(self, root):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required.')
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        mapping = isolated(HERE.parent/'denoised_tto/mapping.py', 'score_probe_mapping')
        adapter = isolated(HERE.parent/'denoised_tto/adapter.py', 'score_probe_adapter', {'mapping': mapping})
        self.denoiser = adapter.Denoiser()
        self.geometry = mapping.LatentGeometry
        self.score_raw = adapter.score_with_raw
        self.sha = adapter.sha
        geom = isolated(HERE.parent/'alignn_templates/geometry.py', 'score_probe_alignn_geometry')
        alignn = isolated(HERE.parent/'alignn_templates/model.py', 'score_probe_alignn', {'geometry': geom})
        models, self.alignn_metadata = alignn.load_models(root)
        self.alignn = alignn.Predictor([m.to(device='cuda', dtype=torch.float64) for m in models])
        runtime = isolated(HERE.parent/'directed_tto/runtime.py', 'score_probe_mechanics')
        rpc = root/'rpc'
        rpc.mkdir()
        self.mechanics = runtime.Mechanics(rpc)

    def branches(self, graph, time):
        from mattergen.property_embeddings import replace_use_unconditional_embedding
        torch = self.torch
        out, checks = {}, {}
        with torch.no_grad():
            for name, disabled in [('conditional', False), ('unconditional', True)]:
                branch = replace_use_unconditional_embedding(graph, {
                    'dft_band_gap': torch.full((1, 1), disabled, dtype=torch.bool, device='cuda')})
                score, raw = self.score_raw(self.denoiser.module, branch, time)
                self.denoiser.metadata['score_calls'] += 1
                if self.denoiser.module.model_targets['pos'].value != 'score_times_std':
                    raise ValueError('Unexpected official positional output target.')
                _, std = self.denoiser.module.corruption.sdes['pos'].marginal_prob(
                    graph.pos, time, graph.batch, graph)
                error = float((score.pos-raw.pos/std).abs().max())
                if error > 1e-5 or not torch.isfinite(score.pos).all():
                    raise ValueError('Official same-output score conversion failed.')
                checks[name+'_raw_to_score_error'] = error
                frac = score.pos.cpu().double().numpy()
                cell = graph.cell[0].cpu().double().numpy()
                out[name] = cartesian_score(frac, cell)
                # The sampler pushes a fractional-coordinate update forward;
                # this differs from pulling a score covector back to Cartesian.
                out['drift_'+name] = project(frac @ cell)
        out['delta'] = out['conditional']-out['unconditional']
        out['guided'] = out['unconditional']+2*out['delta']
        out['drift_delta'] = out['drift_conditional']-out['drift_unconditional']
        out['drift_guided'] = out['drift_unconditional']+2*out['drift_delta']
        return out, checks

    def property(self, numbers, cell, cart, gradient=False, native=False, graphs=None):
        torch = self.torch
        lattice = torch.tensor(np.asarray(cell)[None], device='cuda', dtype=torch.float64)
        positions = torch.tensor(np.asarray(cart), device='cuda', dtype=torch.float64, requires_grad=gradient)
        frac = positions @ torch.linalg.inv(lattice[0])
        onehot = torch.nn.functional.one_hot(torch.as_tensor(numbers, device='cuda')-1, 98).double()
        size = torch.tensor([len(numbers)], device='cuda')
        with torch.enable_grad() if gradient else torch.no_grad():
            if graphs is None:
                graphs = self.alignn.graphs(lattice, frac, onehot, size)
            gap, ef = self.alignn.predict(lattice, frac, onehot, size, graphs=graphs, native_control=native)
            loss = .5*(gap[0]-2.5)**2
            g = torch.autograd.grad(loss, positions)[0] if gradient else None
        return dict(gap=float(gap[0].detach()), eform=float(ef[0].detach()), loss=float(loss.detach()),
                    G=None if g is None else project(-g.detach().cpu().numpy()), graphs=graphs)

    def physical(self, numbers, cell, carts):
        tasks = []
        for cart in carts:
            self.mechanics.serial += 1
            tasks.append(dict(id=f'probe-{self.mechanics.serial}', action='point', numbers=numbers.tolist(),
                              cell=cell.tolist(), frac=(cart@np.linalg.inv(cell)).tolist()))
        rows = self.mechanics.request(tasks)
        if any(r['status'] != 'completed' for r in rows):
            raise RuntimeError('MatterSim point failed: '+repr(rows))
        return rows

    def close(self):
        self.mechanics.close()
