"""Pinned ALIGNN geometry derivatives, reusing the verified GPU implementation."""
from pathlib import Path
import numpy as np
from csp_model import load_file, sha
from relations import centered


class Properties:
    def __init__(self, root):
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Properties require CUDA.')
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        module_root = Path(__file__).resolve().parent.parent / 'alignn_templates'
        geometry = load_file(module_root / 'geometry.py', 'fi_property_geometry')
        model = load_file(module_root / 'model.py', 'fi_property_model', {'geometry': geometry})
        models, metadata = model.load_models(root)
        self.predictor = model.Predictor([m.to(device='cuda', dtype=torch.float64) for m in models])
        if not all(p.is_cuda for m in self.predictor.models for p in m.parameters()):
            raise RuntimeError('Property model CPU fallback.')
        self.metadata = dict(metadata, adapter_sha256={p.name: sha(p) for p in [module_root/'model.py', module_root/'geometry.py']},
                             dtype='float64', device=torch.cuda.get_device_name(), calls=0,
                             properties=['gap_eV', 'eform_eV_atom'])

    def evaluate(self, numbers, cell, cart, gradient=False, native=False, graphs=None):
        torch = self.torch
        lattice = torch.tensor(np.asarray(cell)[None], device='cuda', dtype=torch.float64)
        positions = torch.tensor(cart, device='cuda', dtype=torch.float64, requires_grad=gradient)
        frac = positions @ torch.linalg.inv(lattice[0])
        onehot = torch.nn.functional.one_hot(torch.as_tensor(numbers, device='cuda')-1, 98).double()
        sizes = torch.tensor([len(numbers)], device='cuda')
        with torch.enable_grad() if gradient else torch.no_grad():
            if graphs is None:graphs = self.predictor.graphs(lattice, frac, onehot, sizes)
            outputs = self.predictor.predict(lattice, frac, onehot, sizes, graphs=graphs, native_control=native)
            jacobian = []
            if gradient:
                for index, output in enumerate(outputs):
                    g = torch.autograd.grad(output.sum(), positions, retain_graph=index == 0)[0]
                    jacobian.append(centered(g.detach().cpu().numpy()).reshape(-1, 3))
            values = np.array([float(y[0].detach()) for y in outputs])
        g, _, images = graphs[0]
        src, dst = g.edges(order='eid')
        topology = np.column_stack([src.numpy(), dst.numpy(), images.numpy()]).astype(np.int64)
        topology = topology[np.lexsort(topology.T[::-1])]
        import hashlib
        graph_hash = hashlib.sha256(topology.tobytes()).hexdigest()
        self.metadata['calls'] += 1
        if not np.isfinite(values).all() or (gradient and not np.isfinite(jacobian).all()):
            raise ValueError('Nonfinite property or input derivative.')
        return dict(values=values, gradients=np.array(jacobian) if gradient else None, graph_hash=graph_hash), graphs
