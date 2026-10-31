"""Online generated groups with exact saved latent actions and terminal rewards."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
from policy import action_logp


def generate(model, counts, seed, record_indices=(), progress=None):
    model.module.eval()
    selected = set(record_indices); records = []
    with torch.no_grad():
        x = model.initial(counts, seed)
        initial = model.state(x)
        times = torch.linspace(model.sampler._max_t, model.sampler._eps_t, model.sampler.N, device='cuda')
        dt = -torch.tensor((model.sampler._max_t-model.sampler._eps_t)/(model.sampler.N-1), device='cuda')
        started = time.perf_counter()
        for index, value in enumerate(times):
            t = torch.full((len(counts),), value, device='cuda')
            for stage in ('corrector', 'predictor'):
                distribution = model.kernel.distributions(x, t, dt, stage)
                next_x, action = model.kernel.sample(x, distribution)
                if index in selected:
                    lp = action_logp(distribution, action, x.batch, len(counts))
                    records.append(dict(index=index, stage=stage, t=float(value), dt=float(dt),
                        state=model.state(x), action={k: v.detach().cpu().clone() for k, v in action.items()}, old_logp=lp.cpu()))
                x = next_x
            if progress is not None and (index % 100 == 0 or index == len(times)-1):
                progress(dict(step=index+1, steps=len(times), seconds=time.perf_counter()-started))
        terminal = model.state(x)
    return dict(initial=initial, terminal=terminal, records=records, seconds=time.perf_counter()-started,
        seed=seed, counts=list(counts), score_calls=2*model.sampler.N,
        sampler='explicit latent Gaussian PC; per-crystal expected-noise corrector, SPD projection, std floors0.001')


def structures(state):
    offset = 0; rows = []
    for index, n in enumerate(state['num_atoms'].tolist()):
        count = int(n)
        rows.append(dict(index=index, numbers=state['atomic_numbers'][offset:offset+count].numpy(),
            frac=state['pos'][offset:offset+count].double().numpy(), cell=state['cell'][index].double().numpy()))
        offset += count
    return rows


def reward_value(gap, valid=True):
    if not valid or not np.isfinite(gap): return -1.
    return float(np.exp(-.5*((float(gap)-2.5)/.5)**2))


class Evaluator:
    def __init__(self, directory):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'forward_inverse'))
        from property_model import Properties
        directory.mkdir(parents=True, exist_ok=False)
        self.properties = Properties(directory)
        self.controls = 0

    def __call__(self, row):
        from ase import Atoms
        from ase.neighborlist import neighbor_list
        numbers = np.asarray(row['numbers']); cell = np.asarray(row['cell']); frac = np.asarray(row['frac'])
        event = dict(index=row['index'], valid=False, gap_eV=None, eform_eV_atom=None, reward=-1.)
        try:
            if not np.isfinite(cell).all() or not np.isfinite(frac).all() or not np.all((numbers >= 1) & (numbers <= 98)):
                raise ValueError('Invalid geometry, MASK or unsupported ALIGNN species.')
            volume = float(np.linalg.det(cell))/len(numbers)
            if not 1. < volume < 200.:
                raise ValueError('Volume outside1..200A3/atom.')
            atoms = Atoms(numbers=numbers, cell=cell, scaled_positions=frac, pbc=True)
            if len(neighbor_list('d', atoms, .6)):
                raise ValueError('Periodic interatomic separation below0.6A.')
            p = self.properties.predictor
            c = torch.tensor(cell[None], device='cuda', dtype=torch.float64)
            q = torch.tensor(frac, device='cuda', dtype=torch.float64)
            onehot = torch.nn.functional.one_hot(torch.tensor(numbers, device='cuda').long()-1, 98).double()
            sizes = torch.tensor([len(numbers)], device='cuda')
            with torch.no_grad():
                graphs = p.graphs(c, q, onehot, sizes)
                output = p.predict(c, q, onehot, sizes, graphs=graphs, native_control=self.controls == 0)
            self.controls += 1
            gap, eform = (float(v.item()) for v in output)
            if not np.isfinite([gap, eform]).all(): raise ValueError('Nonfinite property output.')
            event.update(valid=True, gap_eV=gap, eform_eV_atom=eform, reward=reward_value(gap),
                error_eV=abs(gap-2.5), pass_gap=abs(gap-2.5) <= .1+1e-12, volume_A3_atom=volume)
        except (ValueError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            event['error'] = str(error)
        return event
