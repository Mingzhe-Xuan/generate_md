"""Frozen16 low-noise, fixed-cell/species MatterGen score diagnostic, not TTO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from probe_math import project, unit, cosine, local_projection

POLICY = dict(seed=20261049, target_gap_eV=2.5, parents=16, anchors=['raw', 'relaxed'],
              times=[.05, .15, .3], replicas=2, fixed_species=True, fixed_cell=True,
              noise='Official positional marginal; no lattice or type corruption; no redraw/rejection.',
              gradient='negative ALIGNN half-squared target-gap loss gradient; F64 GPU',
              FD_steps_A=[1e-4, 5e-5], FD_abs=1e-5, FD_rel=.002,
              physical_FD_steps_A=[.002, .001], physical_FD_abs=.005, physical_FD_rel=.02,
              native_parents=4, native_time=.15, min_vector_norm=1e-10,
              regression='Unit vectors, no intercept, equal state weight, leave-one-parent-out per anchor/time.',
              purpose='Mechanistic probe only; no sampling modification or post-relaxation efficacy claim.')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False, default=default)+'\n')


def topology(graphs):
    g, _, images = graphs[0]
    a, b = g.edges(order='eid')
    return sorted((i, j, *v) for i, j, v in zip(a.tolist(), b.tolist(), images.long().tolist()))


def close(a, b, absolute, relative):
    return bool(abs(a-b) <= absolute+relative*max(abs(a), abs(b)))


def gradient_checks(models, numbers, cell, cart, value, physical, index):
    rng = np.random.default_rng([POLICY['seed'], index, 777])
    v = unit(project(rng.normal(size=cart.shape))).reshape(cart.shape)
    expected = float(-value['G']@v.ravel())
    records = []
    reference_topology = topology(value['graphs'])
    for eps in POLICY['FD_steps_A']:
        plus = models.property(numbers, cell, cart+eps*v, graphs=value['graphs'])
        minus = models.property(numbers, cell, cart-eps*v, graphs=value['graphs'])
        finite = (plus['loss']-minus['loss'])/(2*eps)
        rec = dict(kind='ALIGNN_fixed_topology', epsilon_A=eps, predicted=expected, FD=finite,
                   passed=close(finite, expected, POLICY['FD_abs'], POLICY['FD_rel']))
        records.append(rec)
        if not rec['passed']:
            raise ValueError('ALIGNN fixed-topology finite difference failed: '+repr(rec))
        freshp = models.property(numbers, cell, cart+eps*v)
        freshm = models.property(numbers, cell, cart-eps*v)
        unchanged = topology(freshp['graphs']) == reference_topology == topology(freshm['graphs'])
        fresh_fd = (freshp['loss']-freshm['loss'])/(2*eps)
        records.append(dict(kind='ALIGNN_rebuilt_topology', epsilon_A=eps, FD=fresh_fd,
                            predicted=expected, same_topology=unchanged,
                            passed=close(fresh_fd, expected, POLICY['FD_abs'], POLICY['FD_rel']) if unchanged else None))
        if unchanged and not records[-1]['passed']:
            raise ValueError('ALIGNN fresh-graph FD failed without topology change.')
    expected_energy = float(-project(physical['forces_eV_A'])@v.ravel())
    for eps in POLICY['physical_FD_steps_A']:
        plus, minus = models.physical(numbers, cell, [cart+eps*v, cart-eps*v])
        finite = (plus['energy_eV']-minus['energy_eV'])/(2*eps)
        rec = dict(kind='MatterSim_force', epsilon_A=eps, FD=finite, predicted=expected_energy,
                   passed=close(finite, expected_energy, POLICY['physical_FD_abs'], POLICY['physical_FD_rel']))
        records.append(rec)
        if not rec['passed']:
            raise ValueError('MatterSim energy-force control failed: '+repr(rec))
    return records


def run(args):
    from pymatgen.core import Structure
    from ase.geometry import find_mic
    from probe_models import Models
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.source/'plan.json').read_text())
    cases = source['cases']
    if len(cases) != POLICY['parents'] or len({c['parent_id'] for c in cases}) != POLICY['parents']:
        raise ValueError('Expected unchanged property-blind16 cohort.')
    inputs = []
    for case in cases:
        for anchor in POLICY['anchors']:
            path = Path(case['raw_path'] if anchor == 'raw' else case['B']['relaxed_path'])
            expected = case['inventory']['sha256'] if anchor == 'raw' else case['B']['relaxed_sha256']
            if sha(path) != expected:
                raise ValueError('Immutable source geometry changed: '+str(path))
            inputs.append(dict(parent_id=case['parent_id'], anchor=anchor, path=str(path), sha256=expected))
    code = {str(p): sha(p) for p in sorted(Path(__file__).parent.glob('*')) if p.suffix in ('.py', '.sbatch', '.md')}
    for folder, files in [('denoised_tto', ['adapter.py', 'mapping.py']),
                          ('alignn_templates', ['model.py', 'geometry.py']),
                          ('directed_tto', ['runtime.py', 'mechanics.py'])]:
        for name in files:
            path = Path(__file__).resolve().parents[1]/folder/name
            code[str(path)] = sha(path)
    save(root/'plan.json', dict(policy=POLICY, inputs=inputs, source_plan_sha256=sha(args.source/'plan.json'),
                               source_selection=source['selection'], code_sha256=code,
                               commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    models = None
    records, checks = [], []
    started = time.perf_counter()
    try:
        models = Models(root)
        torch = models.torch
        for anchor_index, entry in enumerate(inputs):
            structure = Structure.from_file(entry['path'])
            geom = models.geometry(structure.atomic_numbers, structure.lattice.matrix, structure.frac_coords)
            graph = models.denoiser.graph(geom, torch.zeros(geom.size, device='cuda'))
            cell = graph.cell[0].cpu().double().numpy()
            numbers = geom.numbers
            reference_cart = graph.pos.cpu().double().numpy()@cell
            for ti, t in enumerate(POLICY['times']):
                time_t = torch.tensor([t], device='cuda')
                _, std = models.denoiser.module.corruption.sdes['pos'].marginal_prob(graph.pos, time_t, graph.batch, graph)
                for replica in range(POLICY['replicas']):
                    index = len(records)
                    rng = np.random.default_rng([POLICY['seed'], int(entry['parent_id'].split('-')[1].split('.')[0]), ti, replica])
                    noise = torch.tensor(rng.normal(size=tuple(graph.pos.shape)), device='cuda', dtype=graph.pos.dtype)
                    mean, _ = models.denoiser.module.corruption.sdes['pos'].marginal_prob(graph.pos, time_t, graph.batch, graph)
                    noisy = graph.replace(pos=(mean+std*noise).remainder(1.))
                    cart = noisy.pos.cpu().double().numpy()@cell
                    scores, conversion = models.branches(noisy, time_t)
                    native_state = anchor_index//2 < POLICY['native_parents'] and entry['anchor'] == 'raw' and t == POLICY['native_time'] and replica == 0
                    value = models.property(numbers, cell, cart, gradient=True, native=native_state)
                    point = models.physical(numbers, cell, [cart])[0]
                    F = project(point['forces_eV_A'])
                    G = value['G']
                    restore = project(-(std*noise).cpu().double().numpy()@cell)
                    native = []
                    if native_state:
                        native = gradient_checks(models, numbers, cell, cart, value, point, index)
                        checks.append(dict(state=index, checks=native))
                    delta = find_mic(cart-reference_cart, cell, pbc=True)[0]
                    basename = f'{Path(entry["parent_id"]).stem}-{entry["anchor"]}-t{ti}-r{replica}'
                    path = root/(basename+'.npz')
                    np.savez_compressed(path, numbers=numbers, cell=cell, cart=cart, F=F, G=G, restoration=restore, **scores)
                    row = dict(parent_id=entry['parent_id'], anchor=entry['anchor'], t=t, replica=replica,
                               vector_path=path.name, vector_sha256=sha(path), num_atoms=len(numbers),
                               noise_RMS_A=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
                               fractional_noise_std=float(std.flatten()[0]), ALIGNN_gap=value['gap'], ALIGNN_eform=value['eform'],
                               ALIGNN_loss=value['loss'], MatterSim_energy_eV=point['energy_eV'],
                               norms={k:float(np.linalg.norm(v)) for k, v in dict(F=F,G=G,**scores).items()},
                               cos_F_G=cosine(F, G),
                               cosine={k:dict(force=cosine(v,F),property=cosine(v,G),restoration=cosine(v,restore)) for k,v in scores.items()},
                               local_projection={k:local_projection(v,F,G) for k,v in scores.items() if np.linalg.norm(v)>1e-10},
                               conversion=conversion, native_checks=native,
                               elapsed_seconds=time.perf_counter()-started)
                    # Local directional check evaluates score direction at the SAME physical geometry.
                    if native_state:
                        micro=[]
                        for name, direction in dict(force=F, property=G, delta=scores['delta'], unconditional=scores['unconditional']).items():
                            d=unit(direction).reshape(cart.shape)
                            eps=1e-4
                            moved=models.property(numbers,cell,cart+eps*d)
                            physical=models.physical(numbers,cell,[cart+eps*d])[0]
                            micro.append(dict(direction=name, step_global_norm_A=eps,
                                              predicted_dE=-float(F@d.ravel()), actual_dE=(physical['energy_eV']-point['energy_eV'])/eps,
                                              predicted_dLoss=-float(G@d.ravel()), actual_dLoss=(moved['loss']-value['loss'])/eps,
                                              topology_unchanged=topology(moved['graphs'])==topology(value['graphs'])))
                        row['microsteps']=micro
                    records.append(row)
                    save(root/'progress.json', dict(policy=POLICY, states=records, native_checks=checks, NFE=models.mechanics.NFE))
                    print(json.dumps(dict(state=index+1,parent=entry['parent_id'],anchor=entry['anchor'],t=t,replica=replica,
                                          noise_RMS_A=row['noise_RMS_A'],gap=value['gap'],cos_delta_G=cosine(scores['delta'],G))),flush=True)
        if len(records) != 192:
            raise ValueError('Incomplete probe.')
        save(root/'complete.json', dict(passed=True, policy=POLICY, states=records, native_checks=checks,
                                       MatterGen=models.denoiser.metadata, ALIGNN=models.alignn_metadata,
                                       ALIGNN_native=models.alignn.native_controls, mechanics=models.mechanics.ready,
                                       NFE=models.mechanics.NFE, elapsed_seconds=time.perf_counter()-started,
                                       device=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__,
                                       peak_GPU_bytes=torch.cuda.max_memory_allocated()))
    except Exception as error:
        save(root/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), completed_states=len(records)))
        raise
    finally:
        if models is not None:
            models.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('All model compute must run under Slurm.')
    run(args)
