"""Trace graph/decoder/CF discontinuities without relaxing a failed native gate."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from adapter import Denoiser, sha
from mapping import LatentGeometry
from native import GradientScorer, promote, evaluate, POLICY, save


def graph_identity(edges, images):
    edges = np.asarray(edges, dtype=np.int64)
    images = np.asarray(images)
    integer = np.rint(images).astype(np.int64)
    if edges.shape[0] != 2 or images.shape != (edges.shape[1], 3) or not np.allclose(images, integer, atol=1e-6):
        raise ValueError('Invalid periodic graph identity.')
    rows = np.concatenate([edges.T, integer], axis=1)
    canonical = sorted(map(tuple, rows.tolist()))
    digest = lambda value: hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()
    return dict(edges=len(rows), ordered=digest(rows.tolist()), canonical=digest(canonical))


@contextmanager
def graph_trace(gemnet, records):
    name = 'generate_interaction_graph'
    existed = name in gemnet.__dict__
    original = getattr(gemnet, name)
    def traced(*args, **kwargs):
        result = original(*args, **kwargs)
        records.append(graph_identity(result[0].detach().cpu().numpy(), result[8].detach().cpu().numpy()))
        return result
    setattr(gemnet, name, traced)
    try:
        yield
    finally:
        if existed:
            setattr(gemnet, name, original)
        else:
            delattr(gemnet, name)


@contextmanager
def frozen_score(denoiser, latent, time, dtype):
    torch = denoiser.torch
    zero = torch.zeros(latent.size, dtype=dtype, device='cuda')
    with torch.no_grad(), denoiser.precision_context(dtype):
        graph = denoiser.graph(latent, zero)
        t = torch.full((1,), time, dtype=dtype, device='cuda')
        value = denoiser.scores(graph, t)
        pos, cell = value.pos.detach(), value.cell.detach()
    original = denoiser.scores
    denoiser.scores = lambda g, t: g.replace(pos=pos, cell=cell)
    try:
        yield
    finally:
        denoiser.scores = original


def traced_score(denoiser, scorer, latent, time, q, backward=False):
    trace = []
    with graph_trace(denoiser.module.model.gemnet, trace):
        result = evaluate(denoiser, scorer, latent, time, q, scorer.torch.float64, backward)
    result['graphs'] = trace
    return result


def jump(a, b):
    cell_a, cell_b = np.array(a['clean_cell'])[0], np.array(b['clean_cell'])[0]
    delta = np.array(a['clean_frac'])-np.array(b['clean_frac'])
    delta -= np.rint(delta)
    return dict(gap_difference=a['gap']-b['gap'], cell_max_difference=float(np.abs(cell_a-cell_b).max()),
        internal_RMS_A=float(np.sqrt(np.square(delta@cell_b).sum()/len(delta))),
        canonical_graph_equal=[q['canonical'] for q in a['graphs']]==[q['canonical'] for q in b['graphs']],
        ordered_graph_equal=[q['ordered'] for q in a['graphs']]==[q['ordered'] for q in b['graphs']])


def operator_diagnostic(denoiser, scorer, latent, parent, time):
    q = np.zeros(latent.size)
    baseline = traced_score(denoiser, scorer, latent, time, q, True)
    gradient = latent.project(baseline.pop('gradient'))
    rng = np.random.default_rng([POLICY['seed'], int(parent.split('-')[1].split('.')[0]), int(time*1000), 77])
    directions = {}
    for block in ('atomic', 'strain', 'both'):
        d = latent.project(rng.normal(size=latent.size))
        if block == 'atomic':
            d[-6:] = 0
        elif block == 'strain':
            d[:-6] = 0
        directions[block] = d/np.linalg.norm(d)
    records = []
    def scan(mode, g):
        for block, d in directions.items():
            for epsilon in (1e-4, 1e-5, 1e-6, 1e-7):
                plus = traced_score(denoiser, scorer, latent, time, epsilon*d)
                minus = traced_score(denoiser, scorer, latent, time, -epsilon*d)
                fd = (plus['loss']-minus['loss'])/(2*epsilon)
                records.append(dict(mode=mode, block=block, epsilon=epsilon, predicted=float(g@d), FD=fd,
                    plus=plus, minus=minus, jump=jump(plus, minus)))
    scan('full', gradient)
    with frozen_score(denoiser, latent, time, scorer.torch.float64):
        fixed = traced_score(denoiser, scorer, latent, time, q, True)
        fixed_gradient = latent.project(fixed.pop('gradient'))
        scan('frozen_score', fixed_gradient)
    return dict(parent_id=parent, time=time, baseline=baseline, fixed_baseline=fixed,
        full_gradient=gradient.tolist(), fixed_gradient=fixed_gradient.tolist(), checks=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--failed', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use a GPU Slurm job.')
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=False)
    failed = json.loads((args.failed/'native.json').read_text())
    if failed['passed'] or failed.get('error') != "ValueError('CF-through-denoiser finite difference failed.')":
        raise ValueError('Expected retained failed derivative proof.')
    for name, expected in failed['source_hashes'].items():
        if sha(args.source/name) != expected:
            raise ValueError('Original source changed.')
    output = dict(role='Component diagnostic only; original native gate stays failed.',
        native_passed=False, goal_completion_proven=False, source_hashes=failed['source_hashes'],
        failed_native_sha256=sha(args.failed/'native.json'), cases=[], completed=False,
        implementation_sha256={q.name: sha(q) for q in Path(__file__).parent.glob('*.py')})
    try:
        from pymatgen.core import Structure
        denoiser = Denoiser()
        scorer = GradientScorer()
        torch = scorer.torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(2)
        plan = json.loads((args.source/'plan.json').read_text())
        operators = []
        for case in plan['cases'][:4]:
            if sha(case['B']['relaxed_path']) != case['B']['relaxed_sha256']:
                raise ValueError('Original baseline changed.')
            s = Structure.from_file(case['B']['relaxed_path'])
            geometry = LatentGeometry(s.atomic_numbers, s.lattice.matrix, s.frac_coords)
            for time in POLICY['times']:
                index = int(case['parent_id'].split('-')[1].split('.')[0])
                with torch.no_grad():
                    graph = denoiser.graph(geometry, torch.zeros(geometry.size, device='cuda'))
                    noisy, _ = denoiser.forward_noise(graph, time, [POLICY['seed'], index, int(time*1000)])
                latent = LatentGeometry(geometry.numbers, noisy.cell[0].cpu().double().numpy(), noisy.pos.cpu().double().numpy())
                operators.append((case['parent_id'], time, latent))
        denoiser.module.double()
        promote(scorer, torch.float64)
        output.update(MatterGen=denoiser.metadata, CF=scorer.metadata)
        for parent, time, latent in operators:
            record = operator_diagnostic(denoiser, scorer, latent, parent, time)
            output['cases'].append(record)
            save(root/'diagnostic.json', output)
            print(json.dumps(dict(parent=parent, time=time, graph_switch_checks=sum(not r['jump']['canonical_graph_equal'] for r in record['checks'] if r['mode']=='full'))), flush=True)
        output['completed'] = True
    finally:
        save(root/'diagnostic.json', output)


if __name__ == '__main__':
    main()
