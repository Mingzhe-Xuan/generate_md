"""Native-only proof for differentiating CF loss through MatterGen denoising."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from adapter import Denoiser, sha, score_with_raw
from mapping import LatentGeometry, clean_estimate

DIRECTED = Path(__file__).resolve().parents[1]/'directed_tto'
sys.path.insert(0, str(DIRECTED))
from cf_gradient import GradientScorer
from numerics import promote

POLICY = dict(parents=4, times=[.1, .3], seed=20261032, target_gap=2.5,
    CF_control_tolerance=1e-5, precision_forward_tolerance=1e-5,
    precision_gradient_relative_tolerance=.001, FD_absolute=.001, FD_relative=.02,
    FD_steps=[1e-5, 1e-6], CFG=2., search_trials=0,
    purpose='Native feasibility only; no claim of post-relaxation efficacy.')


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def evaluate(denoiser, scorer, geometry, time, q, dtype, backward=False, stop=False):
    torch = scorer.torch
    with torch.enable_grad() if backward else torch.no_grad():
        variable = torch.tensor(np.asarray(q), dtype=dtype, device='cuda', requires_grad=backward)
        graph = denoiser.graph(geometry, variable)
        t = torch.full((1,), time, dtype=dtype, device='cuda')
        clean = denoiser.decode(graph, t, stop_score_gradient=stop)
        if not torch.equal(clean.atomic_numbers, graph.atomic_numbers):
            raise ValueError('Denoising changed elemental identities.')
        if not all(x.is_cuda and torch.isfinite(x).all() for x in (clean.cell, clean.pos)):
            raise ValueError('Nonfinite or CPU clean geometry.')
        if torch.linalg.det(clean.cell)[0] <= 0:
            raise ValueError('Denoising produced a nonpositive cell determinant.')
        atomic = torch.nn.functional.one_hot(clean.atomic_numbers-1, num_classes=98).to(dtype)
        with scorer.operators(backward):
            gap, ef = scorer.author.Crystalformer_prediction(*scorer.models, atomic,
                clean.pos@clean.cell[0], graph.batch, clean.cell, graph.num_atoms)
            loss = .5*(gap.reshape(-1)[0]-2.5)**2
            gradient = torch.autograd.grad(loss, variable)[0] if backward else None
        if not all(x.is_cuda and torch.isfinite(x).all() for x in (gap, ef, loss)):
            raise ValueError('Nonfinite or CPU composed CF score.')
        if gradient is not None and (not gradient.is_cuda or not torch.isfinite(gradient).all()):
            raise ValueError('Invalid CF-through-denoiser gradient.')
        if backward and scorer.gaussian.calls <= 0:
            raise RuntimeError('Corrected CF Gaussian backward was not executed.')
    scorer.metadata['CF_calls'] += 1
    if backward:
        scorer.metadata['CF_backward_calls'] += 1
    return dict(gap=float(gap.detach().reshape(-1)[0]), eform=float(ef.detach().reshape(-1)[0]),
        loss=float(loss.detach()), gradient=gradient.detach().cpu().double().numpy() if backward else None,
        clean_cell=clean.cell.detach().cpu().double().numpy().tolist(),
        clean_frac=clean.pos.detach().cpu().double().numpy().tolist())


def conversion_check(denoiser, geometry, time):
    torch = denoiser.torch
    zero = torch.zeros(geometry.size, dtype=torch.float32, device='cuda')
    graph = denoiser.graph(geometry, zero)
    t = torch.tensor([time], dtype=zero.dtype, device=zero.device)
    score, raw = score_with_raw(denoiser.module, graph, t)
    errors = {}
    for field in ('pos', 'cell'):
        sde = denoiser.module.corruption.sdes[field]
        if denoiser.module.model_targets[field].value != 'score_times_std':
            raise ValueError('Unexpected official model prediction target.')
        _, std = sde.marginal_prob(graph[field], t, graph.get_batch_idx(field), graph)
        error = float((score[field]-raw[field]/std).abs().max())
        if error > 1e-5:
            raise ValueError('Explicit official score conversion mismatch: '+repr(errors | {field: error}))
        errors[field] = error
    cell_sde = denoiser.module.corruption.sdes['cell']
    mean, std = cell_sde.marginal_prob(graph.cell, t, None, graph)
    coefficient = cell_sde._marginal_mean_coeff(t)[:, None, None]
    limit = cell_sde.get_limit_mean(graph.cell, graph)
    ideal = clean_estimate(mean+std*.3, -.3/std, std, coefficient, limit)
    roundtrip = float((ideal-graph.cell).abs().max())
    if roundtrip > 1e-5:
        raise ValueError('Cell affine VP inversion failed.')
    return dict(explicit_score_errors=errors, ideal_VP_roundtrip_error=roundtrip)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Native model proof must run under Slurm.')
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = dict(policy=POLICY, cases=[], passed=False, source=str(args.source.resolve()),
        implementation_sha256={q.name: sha(q) for q in Path(__file__).parent.glob('*.py')})
    try:
        from pymatgen.core import Structure
        source = args.source.resolve()
        plan = json.loads((source/'plan.json').read_text())
        if len(plan['cases']) != 16 or not all(q['neutral'] for q in plan['cases']):
            raise ValueError('Original neutral16 cohort required.')
        if not json.loads((source/'native-probe.json').read_text())['passed']:
            raise ValueError('Original source native controls failed.')
        result['source_hashes'] = {n: sha(source/n) for n in ('plan.json', 'native-probe.json', 'direction-results.json', 'report.json')}
        denoiser = Denoiser()
        scorer = GradientScorer()
        torch = scorer.torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(2)
        result['device'] = torch.cuda.get_device_name()
        result['MatterGen'] = denoiser.metadata
        result['CF'] = scorer.metadata
        operators = []
        for case in plan['cases'][:POLICY['parents']]:
            path = Path(case['B']['relaxed_path'])
            if sha(path) != case['B']['relaxed_sha256']:
                raise ValueError('Source baseline pose changed.')
            s = Structure.from_file(path)
            control = scorer.predict(path)
            old = case['predictions']['after']
            error = max(abs(control[0]-old['crystalformer_gap']), abs(control[1]-old['crystalformer_eform']))
            if error > POLICY['CF_control_tolerance']:
                raise ValueError('Original frozen CF control failed.')
            geom = LatentGeometry(s.atomic_numbers, s.lattice.matrix, s.frac_coords)
            if not np.allclose(geom.cell@geom.cell.T, s.lattice.matrix@s.lattice.matrix.T, atol=1e-9, rtol=1e-9):
                raise ValueError('Symmetric rotation changed lattice metric.')
            for time in POLICY['times']:
                zero = torch.zeros(geom.size, device='cuda')
                with torch.no_grad():
                    graph = denoiser.graph(geom, zero)
                    native = conversion_check(denoiser, geom, time)
                    index = int(Path(case['parent_id']).stem.split('-')[1])
                    noisy, _ = denoiser.forward_noise(graph, time, [POLICY['seed'], index, int(time*1000)])
                latent = LatentGeometry(geom.numbers, noisy.cell[0].cpu().double().numpy(), noisy.pos.cpu().double().numpy())
                operators.append((case['parent_id'], path, latent, time, error, native))
        first = {}
        for precision, dtype in [('float32', torch.float32), ('float64', torch.float64)]:
            denoiser.module.to(dtype=dtype)
            promote(scorer, dtype)
            for parent, path, latent, time, control_error, native in operators:
                q = np.zeros(latent.size)
                ad = evaluate(denoiser, scorer, latent, time, q, dtype, True)
                direct = evaluate(denoiser, scorer, latent, time, q, dtype)
                gradient = latent.project(ad.pop('gradient'))
                if np.linalg.norm(gradient) <= 1e-12:
                    raise ValueError('Stationary composed gradient in native proof.')
                row = dict(parent_id=parent, source_pose_sha256=sha(path), precision=precision, time=time,
                    source_CF_error=control_error, conversion=native, score=ad,
                    corrected_vs_native_forward=max(abs(ad[k]-direct[k]) for k in ('gap', 'eform')),
                    gradient_norm=float(np.linalg.norm(gradient)), checks=[])
                if row['corrected_vs_native_forward'] > POLICY['CF_control_tolerance']:
                    raise ValueError('Corrected Gaussian changed composed forward.')
                key = (parent, time)
                if precision == 'float32':
                    first[key] = (gradient, ad)
                else:
                    g32, a32 = first[key]
                    row['float32_gradient_relative_error'] = float(np.linalg.norm(gradient-g32)/np.linalg.norm(gradient))
                    row['float32_forward_error'] = max(abs(a32[k]-ad[k]) for k in ('gap', 'eform'))
                    if row['float32_gradient_relative_error'] > POLICY['precision_gradient_relative_tolerance'] or row['float32_forward_error'] > POLICY['precision_forward_tolerance']:
                        result['cases'].append(row)
                        raise ValueError('Composed float32/float64 control failed.')
                    rng = np.random.default_rng([POLICY['seed'], int(parent.split('-')[1].split('.')[0]), int(time*1000), 77])
                    for block in ('atomic', 'strain', 'both'):
                        direction = latent.project(rng.normal(size=latent.size))
                        if block == 'atomic':
                            direction[-6:] = 0
                        elif block == 'strain':
                            direction[:-6] = 0
                        direction /= np.linalg.norm(direction)
                        predicted = float(gradient@direction)
                        for epsilon in POLICY['FD_steps']:
                            plus = evaluate(denoiser, scorer, latent, time, q+epsilon*direction, dtype)['loss']
                            minus = evaluate(denoiser, scorer, latent, time, q-epsilon*direction, dtype)['loss']
                            fd = (plus-minus)/(2*epsilon)
                            passed = bool(abs(fd-predicted) <= POLICY['FD_absolute']+POLICY['FD_relative']*max(abs(fd), abs(predicted)))
                            row['checks'].append(dict(block=block, epsilon=epsilon, FD=fd, predicted=predicted, passed=passed))
                    if not all(q['passed'] for q in row['checks']):
                        result['cases'].append(row)
                        raise ValueError('CF-through-denoiser finite difference failed.')
                result['cases'].append(row)
                save(root/'native.json', result)
                print(json.dumps(dict(parent=parent, time=time, precision=precision, gradient_norm=row['gradient_norm'])), flush=True)
        if len(result['cases']) != 16 or sum(len(q['checks']) for q in result['cases']) != 48:
            raise ValueError('Native proof coverage incomplete.')
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        save(root/'native.json', result)


if __name__ == '__main__':
    main()
