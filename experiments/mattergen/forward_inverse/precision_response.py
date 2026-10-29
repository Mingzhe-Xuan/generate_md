"""Same-pose precision and step-size response; never substitute FP64 for native."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from csp_model import CSP, sha
from relations import projected_adjoint_jacobian, relative_error


STEPS = [.000125, .00025, .0005, .001, .002, .004, .008, .016, .032]


def response_summary(jacobian, v, w, plus, minus, steps, same_graph):
    target = jacobian @ v
    finite = np.array([(p.astype(float) - m.astype(float)) / (2 * h)
                       for p, m, h in zip(plus, minus, steps)])
    return [dict(step_A=h, same_graph=bool(graph),
                 vector_AD_FD_relative_error=relative_error(target, estimate),
                 scalar_AD=float(w @ target), scalar_FD=float(w @ estimate),
                 FD_norm=float(np.linalg.norm(estimate)), AD_norm=float(np.linalg.norm(target)))
            for h, graph, estimate in zip(steps, same_graph, finite)]


def execute(run, output):
    if output.exists():raise ValueError('Preserve previous precision runs.')
    output.mkdir(parents=True)
    plan = json.loads((run / 'plan.json').read_text())
    complete = json.loads((run / 'complete.json').read_text())
    if not complete['passed']:raise ValueError('Original execution incomplete.')
    model = CSP();started = time.perf_counter()
    report = dict(original_plan_sha256=sha(run / 'plan.json'),
                  original_complete_sha256=sha(run / 'complete.json'),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  script_sha256=sha(Path(__file__)), steps_A=STEPS,
                  fp64_equivalence_tolerance=.005, cases=[], model=model.metadata,
                  interpretation='Native and FP64 separately; numerical failure never establishes physical-law failure.')
    for row in complete['cases']:
        if row['status'] != 'completed':raise ValueError('Missing original case.')
        folder = run / row['id']
        if sha(folder / 'vectors.npz') != row['vectors_sha256']:raise ValueError('Original vectors changed.')
        with np.load(folder / 'vectors.npz') as z, np.load(folder / 'force-data.npz') as f:
            numbers, cell, cart, q, v, w = z['numbers'], z['cell'], z['probe'], z['Q'], f['v'], f['w']
            case = dict(id=row['id'], system=row['system'], vectors_sha256=row['vectors_sha256'], times=[])
            for j, details in enumerate(row['times']):
                t = details['t'];raw = {};summaries = {};centers = {};matrices = {}
                for dtype in ('single', 'double'):
                    field = lambda coordinates, **kw: model.score(numbers, cell, coordinates, t, dtype=dtype, **kw)
                    base = field(cart);repeat = field(cart)
                    adjoints = [field(cart, adjoint=q[:, k]) for k in range(q.shape[1])]
                    matrix = projected_adjoint_jacobian(q, [a['adjoint'] for a in adjoints])
                    direction_checks, direction_graphs, finite_vectors = {}, {}, {}
                    for name, direction, covector in [('v', v, w), ('w', w, v)]:
                        plus, minus, graphs = [], [], []
                        for step in STEPS:
                            p = field(cart + step * (q @ direction).reshape(-1, 3))
                            m = field(cart - step * (q @ direction).reshape(-1, 3))
                            plus.append(p['score']);minus.append(m['score'])
                            graphs.append([p['graph_hash'], m['graph_hash']])
                        projected_plus, projected_minus = np.asarray(plus) @ q, np.asarray(minus) @ q
                        direction_checks[name] = response_summary(matrix, direction, covector,
                            projected_plus, projected_minus, STEPS,
                            [a == b == base['graph_hash'] for a, b in graphs])
                        finite_vectors[name] = (projected_plus.astype(float)-projected_minus.astype(float))/(2*np.array(STEPS)[:, None])
                        direction_graphs[name] = graphs
                        raw[dtype + '_' + name + '_plus'] = plus
                        raw[dtype + '_' + name + '_minus'] = minus
                    curl = float(w @ matrix @ v - v @ matrix @ w)
                    curl_fd = finite_vectors['v'] @ w - finite_vectors['w'] @ v
                    summaries[dtype] = dict(directions=direction_checks, replay_relative_error=relative_error(base['score'], repeat['score']),
                        original_native_J_relative_error=relative_error(matrix, z['score_jacobian_ad'][j]) if dtype == 'single' else None,
                        curl_AD=curl, curl_FD=curl_fd.tolist(),
                        base_graph=base['graph_hash'], pair_graphs=direction_graphs,
                        adjoint_graphs=[a['graph_hash'] for a in adjoints])
                    raw.update({dtype + '_score': base['score'], dtype + '_repeat': repeat['score'],
                                dtype + '_adjoints': np.stack([a['adjoint'] for a in adjoints]),
                                dtype + '_J': matrix})
                    centers[dtype] = base['score'];matrices[dtype] = matrix
                filename = row['id'] + '-t' + str(t) + '.npz'
                np.savez_compressed(output / filename, Q=q, v=v, w=w, numbers=numbers, cell=cell, cart=cart, **raw)
                value_error = relative_error(centers['single'], centers['double'])
                case['times'].append(dict(t=t, dtype=summaries, vectors=filename, vectors_sha256=sha(output / filename),
                    fp32_fp64_value_relative_error=value_error, fp64_value_equivalence_passed=value_error <= .005,
                    fp32_fp64_J_relative_error=relative_error(matrices['single'], matrices['double'])))
                print('RESPONSE', row['id'], t, value_error, flush=True)
            report['cases'].append(case)
            (output / 'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    report.update(seconds=time.perf_counter()-started, execution_complete=True)
    (output / 'complete.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    execute(args.run, args.output)


if __name__ == '__main__':main()
