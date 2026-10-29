"""Independently audit two-direction native/FP64 response and report local curl."""
import argparse
import json
from pathlib import Path
import numpy as np
from csp_model import sha
from relations import cartesian_noise_covariance, matrix_diagnostics, relative_error


WINDOW = [.0005, .001]


def curl_evidence(matrix, v, w, finite_v, finite_w, graph_ok):
    ad = float(w @ matrix @ v - v @ matrix @ w)
    fd = finite_v @ w - finite_w @ v
    errors = [max(relative_error(matrix @ v, a), relative_error(matrix @ w, b))
              for a, b in zip(finite_v, finite_w)]
    uncertainty = float(max(np.max(np.abs(fd-ad)), np.ptp(fd), 1e-8))
    numerical = bool(all(graph_ok) and max(errors) <= .02)
    return dict(curl_AD=ad, curl_FD=fd.tolist(), vector_relative_errors=errors,
                FD_discrepancy_scale=uncertainty, numerical_direction_QC=numerical,
                corroborated_nonzero_curl=bool(numerical and abs(ad) > 5*uncertainty and
                                               np.all(fd*ad > 0)))


def execute(root, pilot, output):
    if output.exists():raise ValueError('Use a new report directory.')
    source = json.loads((root / 'complete.json').read_text())
    original = json.loads((pilot / 'plan.json').read_text())
    if sha(pilot / 'plan.json') != source['original_plan_sha256'] or sha(pilot / 'complete.json') != source['original_complete_sha256']:
        raise ValueError('Original experiment manifests changed.')
    if not source['execution_complete']:raise ValueError('Response experiment incomplete.')
    rows = [];audits = [];plot_rows = []
    for case in source['cases']:
        for time in case['times']:
            path = root / time['vectors']
            if sha(path) != time['vectors_sha256']:raise ValueError('Response vector hash mismatch.')
            row = dict(id=case['id'], system=case['system'], t=time['t'],
                       fp32_fp64_value_relative_error=time['fp32_fp64_value_relative_error'],
                       fp64_value_equivalence_passed=time['fp64_value_equivalence_passed'],
                       dtype={})
            with np.load(path) as z:
                q, v, w = z['Q'], z['v'], z['w']
                for dtype in ('single', 'double'):
                    matrix = z[dtype + '_adjoints'] @ q
                    audits.append(relative_error(matrix, z[dtype + '_J']))
                    finite = {}
                    for name, direction in [('v', v), ('w', w)]:
                        delta = z[dtype + '_' + name + '_plus'].astype(float)-z[dtype + '_' + name + '_minus'].astype(float)
                        finite[name] = (delta @ q) / (2*np.asarray(source['steps_A'])[:, None])
                        for j, details in enumerate(time['dtype'][dtype]['directions'][name]):
                            error = relative_error(matrix @ direction, finite[name][j])
                            if abs(error-details['vector_AD_FD_relative_error']) > 1e-10:
                                raise ValueError('Response summary differs from raw vectors.')
                    indices = [source['steps_A'].index(h) for h in WINDOW]
                    meta = time['dtype'][dtype]
                    graphs = meta['pair_graphs']
                    graph_ok = [all(graphs[name][j][0] == graphs[name][j][1] == meta['base_graph'] for name in ('v', 'w')) for j in indices]
                    evidence = curl_evidence(matrix, v, w, finite['v'][indices], finite['w'][indices], graph_ok)
                    evidence.update(jacobian=matrix_diagnostics(matrix), replay_relative_error=meta['replay_relative_error'],
                                    original_native_J_relative_error=meta['original_native_J_relative_error'])
                    if meta['replay_relative_error'] > 1e-5:evidence['corroborated_nonzero_curl'] = False
                    with np.load(pilot / case['id'] / f"time-{time['t']}.npz") as initial:
                        c = cartesian_noise_covariance(z['cell'], float(initial['std']), len(z['numbers']))
                    e, u = np.linalg.eigh(c);square_root = (u*np.sqrt(e)) @ u.T
                    proxy = np.eye(len(c)) + square_root @ matrix @ square_root
                    evidence['whitened_posterior_proxy'] = matrix_diagnostics(proxy)
                    evidence['posterior_proxy_caveat'] = 'Algebraic full-matrix diagnostic; full-matrix FD QC failed in pilot. Not a validated uncertainty estimate.'
                    row['dtype'][dtype] = evidence
                    plot_rows.append(dict(id=case['id'], t=time['t'], dtype=dtype, steps=source['steps_A'],
                        errors=[max(relative_error(matrix@v, a), relative_error(matrix@w, b))
                                for a, b in zip(finite['v'], finite['w'])]))
            rows.append(row)
    if max(audits) > 1e-10:raise ValueError('Raw adjoint reconstruction failed.')
    report = dict(experiment_sha256=sha(root / 'complete.json'), source_commit=source['source_commit'],
        policy=dict(window_A=WINDOW, vector_QC_relative=.02, curl_signal_vs_discrepancy=5,
                    status='Exploratory pilot: window chosen after sensitivity inspection; freeze for future independent confirmation.'),
        cases=len(source['cases']), case_times=len(rows), raw_audit_passed=True,
        max_adjoint_reconstruction_error=max(audits), rows=rows,
        native_corroborated=sum(r['dtype']['single']['corroborated_nonzero_curl'] for r in rows),
        fp64_corroborated=sum(r['dtype']['double']['corroborated_nonzero_curl'] for r in rows),
        conclusions=['Native nonzero cross-curl excludes exact conservative density-score/force identities locally where numerically corroborated.',
                     'No claim that approximate score matching or useful generation requires exact integrability.',
                     'FP64 results remain a distinct numerical representation; failed native-equivalence cases are not relabeled.',
                     'Random test-plane corroboration is not full-Jacobian finite-difference certification.',
                     'Four property-blind anchors are a pilot, not a population incidence estimate.'])
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot(plot_rows, output)
    print(json.dumps(dict(native_corroborated=report['native_corroborated'], fp64_corroborated=report['fp64_corroborated'], total=len(rows))))


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ids = sorted({r['id'] for r in rows})
    times = sorted({r['t'] for r in rows})
    colors = {t: f'C{j}' for j, t in enumerate(times)}
    for axis, case in zip(axes.ravel(), ids):
        for row in rows:
            if row['id'] != case:continue
            axis.loglog(row['steps'], np.maximum(row['errors'], 1e-12),
                color=colors[row['t']], linestyle='-' if row['dtype']=='single' else '--',
                marker='o' if row['dtype']=='single' else 'x', markersize=3,
                label=f"t={row['t']} {'FP32' if row['dtype']=='single' else 'FP64'}")
        axis.axhline(.02, color='black', linestyle=':', linewidth=1)
        axis.set(title=case, xlabel='Cartesian displacement norm (angstrom)', ylabel='Max of two directional AD/FD errors')
        axis.grid(alpha=.2)
    axes.ravel()[0].legend(fontsize=6, ncol=2)
    fig.suptitle('Same poses and directions: precision noise versus finite-step response\nNeighbor-graph stability is reported separately')
    fig.savefig(output / 'response.png', dpi=180)
    fig.savefig(output / 'response.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execute(args.response, args.pilot, args.output)


if __name__ == '__main__':main()
