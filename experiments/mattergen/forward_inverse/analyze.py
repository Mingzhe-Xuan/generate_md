"""Audit raw local-response vectors and compare held-out score-curvature laws."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from curvature_fit import METHODS, heldout_systems
from relations import (cartesian_noise_covariance, fd_jacobian,
                       matrix_diagnostics, projected_adjoint_jacobian,
                       relative_error)


PHYSICAL_TOLERANCE = .02
MIN_EIGENVALUE = 1e-4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_diagnostics(hessians):
    symmetric = .5 * (hessians[0] + hessians[0].T)
    eig = np.linalg.eigvalsh(symmetric)
    diagnostics = [matrix_diagnostics(h) for h in hessians]
    change = relative_error(*hessians)
    return dict(FD_relative_change=change, diagnostics=diagnostics,
                min_eigenvalue=float(eig.min()),
                negative_modes=int(np.sum(eig < -MIN_EIGENVALUE)),
                soft_modes=int(np.sum(np.abs(eig) <= MIN_EIGENVALUE)),
                full_SPD=bool(eig.min() > MIN_EIGENVALUE),
                numerical_QC=bool(change <= PHYSICAL_TOLERANCE and
                    max(d['antisymmetric_fraction'] for d in diagnostics) <= PHYSICAL_TOLERANCE))


def audit_case(folder, plan):
    result = json.loads((folder / 'result.json').read_text())
    if result['status'] != 'completed':
        return dict(id=result['id'], system=result['system'], status=result['status']), []
    vectors = folder / 'vectors.npz'
    if digest(vectors) != result['vectors_sha256']:
        raise ValueError('Vector hash mismatch: ' + str(vectors))
    with np.load(vectors) as z:
        q, cell = z['Q'], z['cell']
        if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-12):
            raise ValueError('Nonorthonormal centered basis.')
        audit_errors = []
        physical = {}
        for label in ('anchor', 'probe'):
            matrices = z['H_' + label]
            for j, step in enumerate(plan['policy']['FD_steps_A']):
                raw = z[label + '_' + str(step)]
                rebuilt = -q.T @ fd_jacobian(raw[:, 0], raw[:, 1], step)
                audit_errors.append(relative_error(rebuilt, matrices[j]))
            physical[label] = physical_diagnostics(matrices)
        h = .5 * (z['H_probe'][0] + z['H_probe'][0].T)
        rows, records = [], []
        for j, details in enumerate(result['times']):
            t = details['t']
            with np.load(folder / f'time-{t}.npz') as tz:
                ad = projected_adjoint_jacobian(q, tz['adjoints'])
                audit_errors.append(relative_error(ad, z['score_jacobian_ad'][j]))
                audit_errors.append(relative_error(ad, tz['J_ad']))
            for k, step in enumerate(plan['policy']['FD_steps_A']):
                raw = z['score_fields'][j, k]
                rebuilt = q.T @ fd_jacobian(raw[:, 0], raw[:, 1], step)
                audit_errors.append(relative_error(rebuilt, z['score_jacobian'][j, k]))
            observed = -ad
            covariance = cartesian_noise_covariance(cell, z['fractional_std'][j], len(z['numbers']))
            row = dict(details, physical_numerical_QC=physical['probe']['numerical_QC'],
                       full_SPD=physical['probe']['full_SPD'],
                       score_curvature=matrix_diagnostics(observed))
            row['joint_numerical_QC'] = bool(row['native_derivative_QC_passed'] and row['physical_numerical_QC'])
            rows.append(row)
            records.append(dict(id=result['id'] + '@' + str(t), system=result['system'], t=t,
                                H=h, C=covariance, observed=observed,
                                native_QC=row['native_derivative_QC_passed'],
                                joint_QC=row['joint_numerical_QC'], SPD=row['full_SPD']))
        if max(audit_errors, default=0.) > 1e-10:
            raise ValueError('Raw-vector reconstruction discrepancy: ' + str(max(audit_errors)))
        return dict(id=result['id'], system=result['system'], num_atoms=result['num_atoms'],
                    status='completed', physical=physical, times=rows,
                    max_raw_reconstruction_error=max(audit_errors),
                    seconds=result['seconds'], NFE=result['NFE']), records


def fit_cohorts(records):
    output = []
    for t in sorted({r['t'] for r in records}):
        at_time = [r for r in records if r['t'] == t]
        for label, predicate, methods in (
                ('all_native', lambda r: True, ('zero', 'noise', 'scaled_noise')),
                ('native_QC', lambda r: r['native_QC'], ('zero', 'noise', 'scaled_noise')),
                ('full_SPD_all', lambda r: r['SPD'], METHODS),
                ('full_SPD_joint_QC', lambda r: r['SPD'] and r['joint_QC'], METHODS)):
            cohort = [r for r in at_time if predicate(r)]
            groups = sorted({r['system'] for r in cohort})
            item = dict(t=t, cohort=label, cases=len(cohort), systems=groups,
                        excluded=[r['id'] for r in at_time if not predicate(r)])
            if len(groups) < 3:
                item.update(status='insufficient_chemical_systems', fits=[])
            else:
                item.update(status='computed', fits=[heldout_systems(cohort, m) for m in methods])
            output.append(item)
    return output


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    completed = [r for r in report['cases'] if r['status'] == 'completed']
    for case in completed:
        times = [r['t'] for r in case['times']]
        axes[0, 0].plot(times, [r['score_curvature']['antisymmetric_fraction'] for r in case['times']],
                        'o-', label=case['system'] + '/' + case['id'])
        axes[0, 1].plot(times, [max(r['jacobian_ad_FD_errors']) for r in case['times']], 'o-')
        axes[1, 0].plot(times, [r['cos_score_force'] for r in case['times']], 'o-')
        axes[1, 0].plot(times, [r['cos_score_noise_restoration'] for r in case['times']], 'x--', alpha=.7)
    axes[0, 0].set(title='Native score Jacobian antisymmetry', ylabel='||J-J.T|| / (2 ||J||)')
    if completed:axes[0, 0].legend(fontsize=7)
    axes[0, 1].axhline(.02, color='black', linestyle=':', label='QC limit')
    axes[0, 1].set(title='Autograd versus finite differences', ylabel='Max relative matrix error')
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].set(title='Score alignment: force (o), noise restoration (x)', ylabel='Cosine')
    axes[1, 0].set_ylim(-1.05, 1.05)
    for method in METHODS:
        cohort = [r for r in report['curvature_fits'] if r['cohort'] == 'full_SPD_joint_QC' and r['status'] == 'computed']
        if cohort:
            axes[1, 1].plot([r['t'] for r in cohort],
                [next(f['R2_zero_baseline'] for f in r['fits'] if f['method'] == method) for r in cohort], 'o-', label=method)
    axes[1, 1].set(title='Held-out chemical systems: SPD + numerical QC', ylabel='1 - mean relative squared error')
    if axes[1, 1].lines:axes[1, 1].legend(fontsize=7)
    else:axes[1, 1].text(.5, .5, 'Fewer than 3 eligible chemical systems\nNo held-out physical-law fit', ha='center', va='center', transform=axes[1, 1].transAxes)
    for axis in axes.ravel():
        axis.set_xlabel('Diffusion time')
        axis.grid(alpha=.2)
    fig.savefig(output / 'curvature.png', dpi=180)
    fig.savefig(output / 'curvature.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise ValueError('Use a new analysis directory.')
    plan = json.loads((args.run / 'plan.json').read_text())
    complete = json.loads((args.run / 'complete.json').read_text())
    selected = plan['selection']['selected']
    if [r['id'] for r in complete['cases']] != [r['id'] for r in selected]:
        raise ValueError('Completion manifest does not cover all selected cases in order.')
    cases, records = [], []
    for selected_case in selected:
        case, vectors = audit_case(args.run / selected_case['id'], plan)
        cases.append(case); records.extend(vectors)
    report = dict(run=str(args.run), plan_sha256=digest(args.run / 'plan.json'),
                  complete_sha256=digest(args.run / 'complete.json'), source_commit=plan['commit'],
                  physical_QC_tolerance=PHYSICAL_TOLERANCE, SPD_threshold=MIN_EIGENVALUE,
                  selected=len(selected), completed=sum(r['status'] == 'completed' for r in cases),
                  cases=cases, curvature_fits=fit_cohorts(records),
                  caveats=['Fixed-cell Gamma-point local Hessian, not full phonon or finite-temperature stability.',
                           'Curated structure density is not demonstrated to be a Boltzmann ensemble.',
                           'FP64 equivalence is diagnostic, native FP32 autograd is primary.',
                           'Graph changes and numerical QC failures remain in all-case denominators.',
                           'Coefficients hold entire chemical systems out; beta is not a measured temperature.'])
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    plot(report, args.output)
    print(json.dumps(dict(selected=report['selected'], completed=report['completed'], output=str(args.output))))


if __name__ == '__main__':main()
