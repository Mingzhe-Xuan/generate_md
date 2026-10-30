"""Independent raw-state audit and descriptive phase plots."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from metrics import ARMS, summary, fixed_cohort, geometry
from run import save, sha, owned_artifacts


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = json.loads((root/'complete.json').read_text()); rows = complete['rows']
    if len(rows) != 4872 or complete['sampling']['raw_forward_calls'] != 12000:
        raise ValueError('Run coverage incomplete.')
    index = {(r['arm'], r['parent'], r['frame']): r for r in rows}
    if len(index) != len(rows): raise ValueError('Duplicate states.')
    priors = {}; raw_checked = 0
    for path in complete['sampling']['paths']:
        if sha(root/path['file']) != path['sha256']: raise ValueError('State hash mismatch.')
        with np.load(root/path['file'], allow_pickle=False) as d:
            key = path['parent']; prior = [d[k][0].copy() for k in ('numbers', 'cell', 'frac')]
            if key in priors and not all(np.array_equal(a, b) for a, b in zip(priors[key], prior)):
                raise ValueError('Initial pairing broken.')
            priors[key] = prior
            for k in range(203):
                r = index[path['arm'], key, k]
                reasons, nearest = geometry(d['numbers'][k], d['cell'][k], d['frac'][k], 98)
                if reasons != r['reasons'] or r['masked'] != int(np.sum(d['numbers'][k] == 101)):
                    raise ValueError('Eligibility audit failed.')
                if bool(reasons) == (r['gap'] is not None): raise ValueError('Missing-prediction eligibility disagreement.')
                if r['gap'] is not None and not np.isfinite([r['gap'], r['eform']]).all(): raise ValueError('Nonfinite property.')
                raw_checked += 1
    # Check MatterSim raw-response energies against the independently logged ledger.
    physics_checked = 0
    for line in (root/'physics/ledger.jsonl').read_text().splitlines():
        record = json.loads(line)
        for task, result in zip(record['tasks'], record['results']):
            arm, parent, frame = task['id'].split(':'); r = index[arm, int(parent), int(frame)]
            if result['status'] == 'completed':
                if not np.isclose(r['energy_eV_atom'], result['energy_eV']/len(task['numbers']), atol=1e-12, rtol=0):
                    raise ValueError('Energy ledger mismatch.')
                physics_checked += 1
    stats = summary(rows); cohorts = {str(start): fixed_cohort(rows, start) for start in (500, 750, 900)}
    report = dict(summary=stats, fixed_cohorts=cohorts,
        terminal={arm: [r for r in stats if r['arm'] == arm and r['stage'] == 'terminal'][0] for arm in ARMS},
        phase_windows={}, raw_states_audited=raw_checked, physics_points_audited=physics_checked)
    for start, end in ((0, 249), (250, 499), (500, 749), (750, 999)):
        section = {}
        for arm in ARMS:
            rr = [r for r in rows if r['arm'] == arm and r['stage'] == 'predictor' and start <= r['step'] <= end]
            good = [r for r in rr if r['gap'] is not None]
            section[arm] = dict(states=len(rr), valid=len(good), known=sum(r['masked'] == 0 for r in rr),
                median_mask_fraction=float(np.median([r['masked']/r['num_atoms'] for r in rr])),
                pooled_mae=float(np.mean([abs(r['gap']-r['target']) for r in good])) if good else None)
        report['phase_windows'][f'{start}-{end}'] = section
    save(root/'summary.json', report)
    keys = list(rows[0])
    keys += ['graph_hash'] if 'graph_hash' not in keys else []
    with (root/'states.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    colors = {'cfg2': '#2467aa', 'cfg0': '#d66b20'}
    labels = {'cfg2': 'Gap-conditioned, CFG=2', 'cfg0': 'Unconditional, CFG=0'}
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for arm in ARMS:
        color = colors[arm]; rr = [r for r in stats if r['arm'] == arm and r['stage'] != 'prior']
        x = [r['step'] for r in rr]
        for ax, key in ((axes[0, 0], 'mae'), (axes[0, 1], 'paired_mae')):
            ax.plot(x, [np.nan if r[key] is None else r[key] for r in rr], color=color, label=labels[arm])
        axes[1, 0].plot(x, [r['n'] for r in rr], color=color, label='Evaluable '+arm)
        axes[1, 1].plot(x, [r['mask_fraction'] for r in rr], color=color, label=labels[arm])
        first_errors = []; final_errors = []
        for parent in range(12):
            path = [r for r in rows if r['arm'] == arm and r['parent'] == parent and r['stage'] != 'prior']
            axes[2, 0].plot([r['step'] for r in path], [abs(r['gap']-r['target']) if r['gap'] is not None else np.nan for r in path], color=color, alpha=.24, lw=.7)
            valid = sorted([r for r in path if r['gap'] is not None], key=lambda r: r['frame'])
            terminal = [r for r in path if r['stage'] == 'terminal' and r['gap'] is not None]
            if valid and terminal:
                first_errors.append(abs(valid[0]['gap']-valid[0]['target']))
                final_errors.append(abs(terminal[0]['gap']-terminal[0]['target']))
        axes[2, 1].scatter(first_errors, final_errors, color=color, alpha=.75, s=27, label=f'{arm}, n={len(first_errors)}')
        if first_errors:
            axes[2, 1].scatter(np.mean(first_errors), np.mean(final_errors), color=color, marker='X', s=110, edgecolor='black', linewidth=.7)
    paired = [r for r in stats if r['arm'] == 'cfg2' and r['stage'] != 'prior']
    axes[1, 0].plot([r['step'] for r in paired], [r['paired_n'] for r in paired], color='gray', ls='--', label='Paired evaluable')
    titles = ['ALIGNN target MAE: available states', 'ALIGNN target MAE: matched parents at each step',
              'Coverage out of 12 paths per arm', 'Mean fraction of MASK atom types',
              'Individual target-error trajectories', 'Same paths: first evaluable vs terminal (X = mean)']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=11); ax.grid(alpha=.22)
    for ax in (axes[0, 0], axes[0, 1], axes[2, 0], axes[2, 1]): ax.set_ylabel('Absolute gap error (eV)')
    axes[1, 0].set_ylabel('Structures'); axes[1, 1].set_ylabel('MASK fraction')
    axes[1, 0].set_ylim(-.3, 12.5); axes[1, 1].set_ylim(-.02, 1.02)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]):
        ax.set_xlim(0, 1000); ax.set_xlabel('Native PC rounds (terminal mean at 1000)')
    for ax in (axes[0, 0], axes[2, 0]):
        valid_steps = [r['step'] for r in rows if r['gap'] is not None and r['stage'] != 'prior']
        if valid_steps: ax.set_xlim(max(0, min(valid_steps)-15), 1000)
    matched_steps = [r['step'] for r in stats if r['paired_n'] > 0 and r['stage'] != 'prior']
    if matched_steps: axes[0, 1].set_xlim(max(0, min(matched_steps)-5), 1000)
    limit = max(abs(r['gap']-r['target']) for r in rows if r['gap'] is not None)*1.06
    axes[2, 1].plot([0, limit], [0, limit], color='gray', ls=':', lw=1)
    axes[2, 1].set_xlim(0, limit); axes[2, 1].set_ylim(0, limit)
    axes[2, 1].set_xlabel('First evaluable absolute error (eV)')
    axes[2, 1].set_ylabel('Terminal absolute error (eV)')
    for ax in (axes[0, 0], axes[1, 0], axes[2, 1]):
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    fig.suptitle('MatterGen native joint generation, target 2.5 eV | ALIGNN proxy\n12 paired paths per arm; 3 coupled batches; no relaxation; early MASK states excluded', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for ext in ('png', 'pdf', 'svg'): fig.savefig(root/f'bandgap-trajectory.{ext}', dpi=180)
    plt.close(fig)
    text = ['# Bandgap-conditioned native trajectory', '',
        'ALIGNN target MAE is target adherence, not error against DFT labels. Predictions on noisy geometries are uncalibrated.',
        'The native joint model changes atom types and geometry; absolute MatterSim energies across composition changes are confounded by elemental reference offsets.',
        'Same checkpoint, common initial priors and per-batch seeds. Three native coupled batches per arm; descriptive statistics only.',
        'MASK and unsupported/invalid states are missing, never zero-filled. Actual returned terminal mean is recorded separately at 1000.', '',
        'Terminal descriptive statistics:', '```json', json.dumps(report['terminal'], indent=2), '```',
        'Phase windows below pool available states; changing cohort can affect their mean:', '```json', json.dumps(report['phase_windows'], indent=2), '```',
        'Fixed late cohorts:', json.dumps(cohorts), '',
        f'Raw audited states: {raw_checked}; MatterSim ledger points: {physics_checked}.']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    save(root/'audit.json', dict(passed=True, raw_states=raw_checked, exact_shared_priors=True,
        native_controls=len(complete['controls']), max_native_errors=[c['max_errors'] for c in complete['controls']],
        complete_sha256=sha(root/'complete.json'), report_code_sha256=sha(Path(__file__)),
        artifacts=owned_artifacts(root, 'bandgap-trajectory', ('states.csv',))))
    print(json.dumps(report['terminal']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Use Slurm.')
    main(args.run.absolute())
