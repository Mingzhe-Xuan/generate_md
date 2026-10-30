"""Descriptive event-aligned stage analysis; no extra inference or chosen change points."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from run import save, sha


def events(numbers, steps, stages, rows):
    selected = [i for i, stage in enumerate(stages) if stage != 'prior']
    known = [i for i in selected if np.all(numbers[i] != 101)]
    eligible = [i for i in selected if rows[i]['gap'] is not None]
    final = numbers[-1]
    stable = len(numbers)-1
    while stable > 0 and np.array_equal(numbers[stable-1], final): stable -= 1
    transitions = [i for i in selected if i > 0 and not np.array_equal(numbers[i-1], numbers[i])]
    consecutive = [(i, j) for i, j in zip(eligible, eligible[1:]) if j == i+1]
    comparable = [(i, j) for i, j in consecutive if np.array_equal(numbers[i], numbers[j])]
    result = dict(first_types_known=int(steps[known[0]]) if known else None,
        first_evaluable=int(steps[eligible[0]]) if eligible else None,
        final_composition_sustained_from=int(steps[stable]),
        observed_species_transitions=len(transitions), evaluable_states=len(eligible),
        adjacent_evaluable_pairs=len(consecutive), fixed_species_adjacent_pairs=len(comparable),
        target_error_increase_pairs=sum(abs(rows[j]['gap']-rows[j]['target']) > abs(rows[i]['gap']-rows[i]['target']) for i, j in consecutive),
        fixed_species_error_increase_pairs=sum(abs(rows[j]['gap']-rows[j]['target']) > abs(rows[i]['gap']-rows[i]['target']) for i, j in comparable))
    if eligible and rows[-1]['gap'] is not None:
        first = rows[eligible[0]]; last = rows[-1]
        result.update(first_error=abs(first['gap']-first['target']), terminal_error=abs(last['gap']-last['target']),
            terminal_minus_first_error=abs(last['gap']-last['target'])-abs(first['gap']-first['target']),
            first_vs_terminal_same_species=bool(np.array_equal(numbers[eligible[0]], final)))
        if first.get('force_rms') is not None and last.get('force_rms') is not None:
            result.update(first_force_rms=first['force_rms'], terminal_force_rms=last['force_rms'],
                terminal_minus_first_force_rms=last['force_rms']-first['force_rms'])
    return result


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
        median=float(np.median(a)), q25=float(np.quantile(a, .25)), q75=float(np.quantile(a, .75))) if len(a) else dict(n=0)


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = json.loads((root/'complete.json').read_text()); rows = complete['rows']
    all_events = []; lookup = {}
    for path in complete['sampling']['paths']:
        if sha(root/path['file']) != path['sha256']: raise ValueError('State hash mismatch.')
        rr = sorted([r for r in rows if r['arm'] == path['arm'] and r['parent'] == path['parent']], key=lambda r: r['frame'])
        lookup[path['arm'], path['parent']] = rr
        with np.load(root/path['file'], allow_pickle=False) as d:
            info = events(d['numbers'], d['step'], d['stage'], rr)
        info.update(arm=path['arm'], parent=path['parent'], batch=path['batch'])
        all_events.append(info)
    output = dict(paths=all_events, arms={})
    for arm in ('cfg2', 'cfg0'):
        subset = [p for p in all_events if p['arm'] == arm]
        output['arms'][arm] = {key: describe([p[key] for p in subset if p.get(key) is not None]) for key in
            ('first_types_known', 'first_evaluable', 'final_composition_sustained_from', 'first_error', 'terminal_error',
             'terminal_minus_first_error', 'first_force_rms', 'terminal_force_rms', 'terminal_minus_first_force_rms')}
        output['arms'][arm]['first_to_terminal_improved'] = sum(p.get('terminal_minus_first_error', 0) < 0 for p in subset)
        output['arms'][arm]['first_to_terminal_force_reduced'] = sum(p.get('terminal_minus_first_force_rms', 0) < 0 for p in subset)
        for key in ('adjacent_evaluable_pairs', 'fixed_species_adjacent_pairs', 'target_error_increase_pairs', 'fixed_species_error_increase_pairs'):
            output['arms'][arm][key] = sum(p[key] for p in subset)
    save(root/'phase-analysis.json', output)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for arm, color in [('cfg2', '#2467aa'), ('cfg0', '#d66b20')]:
        subset = [r for r in rows if r['arm'] == arm and r['stage'] == 'predictor']
        xx = sorted({r['step'] for r in subset})
        for ax, key in ((axes[0, 1], 'minimum_pair_A'), (axes[1, 0], 'force_rms'), (axes[1, 1], 'eform')):
            yy = []
            for step in xx:
                values = [r[key] for r in subset if r['step'] == step and r[key] is not None]
                yy.append(np.median(values) if values else np.nan)
            ax.plot(xx, yy, color=color, label=arm)
        for parent in range(12):
            rr = lookup[arm, parent]; event = next(p for p in all_events if p['arm'] == arm and p['parent'] == parent)
            first = event['first_types_known']
            if first is None: continue
            axes[0, 0].plot([r['step']-first for r in rr if r['stage'] != 'prior'],
                [abs(r['gap']-r['target']) if r['gap'] is not None else np.nan for r in rr if r['stage'] != 'prior'], color=color, alpha=.4, lw=.9)
    axes[0, 0].axvline(0, color='gray', ls=':', lw=1)
    axes[0, 0].set_title('Target error aligned to first fully known composition')
    axes[0, 0].set_xlabel('Rounds since first known atom types (observed every 5)')
    axes[0, 0].set_ylabel('ALIGNN absolute gap error (eV)')
    axes[0, 1].set_title('Median nearest distinct-atom periodic separation')
    axes[0, 1].set_ylabel('Distance (A)')
    axes[1, 0].set_title('Median MatterSim RMS atomic force, eligible states')
    axes[1, 0].set_ylabel('Force (eV/A)'); axes[1, 0].set_yscale('log')
    axes[1, 1].set_title('Median ALIGNN formation energy, eligible states')
    axes[1, 1].set_ylabel('Formation energy proxy (eV/atom)')
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]): ax.set_xlabel('Native PC round'); ax.legend()
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle('Descriptive phase probes | unrelaxed native joint trajectories\nChanging cohorts and compositions; model estimates on noisy states are uncalibrated', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for ext in ('png', 'svg', 'pdf'): fig.savefig(root/f'stage-probes.{ext}', dpi=180)
    plt.close(fig)
    print(json.dumps(output['arms']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Use Slurm.')
    main(args.run.absolute())
