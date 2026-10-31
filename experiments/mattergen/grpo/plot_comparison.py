"""Plot the completed seven-arm comparison without dropping failed candidates."""
import argparse
import hashlib
import json
import math
from pathlib import Path


ARM_LABELS = {
    'base': 'Base / matched',
    'base_native': 'Base / native',
    'grpo_best': 'GRPO best / matched',
    'grpo_best_native': 'GRPO best / native',
    'grpo_last': 'GRPO last / matched',
    'conditional_matched': 'Conditional / matched',
    'conditional_native': 'Conditional / native',
}


def validate_comparison(data):
    if data.get('passed') is not True or set(data.get('rows', {})) != set(ARM_LABELS):
        raise ValueError('A completed seven-arm comparison is required.')
    n = data['rows']['base']['raw']['n']
    if type(n) is not int or n < 1:
        raise ValueError('Requested count must be a positive integer.')
    for arm in ARM_LABELS:
        row = data['rows'][arm]
        for stage in ('raw', 'relaxed'):
            m = row[stage]
            if m['n'] != n:
                raise ValueError('All arms/stages must retain the same full requested denominator.')
            valid, passed = m['valid'], m['gap_pass']
            if any(type(v) is not int for v in (valid, passed)) or not 0 <= passed <= valid <= n:
                raise ValueError('Invalid evaluability/pass counts.')
            for key, count in (('valid_fraction', valid), ('gap_pass_fraction', passed)):
                if not math.isclose(m[key], count/n, rel_tol=0., abs_tol=1e-12):
                    raise ValueError('Stored rate differs from its full-denominator count.')
            mae = m['gap_MAE_valid_eV']
            if valid == 0:
                if mae is not None:
                    raise ValueError('No evaluable samples: MAE must be undefined, not zero.')
            elif mae is None or not math.isfinite(mae) or mae < 0:
                raise ValueError('Evaluable samples require a finite nonnegative MAE.')
        force, joint = row['force_converged'], row['joint_gap_force_pass']
        if any(type(v) is not int for v in (force, joint)) or not 0 <= joint <= force <= n:
            raise ValueError('Invalid force/joint counts.')
        if joint > row['relaxed']['gap_pass']:
            raise ValueError('Joint successes cannot exceed relaxed gap successes.')
    return n


def render(data, prefix, fixture=False):
    n = validate_comparison(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = ('#466C99', '#DA9454')
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1.5, 1]})
    max_mae = max((r[s]['gap_MAE_valid_eV'] or 0.) for r in data['rows'].values()
                  for s in ('raw', 'relaxed'))
    mae_limit = max(1., max_mae)*1.6
    for i, arm in enumerate(ARM_LABELS):
        row = data['rows'][arm]
        for j, stage in enumerate(('raw', 'relaxed')):
            m = row[stage]; y = i + (j-.5)*.36
            rate = 100*m['gap_pass']/n
            axes[0].barh(y, rate, height=.3, color=colors[j], label=stage.title() if i == 0 else None)
            axes[0].text(rate+1, y, f"{m['gap_pass']}/{n}", va='center', fontsize=9)
            mae = m['gap_MAE_valid_eV']
            if mae is None:
                axes[1].text(.02*mae_limit, y, f'Undefined (0/{n} evaluable)', va='center', fontsize=9)
            else:
                axes[1].barh(y, mae, height=.3, color=colors[j])
                axes[1].text(mae+.015*mae_limit, y, f"{mae:.3f}  ({m['valid']}/{n} evaluable)",
                             va='center', fontsize=9)
        rate = 100*row['joint_gap_force_pass']/n
        axes[2].barh(i, rate, height=.55, color=colors[1])
        axes[2].text(rate+1, i, f"{row['joint_gap_force_pass']}/{n}", va='center', fontsize=9)
    axes[0].set_yticks(range(len(ARM_LABELS)), ARM_LABELS.values())
    axes[0].invert_yaxis()
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.2); axis.set_axisbelow(True)
    for axis in (axes[0], axes[2]):
        axis.set_xlim(0, 115); axis.set_xticks([0, 25, 50, 75, 100])
        axis.xaxis.set_major_formatter(PercentFormatter())
    axes[1].set_xlim(0, mae_limit)
    axes[0].set_title('Bandgap pass rate')
    axes[1].set_title('Bandgap MAE to 2.5 eV')
    axes[2].set_title('Relaxed gap AND force pass')
    axes[0].set_xlabel('All requested candidates')
    axes[1].set_xlabel('eV; evaluable candidates only')
    axes[2].set_xlabel('All requested candidates')
    title = 'TEST FIXTURE - NOT EXPERIMENT RESULTS' if fixture else 'MatterGen GRPO versus conditional generation'
    fig.suptitle(title+'\nBandgap target: 2.5 +/- 0.1 eV', fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .89), ncol=2)
    fig.text(.02, .045, 'Force pass: canonical MatterSim convergence and raw force <= 0.01 eV/Angstrom.', fontsize=10)
    fig.text(.02, .02, 'Same ALIGNN reward and evaluator; surrogate adherence only. One training seed; no DFT/SUN validation.', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .86))
    prefix = Path(prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [prefix.with_suffix(ext) for ext in ('.png', '.svg')]
    for path in paths: fig.savefig(path, dpi=180)
    plt.close(fig)
    return paths


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--fixture', action='store_true', help='Prominently label synthetic QA data.')
    args = p.parse_args(); payload = args.comparison.read_bytes()
    paths = render(json.loads(payload), args.output_prefix, args.fixture)
    metadata = dict(source=str(args.comparison), source_sha256=hashlib.sha256(payload).hexdigest(),
                    fixture=args.fixture, outputs=[str(path) for path in paths])
    args.output_prefix.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata))
