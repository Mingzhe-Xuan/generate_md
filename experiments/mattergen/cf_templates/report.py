"""Paired direct MatterGen B versus MatterGen-template CF A600/B feasibility."""
import argparse
import json
from pathlib import Path
import statistics
from budget import steps_for


def summarize(rows):
    output = {}
    for model in ('crystalformer', 'alignn'):
        for metric in ('gap', 'eform', 'joint'):
            values = [r[model+'_flags'][metric] for r in rows]
            output[model+'_'+metric] = dict(passed=sum(v is True for v in values),
                unknown=sum(v is None for v in values), denominator=len(rows))
        errors = [abs(r[model+'_gap']-2.5) for r in rows if r[model+'_gap'] is not None]
        output[model+'_gap_MAE'] = dict(n=len(errors), value=statistics.mean(errors) if errors else None)
    validity = [r.get('structure_valid') for r in rows]
    output['structure_valid'] = dict(passed=sum(v is True for v in validity), unknown=sum(v is None for v in validity), denominator=len(rows))
    both = []
    for row in rows:
        flags = [row[m+'_flags']['joint'] for m in ('crystalformer', 'alignn')]
        both.append(False if False in flags else None if None in flags else True)
    output['both_models_joint'] = dict(passed=sum(v is True for v in both), unknown=sum(v is None for v in both), denominator=len(rows))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    steps = steps_for(args.run)
    ids = protocol['ids']
    n = len(ids)
    if not n or len(set(ids)) != n:
        raise ValueError('Require unchanged distinct selected IDs.')
    eligibility = json.loads((args.run/'eligibility.json').read_text())
    if ids != eligibility['selected_ids']:
        raise ValueError('Selected source IDs differ.')
    gpu = json.loads((args.run/'gpu-audit.json').read_text())
    gradient = json.loads((args.run/'initial-gradient-audit.json').read_text())
    if not (gpu['passed'] and gpu['all_cuda'] and gradient['passed']):
        raise ValueError('Native CUDA optimization audit failed.')
    baseline = json.loads((args.source/'cross-prediction/predictions.json').read_text())
    optimized = json.loads((args.run/'evaluation-cycle1/cross-prediction/predictions.json').read_text())

    def select(rows, stage):
        selected = {r['candidate_id']: r for r in rows if r['stage'] == stage and r['candidate_id'] in ids}
        if set(selected) != set(ids):
            raise ValueError('Missing paired template outcomes.')
        return [selected[i] for i in ids]

    cohorts = dict(MatterGen_raw=select(baseline, 'before'), MatterGen_B=select(baseline, 'after'))
    cohorts.update({f'MatterGen_CF{steps}': select(optimized, 'before'), f'MatterGen_CF{steps}_B': select(optimized, 'after')})
    strategy = eligibility.get('strategy', 'pilot')
    result = dict(denominator=n, ids=ids, strategy=strategy, eligibility=eligibility, gpu_audit=gpu,
        gradient_audit=gradient, protocol=protocol, metrics={k: summarize(v) for k, v in cohorts.items()},
        per_case=cohorts, note=f'{strategy} initialization strategy, {n} original source IDs. Source cells/coordinates/initial hard species preserved; soft distributions differ explicitly. '
        'Oxidation masking is absent for the unmasked strategy; its oxidation tensor is an inactive interface placeholder. '
        'Direct-B and optimized-B share source IDs. This per-strategy report alone is not a full128 or SUN result.')
    (args.run/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# MatterGen initial templates: CF GPU feasibility', '', result['note'], '',
             f'| Metric | Raw | Direct B | CF{steps} | CF{steps}+B |', '|---|---:|---:|---:|---:|']
    names = list(cohorts)
    for metric in result['metrics'][names[0]]:
        values = [result['metrics'][name][metric] for name in names]
        cells = [(f"{v['passed']}/{n} (unknown {v['unknown']})" if 'passed' in v else f"{v['value']} (n={v['n']})") for v in values]
        lines.append('| '+metric+' | '+' | '.join(cells)+' |')
    (args.run/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True, denominator=n, strategy=strategy, eligible=eligibility['eligible'], ids=ids)), flush=True)


if __name__ == '__main__':
    main()
