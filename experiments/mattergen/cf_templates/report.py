"""Paired direct MatterGen B versus MatterGen-template CF A600/B feasibility."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(rows):
    output = {}
    for model in ('crystalformer', 'alignn'):
        for metric in ('gap', 'eform', 'joint'):
            values = [r[model+'_flags'][metric] for r in rows]
            output[model+'_'+metric] = dict(passed=sum(v is True for v in values),
                unknown=sum(v is None for v in values), denominator=len(rows))
        errors = [abs(r[model+'_gap']-2.5) for r in rows if r[model+'_gap'] is not None]
        output[model+'_gap_MAE'] = dict(n=len(errors), value=statistics.mean(errors) if errors else None)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    ids = protocol['ids']
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError('Require the unchanged four selected IDs.')
    eligibility = json.loads((args.run/'eligibility.json').read_text())
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

    cohorts = dict(MatterGen_raw=select(baseline, 'before'), MatterGen_B=select(baseline, 'after'),
                   MatterGen_CF600=select(optimized, 'before'), MatterGen_CF600_B=select(optimized, 'after'))
    result = dict(denominator=4, ids=ids, eligibility=eligibility, gpu_audit=gpu,
        gradient_audit=gradient, protocol=protocol, metrics={k: summarize(v) for k, v in cohorts.items()},
        per_case=cohorts, note='Feasibility pilot, not full128 performance or SUN. Source cells/coordinates/initial hard species preserved; soft distributions differ explicitly. '
        'Four eligible size-selected cases;63 nonneutral cases are not silently discarded from full128 eligibility. Direct-B and optimized-B share the same original generated IDs.')
    (args.run/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# MatterGen initial templates: CF GPU feasibility', '', result['note'], '',
             '| Metric | Raw | Direct B | CF600 | CF600+B |', '|---|---:|---:|---:|---:|']
    names = list(cohorts)
    for metric in result['metrics'][names[0]]:
        values = [result['metrics'][name][metric] for name in names]
        cells = [(f"{v['passed']}/4 (unknown {v['unknown']})" if 'passed' in v else f"{v['value']} (n={v['n']})") for v in values]
        lines.append('| '+metric+' | '+' | '.join(cells)+' |')
    (args.run/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True, denominator=4, eligible=eligibility['eligible'], ids=ids)), flush=True)


if __name__ == '__main__':
    main()
