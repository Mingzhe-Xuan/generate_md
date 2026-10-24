"""Merge disjoint initialization strategies and retain every original sample."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from report import summarize
from context import SMOACS
from budget import full_steps

sys.path.insert(0,str(SMOACS/'postprocess'))
from mattersim_support import needs_isolated_retry,error_metadata


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_partition(ids, groups):
    actual = [i for values in groups.values() for i in values]
    if len(ids) != 128 or len(set(ids)) != 128 or len(actual) != 128 or len(set(actual)) != 128 or set(actual) != set(ids):
        raise ValueError('Strategies must partition all128 original IDs exactly.')


def rate(values):
    n = len(values)
    if not n or not all(v is None or type(v) is bool for v in values):
        raise ValueError('Explicit boolean/unknown outcomes required.')
    p, u = sum(v is True for v in values), sum(v is None for v in values)
    return dict(passed=p, unknown=u, denominator=n, confirmed_rate=p/n, possible_rate_bounds=[p/n, (p+u)/n])


def sun_metrics(rows, properties, ids):
    indexed = {r['candidate_id']: r for r in rows}
    predicted = {r['candidate_id']: r for r in properties}
    if not set(ids) <= indexed.keys() or not set(ids) <= predicted.keys():
        raise ValueError('Missing stratified SUN cases.')
    result = {key: rate([indexed[i][key] for i in ids]) for key in ('stable', 'unique', 'novel', 'sun')}
    for model in ('crystalformer', 'alignn'):
        values = []
        for i in ids:
            a, b = indexed[i]['sun'], predicted[i][model+'_flags']['joint']
            values.append(False if a is False or b is False else None if a is None or b is None else True)
        result['SUN_and_'+model+'_joint'] = rate(values)
    return result


def merge(root, source):
    steps = full_steps(root)
    original = load(source/'inventory-249/inventory.json')
    ids = [c['candidate_id'] for c in original['candidates']]
    groups, candidates, predictions, metadata = {}, {}, [], {}
    for strategy in ('neutral', 'unmasked'):
        arm = root/strategy
        report = load(arm/'comparison.json')
        groups[strategy] = report['ids']
        if report['strategy'] != strategy or not report['gpu_audit']['passed']:
            raise ValueError('Incomplete strategy or CUDA audit.')
        initialization = load(arm/'initialization.json')
        if initialization['source_inventory_sha256'] != digest(source/'inventory-249/inventory.json'):
            raise ValueError('Strategies started from a different source cohort.')
        if any(v['eligible'] != (strategy == 'neutral') for v in initialization['selected']):
            raise ValueError('Wrong initialization-strategy assignment.')
        evaluated = arm/'evaluation-cycle1'
        inventory = load(evaluated/'inventory-249/inventory.json')
        if [c['candidate_id'] for c in inventory['candidates']] != groups[strategy]:
            raise ValueError('Evaluation IDs differ.')
        for candidate in inventory['candidates']:
            if candidate['a_updates'] != steps:
                raise ValueError('Candidate A updates differ from the reported optimization budget.')
            if digest(candidate['source']) != candidate['sha256']:
                raise ValueError('Optimized endpoint changed.')
            candidates[candidate['candidate_id']] = candidate
            record = load(evaluated/'relaxation'/Path(candidate['candidate_id']).stem/'result.json')
            if needs_isolated_retry(record):
                raise ValueError('Unrechecked CUDA failure cannot enter the full evaluation.')
        rows = load(evaluated/'cross-prediction/predictions.json')
        control = load(evaluated/'cross-prediction/metadata.json')
        if control['denominator'] != len(groups[strategy]) or not control['crystalformer_control']['control_passed']:
            raise ValueError('Frozen predictor control failed.')
        if len(control['controls']) < 2 or any(max(c['max_errors'])>1e-5 for c in control['controls']):
            raise ValueError('Independent ALIGNN controls failed.')
        metadata[strategy] = control
        for stage in ('before', 'after'):
            selected = [r for r in rows if r['stage'] == stage]
            if len(selected) != len(groups[strategy]) or {r['candidate_id'] for r in selected} != set(groups[strategy]):
                raise ValueError('Missing strategy prediction rows.')
        predictions += rows
    validate_partition(ids, groups)
    evaluation = root/'evaluation'
    if not (evaluation/'inventory-249/inventory.json').exists():
        (evaluation/'inventory-249').mkdir(parents=True, exist_ok=False)
        (evaluation/'relaxation').mkdir()
        for key in ids:
            strategy = 'neutral' if key in groups['neutral'] else 'unmasked'
            prior = root/strategy/'evaluation-cycle1/relaxation'/Path(key).stem
            destination = evaluation/'relaxation'/Path(key).stem
            destination.mkdir()
            record = load(prior/'result.json')
            if needs_isolated_retry(record):
                raise ValueError('Unrechecked CUDA failure cannot enter the full evaluation.')
            if record['input_sha256'] != candidates[key]['sha256']:
                raise ValueError('B input differs from optimized endpoint.')
            if record['status'] in ('converged', 'unconverged'):
                if digest(prior/'relaxed.vasp') != record['relaxed_sha256']:
                    raise ValueError('B geometry changed.')
                shutil.copy2(prior/'relaxed.vasp', destination/'relaxed.vasp')
                record['relaxed_path'] = str((destination/'relaxed.vasp').resolve())
            (destination/'result.json').write_text(json.dumps(record, indent=2)+'\n')
        merged = dict(role=f'MatterGen_CF{steps}_B_full128_two_initialization_strategies', denominator=128,
                      candidates=[candidates[i] for i in ids], groups=groups, controls=metadata)
        (evaluation/'inventory-249/inventory.json').write_text(json.dumps(merged, indent=2)+'\n')
    else:
        prior = load(evaluation/'inventory-249/inventory.json')
        if [c['sha256'] for c in prior['candidates']] != [candidates[i]['sha256'] for i in ids]:
            raise ValueError('Merged endpoints changed.')
        for key in ids:
            strategy = 'neutral' if key in groups['neutral'] else 'unmasked'
            a = load(root/strategy/'evaluation-cycle1/relaxation'/Path(key).stem/'result.json')
            b = load(evaluation/'relaxation'/Path(key).stem/'result.json')
            if any(a.get(k)!=b.get(k) for k in ('input_sha256','status','relaxed_sha256')):
                raise ValueError('Merged B endpoints are stale; refresh after repair.')
    indexed = {(r['stage'], r['candidate_id']): r for r in predictions}
    ordered = [indexed[stage, key] for stage in ('before', 'after') for key in ids]
    (evaluation/'predictions.json').write_text(json.dumps(ordered, indent=2)+'\n')
    scope = []
    for label, proposal, cohort in [('direct', source, original['candidates']), ('hybrid', evaluation, [candidates[i] for i in ids])]:
        for c in cohort:
            record = load(proposal/'relaxation'/Path(c['candidate_id']).stem/'result.json')
            if record['status'] == 'converged':
                scope.append(dict(candidate_id=label+'_'+c['candidate_id'], elements=c['elements']))
    (root/'reference-scope.json').write_text(json.dumps(dict(denominator=len(scope), candidates=scope), indent=2)+'\n')
    return ids, groups, ordered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--refresh-merged', action='store_true')
    args = p.parse_args()
    root, source = args.run.resolve(), args.source.resolve()
    if args.refresh_merged:
        if not os.environ.get('SLURM_JOB_ID') or not root.is_relative_to((Path.cwd()/'results/mattergen').resolve()):
            raise ValueError('Refresh only current task results inside Slurm.')
        for name in ('evaluation','comparison.json','comparison.md','reference-scope.json','SUN-direct','SUN-hybrid','SUN.complete','SUN-job-id.txt'):
            path = root/name
            if path.exists():
                path.rename(root/(name+'-before-cuda-repair-'+os.environ['SLURM_JOB_ID']))
    ids, groups, optimized = merge(root, source)
    steps = full_steps(root)
    a_name, b_name = f'MatterGen_CF{steps}', f'MatterGen_CF{steps}_B'
    baseline = load(source/'cross-prediction/predictions.json')
    cohorts = {}
    for name, rows, stage in [('MatterGen_raw', baseline, 'before'), ('MatterGen_B', baseline, 'after'),
                             (a_name, optimized, 'before'), (b_name, optimized, 'after')]:
        indexed = {r['candidate_id']: r for r in rows if r['stage'] == stage}
        if set(indexed) != set(ids):
            raise ValueError('Full128 comparison outcome IDs differ.')
        cohorts[name] = indexed
    summaries = {}
    for group, chosen in dict(all128=ids, **groups).items():
        summary = dict(denominator=len(chosen), ids=chosen,
            properties={name: summarize([rows[i] for i in chosen]) for name, rows in cohorts.items()})
        summary['paired_gap_errors'] = {}
        for model in ('crystalformer', 'alignn'):
            common = [i for i in chosen if all(cohorts[name][i][model+'_gap'] is not None for name in ('MatterGen_B', b_name))]
            summary['paired_gap_errors'][model] = dict(n=len(common), ids=common,
                MAE={name: sum(abs(cohorts[name][i][model+'_gap']-2.5) for i in common)/len(common) if common else None
                     for name in ('MatterGen_B', b_name)})
        if (root/'SUN-direct/summary.json').exists() and (root/'SUN-hybrid/summary.json').exists():
            summary['SUN'] = {name: sun_metrics(load(root/f'SUN-{label}/candidates.json'),
                list(cohorts[name].values()), chosen) for name, label in [('MatterGen_B', 'direct'), (b_name, 'hybrid')]}
        summaries[group] = summary
    statuses = Counter(load(root/'evaluation/relaxation'/Path(i).stem/'result.json')['status'] for i in ids)
    failures = Counter()
    for i in ids:
        record = load(root/'evaluation/relaxation'/Path(i).stem/'result.json')
        if record['status'] != 'converged':
            failures[record.get('error_type') or (error_metadata(record.get('error',''))['error_type'] if record['status']=='failed' else record['status'])] += 1
    supports = [load(root/k/'repair-model-support.json') for k in groups if (root/k/'repair-model-support.json').exists()]
    model_support = None
    if supports:
        maxima = {r['model_max_z'] for r in supports}
        if len(maxima) != 1:
            raise ValueError('Repair workers used different model encoders.')
        maximum = maxima.pop()
        model_support = dict(max_z=maximum, output_vocabulary_remains98=True, strategy_counts={})
        supported_ids = []
        for k in groups:
            rows = load(root/k/'results.json')['cycles'][0]['candidates']
            supported = sum(all(1<=z<=maximum for z in r['A']['atomic_numbers']) for r in rows)
            model_support['strategy_counts'][k] = dict(supported=supported, unsupported=len(rows)-supported)
            supported_ids += [r['candidate_id'] for r in rows if all(1<=z<=maximum for z in r['A']['atomic_numbers'])]
        model_support['supported_ids'] = supported_ids
        model_support['supported_only_properties'] = {name: summarize([rows[i] for i in supported_ids]) for name, rows in cohorts.items()}
    ref_pointer = root/'reference-memory-latest.txt'
    ref_memory = Path(ref_pointer.read_text().strip()) if ref_pointer.exists() else root/'reference-memory'
    result = dict(denominator=128, steps_per_A=steps, strategy_counts={k: len(v) for k, v in groups.items()}, groups=summaries,
        relaxation_status_counts=dict(statuses), SUN_complete='SUN' in summaries['all128'],
        relaxation_failure_causes=dict(failures), model_support=model_support,
        memory=load(root/'memory-summary.json') if (root/'memory-summary.json').exists() else None,
        torch_memory={k: load(root/k/'torch-memory.json') for k in groups},
        reference_memory=load(ref_memory/'memory-summary.json') if (ref_memory/'memory-summary.json').exists() else None,
        repair_memory={p.parent.name: load(p) for p in root.glob('repair-memory-*/memory-summary.json')},
        thresholds=dict(gap_eV=[2.3, 2.7], eform_eV_atom_less_than=-.5, stable_hull_eV_atom_less_than=.1),
        note='All128 original MatterGen raw structures participate. Neutral65 uses the pilot oxidation masks; unmasked63 uses no oxidation mask, with an inactive interface tensor. '
             f'Each strategy has its own global mean Adam loss; both use{steps}updates/cosine/base rates and the same B profile. '
             'Primary denominator128 retains failures/unknowns;65/63 are separate strata. SUN uses one new union reference pool for direct and hybrid B; U is over each full128 method, then stratified. No DFT.')
    result['note'] += ' CF98 vocabulary is retained by user choice; unsupported MatterSim terminal species are explicit model-domain failures. Supported-only results are postoptimization diagnostics, not the original cohort.'
    (root/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = [f'# Full128 MatterGen -> CF{steps} -> MatterSim', '', result['note'], '']
    for group, values in summaries.items():
        lines += [f'## {group} (n={values["denominator"]})', '', f'| Metric | Raw | Direct B | CF{steps} | CF{steps}+B |', '|---|---:|---:|---:|---:|']
        for metric in next(iter(values['properties'].values())):
            cells = []
            for name in cohorts:
                value = values['properties'][name][metric]
                cells.append(f"{value['passed']}/{value['denominator']} (unknown {value['unknown']})" if 'passed' in value else f"{value['value']} (n={value['n']})")
            lines.append('| '+metric+' | '+' | '.join(cells)+' |')
        if 'SUN' in values:
            lines += ['', f'| SUN metric | Direct B | CF{steps}+B |', '|---|---:|---:|']
            for metric in values['SUN']['MatterGen_B']:
                cells = [f"{values['SUN'][name][metric]['passed']}/{values['denominator']} (unknown {values['SUN'][name][metric]['unknown']})" for name in ('MatterGen_B', b_name)]
                lines.append('| '+metric+' | '+' | '.join(cells)+' |')
        lines.append('')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(denominator=128, strategy_counts=result['strategy_counts'], B=statuses, SUN_complete=result['SUN_complete'])), flush=True)


if __name__ == '__main__':
    main()
