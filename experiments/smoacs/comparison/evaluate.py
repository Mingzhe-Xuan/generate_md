"""Compare original and relaxed candidates using identical references and predictors."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'references'), str(ROOT/'postprocess'), str(ROOT)]
from assess import corrected_entry, load_relaxed, novel_status, stable_status, hull_with_timeout
from inventory import sha256
from paired_statistics import conjunction, index_rows, property_flags, summarize
from summarize_section21 import boolean


def geometry_valid(structure):
    return not any(neighbor.nn_distance < .5 for neighbors in structure.get_all_neighbors(.5)
                   for neighbor in neighbors)


def excluded_references(inventory, selection):
    """Previously failed proposals may need phases outside the relaxation scope."""
    from pymatgen.core import Structure
    excluded = set(inventory.get('scope_excluded_reference_ids', []))
    if not excluded:
        return []
    path = selection/'reference-structures.jsonl.gz'
    if sha256(path) != inventory['reference_records_sha256']:
        raise ValueError('Selected reference source changed.')
    records = []
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            record = json.loads(line)
            if record['material_id'] in excluded:
                structure = Structure.from_dict(record['structure'])
                records.append(dict(candidate_id=record['material_id']+'.vasp',
                    elements=record['chemsys'].split('-'), formula=structure.composition.reduced_formula))
    if len(records) != len(excluded) or len({r['candidate_id'] for r in records}) != len(excluded):
        raise ValueError('Excluded reference coverage audit failed.')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--singlepoint', type=Path, required=True)
    parser.add_argument('--predictor', type=Path, required=True)
    parser.add_argument('--author-csv', type=Path, required=True)
    parser.add_argument('--relaxation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--gap-target', type=float, default=2.5)
    parser.add_argument('--gap-tolerance', type=float, default=.2)
    parser.add_argument('--eform-ceiling', type=float, default=-.5)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit assessment through Slurm.')
    from pymatgen.core import Structure, Composition
    from pymatgen.analysis.structure_matcher import StructureMatcher
    inventory = json.loads(args.inventory.read_text())
    candidates = inventory['candidates']
    ids = [c['candidate_id'] for c in candidates]
    if inventory['denominator'] != len(ids):
        raise ValueError('Fixed denominator mismatch.')
    with (args.predictor/'predictions.csv').open(newline='', encoding='utf8') as stream:
        predictions = index_rows(list(csv.DictReader(stream)), ids)
    if sha256(args.author_csv) != inventory['source_csv_sha256']:
        raise ValueError('Original author CSV provenance mismatch.')
    with args.author_csv.open(newline='', encoding='utf8') as stream:
        original_rows = index_rows([dict(r, candidate_id=r['original_fnames']) for r in csv.DictReader(stream)], ids)
    control = json.loads((args.predictor/'summary.json').read_text())
    if control['control_passed'] is not True or control['candidate_count'] != len(ids):
        raise ValueError('Same-predictor original-geometry control did not pass.')
    energies = index_rows(json.loads((args.singlepoint/'candidates.json').read_text()), ids)
    environment = json.loads((args.singlepoint/'environment.json').read_text())
    sources = json.loads((ROOT/'postprocess/sources.json').read_text())
    if (environment['inventory_sha256'] != sha256(args.inventory)
            or environment['checkpoint_sha256'] != sources['mattersim']['checkpoint_sha256']):
        raise ValueError('Single-point provenance mismatch.')
    args.output.mkdir(parents=True, exist_ok=False)
    reference_inventory = json.loads((args.selection/'mattersim-inventory.json').read_text())
    excluded = excluded_references(reference_inventory, args.selection)
    references, geometries, formulas, failures = {}, {}, defaultdict(list), []
    for candidate in reference_inventory['candidates']:
        identifier = candidate['candidate_id']
        formulas[Composition(candidate['formula']).reduced_formula].append(identifier)
        try:
            path, record = load_relaxed(candidate, args.selection/'mattersim-relaxation')
            if record['checkpoint_sha256'] != environment['checkpoint_sha256']:
                raise ValueError('Reference and original geometries use different MatterSim checkpoints.')
            geometries[identifier] = Structure.from_file(path)
            references[identifier] = corrected_entry(path, record, identifier)
        except Exception as error:
            failures.append(dict(candidate_id=identifier, error=repr(error)))
    for candidate in excluded:
        formulas[candidate['formula']].append(candidate['candidate_id'])
    matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5, primitive_cell=True,
                               scale=True, attempt_supercell=True)
    seen = {'before': [], 'after': []}
    paired_seen = {'before': [], 'after': []}
    rows = []
    for candidate in candidates:
        identifier = candidate['candidate_id']
        if sha256(candidate['source']) != candidate['sha256'] or energies[identifier]['input_sha256'] != candidate['sha256']:
            raise ValueError('Original input provenance mismatch.')
        pred = predictions[identifier]
        neutral = boolean(original_rows[identifier]['is_neutral_cmn'])
        for name, key in (('gap','original_gap'), ('eform','original_eform')):
            if abs(float(pred[name+'_before'])-candidate[key]) > control['control_tolerance']:
                raise ValueError('Same-predictor control disagrees with candidate inventory.')
        original_structure = Structure.from_file(candidate['source'])
        structures = {'before': original_structure}
        records = {'before': energies[identifier]}
        row = dict(candidate_id=identifier, input_sha256=candidate['sha256'],
                   before={}, after={}, relaxation_converged=False, composition_neutral=neutral)
        try:
            relaxed_path, relaxed_record = load_relaxed(candidate, args.relaxation)
            if relaxed_record['checkpoint_sha256'] != environment['checkpoint_sha256']:
                raise ValueError('Proposal and original energies use different MatterSim checkpoints.')
            structures['after'] = Structure.from_file(relaxed_path)
            if structures['after'].composition != original_structure.composition:
                raise ValueError('Relaxation changed composition.')
            records['after'] = relaxed_record
            row['relaxed_sha256'] = relaxed_record['relaxed_sha256']
            row['relaxation_converged'] = True
        except Exception as error:
            row['after']['error'] = repr(error)
        if row['relaxation_converged'] and 'initial_energy_eV' in records['after'] and records['before']['status'] == 'completed':
            discrepancy = abs(records['before']['final_energy_eV']-records['after']['initial_energy_eV'])/len(original_structure)
            row['initial_energy_control_delta_eV_atom'] = discrepancy
            if discrepancy > .001:
                raise ValueError('Original single-point disagrees with relaxation initial energy by >1meV/atom.')
        elements = set(candidate['elements'])
        expected = [c['candidate_id'] for c in reference_inventory['candidates'] if set(c['elements']) <= elements]
        missing_scope = [c['candidate_id'] for c in excluded if set(c['elements']) <= elements]
        expected.extend(missing_scope)
        conversion_errors = [c for c in reference_inventory['conversion_failures']
                             if set(c['chemsys'].split('-')) <= elements]
        available = [references[i] for i in expected if i in references]
        terminals = {next(iter(e.composition.elements)).symbol for e in available if len(e.composition.elements) == 1}
        complete = len(available) == len(expected) and not conversion_errors and elements <= terminals
        row.update(reference_coverage_complete=complete, expected_references=len(expected)+len(conversion_errors),
                   available_references=len(available), missing_terminals=sorted(elements-terminals),
                   unrelaxed_scope_reference_ids=missing_scope)
        for stage, structure in structures.items():
            values = row[stage]
            values['unique'] = not any(matcher.fit(structure, previous) for previous in seen[stage])
            if values['unique']:
                seen[stage].append(structure)
            if row['relaxation_converged']:
                values['paired_unique'] = not any(matcher.fit(structure, previous) for previous in paired_seen[stage])
                if values['paired_unique']:
                    paired_seen[stage].append(structure)
            formula_ids = formulas.get(structure.composition.reduced_formula, [])
            values['novel'], values['matching_relaxed_mp_id'] = novel_status(
                structure, formula_ids, geometries, matcher, not conversion_errors)
            try:
                if stage == 'before' and records[stage]['status'] != 'completed':
                    raise ValueError('Original single-point calculation failed.')
                record = records[stage]
                values['energy_eV_atom'] = record['final_energy_eV']/len(structure)
                if not elements <= terminals:
                    raise ValueError('Missing elemental terminals.')
                path = Path(candidate['source']) if stage == 'before' else relaxed_path
                entry = corrected_entry(path, record, identifier)
                result = hull_with_timeout(available, entry)
                if 'error' in result:
                    raise ValueError(result['error'])
                values.update(result)
                values['stable'] = stable_status(values['hull_eV_atom'], complete)
            except Exception as error:
                values.update(stable=None, hull_error=repr(error))
            try:
                gap, ef = float(pred['gap_'+stage]), float(pred['eform_'+stage])
                if stage == 'after' and pred['relaxation_status'] != 'converged':
                    raise ValueError('Property record is not from a converged relaxed geometry.')
                values['geometry_valid'] = geometry_valid(structure)
                values.update(property_flags(gap, ef, neutral and values['geometry_valid'],
                    args.gap_target, args.gap_tolerance, args.eform_ceiling))
            except Exception as error:
                values['property_error'] = repr(error)
            values['sun'] = conjunction(values.get('stable'), values.get('unique'), values.get('novel'))
            values['stable_and_property'] = conjunction(values.get('stable'), values.get('property_success'))
            values['sun_and_property'] = conjunction(values['sun'], values.get('property_success'))
        rows.append(row)
        (args.output/'candidates.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(candidate_id=identifier, converged=row['relaxation_converged'],
                             before_stable=row['before'].get('stable'), after_stable=row['after'].get('stable'))), flush=True)
    paired = []
    for row in rows:
        if row['relaxation_converged']:
            pair = dict(row, before=dict(row['before']), after=dict(row['after']))
            for stage in ('before','after'):
                values = pair[stage]
                values['unique'] = values['paired_unique']
                values['sun'] = conjunction(values.get('stable'), values['unique'], values.get('novel'))
                values['sun_and_property'] = conjunction(values['sun'], values.get('property_success'))
            paired.append(pair)
    report = dict(benchmark=args.benchmark, job_id=os.environ['SLURM_JOB_ID'],
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  inventory_sha256=sha256(args.inventory),
                  singlepoint_metadata=environment, predictor_control=control,
                  predictor_csv_sha256=sha256(args.predictor/'predictions.csv'),
                  author_csv_sha256=sha256(args.author_csv),
                  reference_inventory_sha256=sha256(args.selection/'mattersim-inventory.json'),
                  full_inventory=summarize(rows), converged_pairs=summarize(paired),
                  reference_failures=failures, reference_conversion_failures=reference_inventory['conversion_failures'],
                  selected_but_unrelaxed_reference_count=len(excluded),
                  thresholds=dict(gap_target=args.gap_target, gap_tolerance=args.gap_tolerance,
                                  eform_strict_upper=args.eform_ceiling, hull_strict_upper=.1),
                  interpretation='Identical relaxed MP reference pool, MatterSim5M and MP2020 correction algorithm in both stages. '
                  'Thermodynamic stability only; does not prove phonon or finite-temperature stability. '
                  'Paired uniqueness is recomputed on the identical converged cohort. '
                  'Missing entries remain unknown; pass rates with unknowns are bounds, not exact rates.')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# MatterSim before/after: '+args.benchmark, '',
             'Both stages use identical candidates, predictors, thresholds and relaxed MP references.',
             'Pass/total retains unknown outcomes in the denominator; percentages are lower bounds.',
             'Delta of lower bounds is not an exact change if unknown outcomes remain.', '']
    for title, cohort in (('Full original inventory',report['full_inventory']),
                          ('Identical converged pairs',report['converged_pairs'])):
        lines.extend(['## '+title, '', '| Metric | Before pass/total | After pass/total | Change (pp, lower bounds) | Unknown before/after | Gained/lost |',
                      '|---|---:|---:|---:|---:|---:|'])
        for metric, values in cohort['metrics'].items():
            b, a, t = values['before'], values['after'], values['transitions']
            delta = values['delta_lower_bound_percentage_points']
            delta_text = f'{delta:+.2f}' if delta is not None else 'unknown'
            lines.append(f"| {metric} | {b['passed']}/{cohort['denominator']} | {a['passed']}/{cohort['denominator']} | "
                         f"{delta_text} | {b['unknown']}/{a['unknown']} | {t['gained']}/{t['lost']} |")
        lines.extend(['', '| Continuous metric | Pair count | Before mean | After mean | Mean change | Median change |',
                      '|---|---:|---:|---:|---:|---:|'])
        for metric, values in cohort['continuous'].items():
            cells = [str(values[k]) for k in ('paired_count','before_mean','after_mean','delta_mean','delta_median')]
            lines.append('| '+metric+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines.extend([report['interpretation'], '', 'DFT remains paused.', ''])
    (args.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
