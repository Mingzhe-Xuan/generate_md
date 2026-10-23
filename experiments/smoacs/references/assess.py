"""Evaluate MatterSim/MP2020 hulls and novelty with explicit reference coverage."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_status(hull, complete):
    if hull is None:
        return None
    # Missing competitors can only lower the hull and raise a candidate's E_hull.
    if hull >= .1:
        return False
    return True if complete else None


def novel_status(proposal, expected_ids, reference_structures, matcher, inputs_complete=True):
    for identifier in expected_ids:
        if identifier in reference_structures and matcher.fit(proposal, reference_structures[identifier]):
            return False, identifier
    if not inputs_complete or any(i not in reference_structures for i in expected_ids):
        return None, None
    return True, None


def corrected_entry(path, record, identifier):
    from pymatgen.core import Structure
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from atomate2.vasp.sets.mp import MPGGARelaxSetGenerator
    structure = Structure.from_file(path)
    settings = MPGGARelaxSetGenerator().get_input_set(structure, potcar_spec=True)
    incar = settings.incar
    symbols = settings.poscar.site_symbols
    hubbards = dict(zip(symbols, incar.get('LDAUU', [0]*len(symbols))))
    hubbard = bool(incar.get('LDAU', False) and any(float(v) != 0 for v in hubbards.values()))
    parameters = dict(run_type='GGA+U' if hubbard else 'GGA', is_hubbard=hubbard,
                      hubbards=hubbards if hubbard else {}, source='mattersim')
    entry = ComputedStructureEntry(structure, record['final_energy_eV'],
                                   parameters=parameters, entry_id=identifier)
    compatibility = MaterialsProject2020Compatibility(check_potcar_hash=False, check_potcar=False)
    processed = compatibility.process_entry(entry, clean=True)
    if processed is None:
        raise ValueError('MP2020Compatibility rejected entry.')
    return processed


def phase_worker(entries, candidate, connection):
    try:
        connection.send(hull_query(entries, candidate))
    except Exception as error:
        connection.send(dict(error=repr(error)))
    finally:
        connection.close()


def hull_query(entries, candidate):
    """Exact point query of the convex envelope, without enumerating all facets."""
    import numpy as np
    from scipy.optimize import linprog
    elements = list(candidate.composition.elements)
    if not entries or any(set(e.composition.elements)-set(elements) for e in entries):
        raise ValueError('Reference entries must belong to the candidate chemical system.')
    matrix = np.array([[entry.composition.get_atomic_fraction(element) for entry in entries]
                       for element in elements], dtype=float)
    target = np.array([candidate.composition.get_atomic_fraction(e) for e in elements])
    energies = np.array([entry.energy_per_atom for entry in entries])
    if not all(np.isfinite(x).all() for x in (matrix, target, energies, candidate.energy_per_atom)):
        raise ValueError('Nonfinite phase-diagram inputs.')
    solution = linprog(energies, A_eq=matrix, b_eq=target, bounds=(0, None), method='highs-ds',
                       options=dict(time_limit=110, primal_feasibility_tolerance=1e-9,
                                    dual_feasibility_tolerance=1e-9))
    if not solution.success or not np.isfinite(solution.fun):
        raise ValueError('Convex-envelope LP failed: '+solution.message)
    if (np.max(np.abs(matrix@solution.x-target)) > 1e-7
            or np.min(solution.x) < -1e-8 or abs(solution.x.sum()-1) > 1e-7):
        raise ValueError('Phase decomposition does not conserve composition.')
    # Including the candidate itself, as PhaseDiagram does, clamps negative E_hull to zero.
    hull = max(0.0, float(candidate.energy_per_atom-solution.fun))
    return dict(hull_eV_atom=hull, reference_hull_energy_eV_atom=float(solution.fun),
                hull_algorithm='composition-balanced linear program, equivalent point query',
                decomposition=[dict(entry_id=e.entry_id, atomic_fraction=float(w))
                               for e,w in zip(entries, solution.x) if w > 1e-8])


def hull_with_timeout(entries, candidate):
    context = mp.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=phase_worker, args=(entries, candidate, writer))
    process.start()
    writer.close()
    if reader.poll(120):
        result = reader.recv()
        process.join(10)
    else:
        result = dict(error='PhaseDiagram exceeded120seconds')
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
    reader.close()
    return result


def load_relaxed(candidate, root):
    folder = root/Path(candidate['candidate_id']).stem
    record = json.loads((folder/'result.json').read_text())
    path = folder/'relaxed.vasp'
    if record['input_sha256'] != candidate['sha256'] or record['status'] != 'converged':
        raise ValueError('Unconverged or stale MatterSim result.')
    if digest(path) != record['relaxed_sha256']:
        raise ValueError('MatterSim relaxed structure digest mismatch.')
    return path, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--proposal-root', type=Path, default=Path('results/smoacs/postprocess-206'))
    parser.add_argument('--output', type=Path, help='Separate output when comparing methods against one pool.')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit through Slurm.')
    from pymatgen.core import Structure, Composition
    from pymatgen.analysis.structure_matcher import StructureMatcher
    proposal_root = args.proposal_root
    inventory = json.loads((proposal_root/'inventory-249/inventory.json').read_text())
    reference_inventory = json.loads((args.selection/'mattersim-inventory.json').read_text())
    output = args.output or args.selection/('assessment-'+os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    references, reference_structures, reference_errors = {}, {}, []
    reference_formula_ids = defaultdict(list)
    for candidate in reference_inventory['candidates']:
        reference_formula_ids[Composition(candidate['formula']).reduced_formula].append(candidate['candidate_id'])
        try:
            path, record = load_relaxed(candidate, args.selection/'mattersim-relaxation')
            reference_structures[candidate['candidate_id']] = Structure.from_file(path)
            references[candidate['candidate_id']] = corrected_entry(path, record, candidate['candidate_id'])
        except Exception as error:
            reference_errors.append(dict(candidate_id=candidate['candidate_id'], error=repr(error)))
    (output/'reference-errors.json').write_text(json.dumps(reference_errors, indent=2)+'\n')
    rows, structures, by_chemistry = [], {}, defaultdict(list)
    for candidate in inventory['candidates']:
        identifier = candidate['candidate_id']
        row = dict(candidate_id=identifier, stable=None, novel=None, unique=None,
                   sun=None, hull_eV_atom=None)
        rows.append(row)
        try:
            path, record = load_relaxed(candidate, proposal_root/'relaxation')
            structure = Structure.from_file(path)
            structures[identifier] = structure
            by_chemistry[frozenset(candidate['elements'])].append(identifier)
            row['status'] = 'awaiting_reference_check'
        except Exception as error:
            row.update(status='proposal_relaxation_failed', error=repr(error), sun=False)
    matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5,
                               primitive_cell=True, scale=True, attempt_supercell=True)
    seen = []
    for row in rows:
        identifier = row['candidate_id']
        if identifier not in structures:
            continue
        structure = structures[identifier]
        row['unique'] = not any(matcher.fit(structure, earlier) for earlier in seen)
        if row['unique']:
            seen.append(structure)
        row['novel_vs_full_raw_MP'] = True
    row_by_id = {row['candidate_id']:row for row in rows}
    with gzip.open(args.selection/'novelty-structures.jsonl.gz', 'rt') as stream:
        for line in stream:
            reference = json.loads(line)
            ids = by_chemistry.get(frozenset(reference['chemsys'].split('-')), [])
            if not ids or not any(row_by_id[i]['novel_vs_full_raw_MP'] for i in ids):
                continue
            other = Structure.from_dict(reference['structure'])
            for identifier in ids:
                row = row_by_id[identifier]
                structure = structures[identifier]
                if row['novel_vs_full_raw_MP'] and structure.composition.reduced_composition == other.composition.reduced_composition:
                    if matcher.fit(structure, other):
                        row.update(novel_vs_full_raw_MP=False, matching_raw_mp_id=reference['material_id'])
    for candidate in inventory['candidates']:
        row = row_by_id[candidate['candidate_id']]
        if candidate['candidate_id'] not in structures:
            continue
        elements = set(candidate['elements'])
        expected = [c['candidate_id'] for c in reference_inventory['candidates'] if set(c['elements']) <= elements]
        failed_conversion = [r for r in reference_inventory['conversion_failures']
                             if set(r['chemsys'].split('-')) <= elements]
        same_formula = reference_formula_ids.get(structures[candidate['candidate_id']].composition.reduced_formula, [])
        novel, matching = novel_status(structures[candidate['candidate_id']], same_formula,
                                       reference_structures, matcher, not failed_conversion)
        row.update(novel=novel, matching_relaxed_mp_id=matching,
                   novelty_basis='MatterSim-relaxed selected MP reference phases, published primary Stage4')
        available = [references[i] for i in expected if i in references]
        terminals = {next(iter(e.composition.elements)).symbol for e in available if len(e.composition.elements) == 1}
        complete = len(available) == len(expected) and not failed_conversion and elements <= terminals
        row.update(expected_references=len(expected)+len(failed_conversion),
                   available_references=len(available), reference_coverage_complete=complete,
                   missing_terminals=sorted(elements-terminals))
        try:
            if not elements <= terminals:
                raise ValueError('Missing elemental terminals.')
            path, record = load_relaxed(candidate, proposal_root/'relaxation')
            entry = corrected_entry(path, record, candidate['candidate_id'])
            hull = hull_with_timeout(available, entry)
            if 'error' in hull:
                raise ValueError(hull['error'])
            row.update(hull)
            row['stable'] = stable_status(row['hull_eV_atom'], complete)
            row['status'] = 'assessed' if complete else 'incomplete_reference_coverage'
        except Exception as error:
            row.update(status='hull_unknown', error=repr(error))
        flags = [row['stable'], row['unique'], row['novel']]
        row['sun'] = False if False in flags else True if all(v is True for v in flags) else None
        print(json.dumps(row), flush=True)
        (output/'candidates.json').write_text(json.dumps(rows, indent=2)+'\n')
    (output/'candidates.json').write_text(json.dumps(rows, indent=2)+'\n')
    summary = dict(denominator=inventory['denominator'], stable_known=sum(r['stable'] is not None for r in rows),
                   stable_pass=sum(r['stable'] is True for r in rows),
                   sun_pass=sum(r['sun'] is True for r in rows), sun_unknown=sum(r['sun'] is None for r in rows),
                   unique_pass=sum(r['unique'] is True for r in rows),
                   novel_pass=sum(r['novel'] is True for r in rows),
                   novel_unknown=sum(r['novel'] is None for r in rows),
                   snapshot='2026-04-13', method='MatterSim5M/MP2020 S.U.N.',
                   convergence_policy='strict actual optimizer convergence; publication wrapper discards its returned flag',
                   reference_failures=len(reference_errors)+len(reference_inventory['conversion_failures']))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
