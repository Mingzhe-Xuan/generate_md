"""Evaluate MatterSim/MP2020 hulls and novelty with explicit reference coverage."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_status(hull, complete):
    if hull is None:
        return None
    # Missing competitors can only lower the hull and raise a candidate's E_hull.
    if hull >= .1:
        return False
    return True if complete else None


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
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        phase = PhaseDiagram(entries+[candidate])
        connection.send(dict(hull_eV_atom=float(phase.get_e_above_hull(candidate))))
    except Exception as error:
        connection.send(dict(error=repr(error)))
    finally:
        connection.close()


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
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit through Slurm.')
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    proposal_root = Path('results/smoacs/postprocess-206')
    inventory = json.loads((proposal_root/'inventory-249/inventory.json').read_text())
    reference_inventory = json.loads((args.selection/'mattersim-inventory.json').read_text())
    output = args.selection/('assessment-'+os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    references, reference_errors = {}, []
    for candidate in reference_inventory['candidates']:
        try:
            path, record = load_relaxed(candidate, args.selection/'mattersim-relaxation')
            references[candidate['candidate_id']] = corrected_entry(path, record, candidate['candidate_id'])
        except Exception as error:
            reference_errors.append(dict(candidate_id=candidate['candidate_id'], error=repr(error)))
    (output/'reference-errors.json').write_text(json.dumps(reference_errors, indent=2)+'\n')
    rows, structures, by_chemistry = [], {}, defaultdict(list)
    for candidate in inventory['candidates']:
        identifier = candidate['candidate_id']
        row = dict(candidate_id=identifier, stable=None, novel=None, unique=None,
                   sun=None, hull_eV_atom=None, bg_and_sun=None)
        rows.append(row)
        try:
            path, record = load_relaxed(candidate, proposal_root/'relaxation')
            structure = Structure.from_file(path)
            structures[identifier] = structure
            by_chemistry[frozenset(candidate['elements'])].append(identifier)
            row['status'] = 'awaiting_reference_check'
        except Exception as error:
            row.update(status='proposal_relaxation_failed', error=repr(error), sun=False, bg_and_sun=False)
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
        row['novel'] = True
    row_by_id = {row['candidate_id']:row for row in rows}
    with gzip.open(args.selection/'novelty-structures.jsonl.gz', 'rt') as stream:
        for line in stream:
            reference = json.loads(line)
            ids = by_chemistry.get(frozenset(reference['chemsys'].split('-')), [])
            if not ids or not any(row_by_id[i]['novel'] for i in ids):
                continue
            other = Structure.from_dict(reference['structure'])
            for identifier in ids:
                row = row_by_id[identifier]
                structure = structures[identifier]
                if row['novel'] and structure.composition.reduced_composition == other.composition.reduced_composition:
                    if matcher.fit(structure, other):
                        row.update(novel=False, matching_mp_id=reference['material_id'])
    for candidate in inventory['candidates']:
        row = row_by_id[candidate['candidate_id']]
        if candidate['candidate_id'] not in structures:
            continue
        elements = set(candidate['elements'])
        expected = [c['candidate_id'] for c in reference_inventory['candidates'] if set(c['elements']) <= elements]
        failed_conversion = [r for r in reference_inventory['conversion_failures']
                             if set(r['chemsys'].split('-')) <= elements]
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
        if row['sun'] is False:
            row['bg_and_sun'] = False
        attempts = list((Path('results/smoacs/openmx-206')/Path(candidate['candidate_id']).stem).glob('attempt-*/result.json'))
        if attempts:
            latest = max(attempts, key=lambda p:int(p.parent.name.split('-')[-1]))
            dft = json.loads(latest.read_text())
            if dft['status'] == 'completed' and dft.get('numerical_convergence_passed'):
                row.update(openmx_bandgap_eV=dft['sampled_bandgap_eV'], openmx_job=dft['job_id'])
                row['bg_and_sun'] = bool(row['sun'] and dft['target_gap_success']) if row['sun'] is not None else None
        print(json.dumps(row), flush=True)
        (output/'candidates.json').write_text(json.dumps(rows, indent=2)+'\n')
    summary = dict(denominator=128, stable_known=sum(r['stable'] is not None for r in rows),
                   stable_pass=sum(r['stable'] is True for r in rows),
                   sun_pass=sum(r['sun'] is True for r in rows), sun_unknown=sum(r['sun'] is None for r in rows),
                   bg_and_sun_pass=sum(r['bg_and_sun'] is True for r in rows),
                   snapshot='2026-04-13', method='MatterSim5M/MP2020 + OpenMX alternative BG',
                   reference_failures=len(reference_errors)+len(reference_inventory['conversion_failures']))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
