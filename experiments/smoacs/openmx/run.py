"""Run one immutable CPU206 candidate through the alternative OpenMX protocol."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import shutil
import traceback

from formats import input_text, parse_output, restart_atoms, validate_log
from resources import audit, digest, expand_basis


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-id', required=True)
    parser.add_argument('--inventory', type=Path, default=Path('results/smoacs/postprocess-206/inventory-249/inventory.json'))
    parser.add_argument('--data', type=Path, default=Path('/opt/openmx3.9/DFT_DATA19'))
    parser.add_argument('--executable', type=Path, default=Path('/opt/openmx3.9/source/openmx'))
    parser.add_argument('--root', type=Path, default=Path('results/smoacs/openmx-206'))
    parser.add_argument('--reuse-attempt', type=Path)
    parser.add_argument('--basis-level', type=int, default=0)
    parser.add_argument('--mesh-increment', type=int, default=0)
    parser.add_argument('--physics-manifest', type=Path)
    parser.add_argument('--starting-attempt', type=Path)
    args = parser.parse_args()
    job = os.environ['SLURM_JOB_ID']
    out = (args.root/Path(args.candidate_id).stem/f'attempt-{job}').resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(candidate_id=args.candidate_id, job_id=job, status='running',
                  method='OpenMX3.9/PBE19 alternative', stages=[], paper_equivalent=False)
    result['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    result['source_hashes'] = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')
                               if not p.name.startswith('test_')}
    dump(out/'result.json', result)
    try:
        from ase.io import read, write
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.io.vasp.inputs import Kpoints
        inventory = json.loads(args.inventory.read_text())
        resources = audit(inventory, args.data)
        expand_basis(resources['species'], args.basis_level)
        dump(out/'resources.json', resources)
        candidate = next(c for c in inventory['candidates'] if c['candidate_id'] == args.candidate_id)
        if args.candidate_id not in resources['element_covered_ids']:
            raise ValueError('Missing valence-matched PBE19 resources.')
        original = Path(candidate['source'])
        if digest(original) != candidate['sha256']:
            raise ValueError('Original input hash mismatch.')
        relaxation_dir = Path('results/smoacs/postprocess-206/relaxation')/Path(args.candidate_id).stem
        relaxation = json.loads((relaxation_dir/'result.json').read_text())
        source = relaxation_dir/'relaxed.vasp'
        if (relaxation['status'] != 'converged' or relaxation['input_sha256'] != candidate['sha256']
                or digest(source) != relaxation['relaxed_sha256']):
            raise ValueError('Missing converged/hash-matched MatterSim input.')
        result.update(original_sha256=candidate['sha256'], mattersim_sha256=digest(source),
                      inventory_sha256=digest(args.inventory), executable_sha256=digest(args.executable))
        if args.reuse_attempt:
            args.reuse_attempt = args.reuse_attempt.resolve()
            if args.reuse_attempt.parent != out.parent or args.reuse_attempt == out:
                raise ValueError('Reuse attempt must be a previous attempt of this candidate.')
            previous = json.loads((args.reuse_attempt/'result.json').read_text())
            for key in ('original_sha256', 'mattersim_sha256', 'inventory_sha256', 'executable_sha256'):
                if previous[key] != result[key]:
                    raise ValueError('Reuse provenance mismatch: '+key)
            previous_resources = json.loads((args.reuse_attempt/'resources.json').read_text())
            for element in candidate['elements']:
                for key in ('vps_sha256', 'pao_sha256'):
                    if previous_resources['species'][element][key] != resources['species'][element][key]:
                        raise ValueError('Reused stage resource hash mismatch.')
        atoms = read(source, format='vasp')
        physics = None
        if args.physics_manifest:
            manifest = json.loads(args.physics_manifest.read_text())
            physics = manifest['records'][args.candidate_id]
            if (physics['original_sha256'] != candidate['sha256']
                    or physics['mattersim_sha256'] != digest(source)):
                raise ValueError('Compound-specific physics manifest is stale.')
            result.update(compound_physics=physics, physics_manifest_sha256=digest(args.physics_manifest))
        write(out/'mattersim-input.vasp', atoms, format='vasp')
        grid = list(Kpoints.automatic_density(AseAtomsAdaptor.get_structure(atoms), 503).kpts[0])
        if args.mesh_increment < 0 or args.mesh_increment % 2:
            raise ValueError('Mesh increment must be a nonnegative even integer.')
        grid = [max(3, int(n) if int(n) % 2 else int(n)+1)+args.mesh_increment for n in grid]
        result.update(basis_level=args.basis_level, mesh_increment=args.mesh_increment,
                      hubbard_U=bool(physics and any(v['U_eff_eV'] for v in physics['hubbards'].values())),
                      initial_moments_source='MP generator' if physics else '1muB/atom calibration')
        result['base_grid'] = grid
        if args.starting_attempt:
            if args.physics_manifest:
                raise ValueError('Calibration warmstarts are forbidden for production physics-manifest runs.')
            previous_path = args.starting_attempt.resolve()
            if previous_path.parent != out.parent or previous_path == out:
                raise ValueError('Starting geometry must come from this same candidate.')
            previous = json.loads((previous_path/'result.json').read_text())
            if (previous['original_sha256'] != result['original_sha256']
                    or previous['mattersim_sha256'] != result['mattersim_sha256']
                    or previous.get('basis_level', 0) >= args.basis_level or previous.get('hubbard_U')):
                raise ValueError('Incompatible calibration warmstart provenance.')
            stage = previous['stages'][1]
            warm = previous_path/'relax2/relaxed.vasp'
            if (stage['name'] != 'relax2' or stage['status'] != 'completed'
                    or stage['generalized_gradient_Ha_Bohr'] >= 1e-4 or digest(warm) != stage['relaxed_sha256']):
                raise ValueError('Calibration starting geometry was not validated.')
            refined = read(warm, format='vasp')
            if Counter(refined.get_chemical_symbols()) != Counter(atoms.get_chemical_symbols()):
                raise ValueError('Calibration warmstart composition differs.')
            atoms = refined
            write(out/'calibration-start.vasp', atoms, format='vasp')
            result['calibration_warmstart'] = dict(attempt=str(previous_path), structure_sha256=digest(warm),
                                                  purpose='numerical profile tuning only')
        launch = ['srun', '--mpi=pmi2', '--cpu-bind=cores', '-n', os.environ['SLURM_NTASKS'], str(args.executable.resolve())]
        dump(out/'result.json', result)
        stages = [('relax1', True, 200, False, grid), ('relax2', True, 200, False, grid),
                  ('static', False, 200, False, grid), ('cutoff-check', False, 250, False, grid),
                  ('basis-check', False, 200, True, grid),
                  ('kmesh-check', False, 200, False, [n+2 for n in grid])]
        for name, relax, cutoff, larger, mesh in stages:
            stage_dir = out/name
            stage_dir.mkdir()
            text = input_text(atoms, resources['species'], str(args.data.resolve()), mesh, relax, cutoff, larger, physics)
            (stage_dir/'input.dat').write_text(text)
            command = launch+['input.dat']
            record = dict(name=name, cutoff_Ry=cutoff, grid=mesh, larger_basis=larger,
                          input_sha256=digest(stage_dir/'input.dat'), status='running')
            result['stages'].append(record)
            dump(out/'result.json', result)
            reusable = args.reuse_attempt/name if args.reuse_attempt else None
            if reusable and (reusable/'calc.out').is_file():
                if digest(reusable/'input.dat') != record['input_sha256']:
                    raise ValueError('Reuse stage input differs.')
                validate_log((reusable/'calc.log').read_text(), relax)
                parse_output((reusable/'calc.out').read_text())
                for filename in ('calc.log', 'calc.out', 'input.dat#'):
                    if (reusable/filename).exists():
                        shutil.copy2(reusable/filename, stage_dir/filename)
                record.update(exit_code=0, reused_from=str(reusable),
                              reused_output_sha256=digest(reusable/'calc.out'))
            else:
                with (stage_dir/'calc.log').open('w') as log, (stage_dir/'stderr.txt').open('w') as err:
                    completed = subprocess.run(command, cwd=stage_dir, stdout=log, stderr=err, check=False)
                record['exit_code'] = completed.returncode
                if completed.returncode != 0:
                    raise RuntimeError(f'{name}: OpenMX exit {completed.returncode}')
            record.update(validate_log((stage_dir/'calc.log').read_text(), relax))
            record.update(parse_output((stage_dir/'calc.out').read_text()))
            record['output_sha256'] = digest(stage_dir/'calc.out')
            record['log_sha256'] = digest(stage_dir/'calc.log')
            if record['sampled_kpoints'] < (int(mesh[0])*int(mesh[1])*int(mesh[2])+1)//2:
                raise ValueError('Truncated k-point eigenvalue output.')
            if relax:
                final = restart_atoms((stage_dir/'input.dat#').read_text())
                if Counter(final.get_chemical_symbols()) != Counter(atoms.get_chemical_symbols()):
                    raise ValueError('Relaxation changed composition.')
                atoms = final
                write(stage_dir/'relaxed.vasp', atoms, format='vasp')
                record['relaxed_sha256'] = digest(stage_dir/'relaxed.vasp')
            record['status'] = 'completed'
            dump(out/'result.json', result)
            print(name, json.dumps(record), flush=True)
        baseline = result['stages'][2]
        checks = [dict(name=r['name'],
                       energy_change_eV_atom=abs(r['energy_eV']-baseline['energy_eV'])/len(atoms),
                       gap_change_eV=abs(r['sampled_bandgap_eV']-baseline['sampled_bandgap_eV']))
                  for r in result['stages'][3:]]
        converged = all(c['energy_change_eV_atom'] <= .01 and c['gap_change_eV'] <= .05 for c in checks)
        result.update(status='completed' if converged else 'numerical_convergence_failed',
                      convergence_checks=checks, numerical_convergence_passed=converged,
                      sampled_bandgap_eV=baseline['sampled_bandgap_eV'],
                      target_gap_success=bool(converged and 2.3 <= baseline['sampled_bandgap_eV'] <= 2.7),
                      stability='unknown_without_consistent_reference_phases')
        write(out/'openmx-relaxed.vasp', atoms, format='vasp')
        dump(out/'result.json', result)
        if not converged:
            raise RuntimeError('Numerical convergence checks failed; retain and investigate.')
    except Exception as error:
        if result['status'] == 'running':
            result['status'] = 'failed'
        result['error'] = repr(error)
        dump(out/'result.json', result)
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
