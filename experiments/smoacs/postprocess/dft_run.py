"""Execute and strictly assess the paper MP double-relax/static on one candidate."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import traceback
from inventory import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--candidate-id', required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit DFT through Slurm.')
    root = args.root.resolve()
    inventory = json.loads(args.inventory.read_text())
    candidate = next(c for c in inventory['candidates'] if c['candidate_id'] == args.candidate_id)
    relaxed_record = json.loads((root/'relaxation'/Path(args.candidate_id).stem/'result.json').read_text())
    if relaxed_record['status'] != 'converged':
        raise ValueError('DFT requires converged MatterSim input.')
    if relaxed_record['input_sha256'] != candidate['sha256'] or sha256(relaxed_record['relaxed_path']) != relaxed_record['relaxed_sha256']:
        raise ValueError('Candidate provenance mismatch.')
    preflight = json.loads((root/'dft-config/preflight.json').read_text())
    if args.candidate_id not in {c['candidate_id'] for c in preflight['eligible_candidates']}:
        raise ValueError('Required matching PBE54 potentials unavailable.')
    out = root/'dft'/Path(args.candidate_id).stem/('attempt-'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    config = out/'atomate2.yaml'
    launcher = 'srun --mpi=pmi2 --cpu-bind=cores -n '+str(int(os.environ['SLURM_NTASKS']))
    config.write_text('VASP_CMD: '+launcher+' /opt/vasp.5.4.4/bin/vasp_std\n'
                      'VASP_GAMMA_CMD: '+launcher+' /opt/vasp.5.4.4/bin/vasp_gam\n')
    os.environ['ATOMATE2_CONFIG_FILE'] = str(config)
    os.environ['PMG_VASP_PSP_DIR'] = str(root/'dft-config/psp')
    os.environ['PMG_DEFAULT_FUNCTIONAL'] = 'PBE_54'
    from pymatgen.core import Structure
    from pymatgen.io.vasp.outputs import Vasprun
    from maggma.stores import MemoryStore
    from jobflow import JobStore, run_locally
    from paper_dft import make_paper_maker, leaf_jobs
    from safe_custodian import scoped_vasp_jobs
    from atomate2 import SETTINGS
    structure = Structure.from_file(relaxed_record['relaxed_path'])
    if structure.composition != Structure.from_file(candidate['source']).composition:
        raise ValueError('Composition changed during geometry relaxation.')
    maker = make_paper_maker()
    flow = maker.make(structure)
    jobs = list(leaf_jobs(flow))
    if len(jobs) != 3:
        raise ValueError('Expected exactly two relaxation stages and one static.')
    (out/'maker.json').write_text(json.dumps(maker.as_dict(), indent=2, default=str)+'\n')
    record = dict(candidate_id=args.candidate_id, job_id=os.environ['SLURM_JOB_ID'],
                  original_input_sha256=candidate['sha256'],
                  mattersim_relaxed_sha256=relaxed_record['relaxed_sha256'],
                  status='running', vasp_command=SETTINGS.VASP_CMD,
                  original_predictor_gap_eV=candidate['original_gap'],
                  stages=[])
    (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    store = JobStore(MemoryStore())
    try:
        with scoped_vasp_jobs(out):
            run_locally(flow, store=store, create_folders=True, root_dir=out/'workflow', ensure_success=True)
        for job in jobs:
            task = store.get_output(job.uuid, load=True)
            directory = task['dir_name'] if isinstance(task, dict) else task.dir_name
            directory = Path(directory.split(':')[-1])
            xml = directory/'vasprun.xml.gz'
            if not xml.is_file():
                xml = directory/'vasprun.xml'
            vr = Vasprun(xml, parse_dos=False, parse_projected_eigen=False, parse_potcar_file=False)
            stage = dict(name=job.name, directory=str(directory),
                         electronic_converged=bool(vr.converged_electronic),
                         ionic_converged=bool(vr.converged_ionic),
                         final_energy_eV=float(vr.final_energy),
                         incar={k:vr.incar.get(k) for k in ['ENCUT','EDIFF','EDIFFG','ISIF','NSW','ISPIN','LDAU','LDAUU','NEDOS']},
                         kpoint_count=len(vr.actual_kpoints), vasprun_sha256=sha256(xml))
            record['stages'].append(stage)
            if not vr.converged or not math.isfinite(float(vr.final_energy)):
                raise RuntimeError('Unconverged or non-finite DFT stage: '+job.name)
        gap, cbm, vbm, direct = vr.eigenvalue_band_properties
        if not math.isfinite(float(gap)) or int(vr.incar.get('NEDOS', 0)) != 2001:
            raise RuntimeError('Final static does not meet the paper bandgap evaluation settings.')
        vr.final_structure.to(filename=str(out/'dft-relaxed.vasp'), fmt='poscar')
        record.update(status='completed', bandgap_eV=float(gap), direct_gap=bool(direct),
                      cbm_eV=float(cbm), vbm_eV=float(vbm),
                      energy_eV=float(vr.final_energy), energy_per_atom_eV=float(vr.final_energy)/len(vr.final_structure),
                      target_gap_success=abs(float(gap)-2.5)<=0.2,
                      relaxed_structure_sha256=sha256(out/'dft-relaxed.vasp'),
                      hull_status='unknown_without_consistent_reference_phases')
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
