"""Submit only certified-profile, SUN-screened OpenMX candidates through Slurm."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from resources import digest


def select_rows(rows, calibration, resources):
    if (calibration['status'] != 'completed' or not calibration.get('numerical_convergence_passed')
            or calibration.get('basis_level') != 1 or calibration.get('mesh_increment', 0) != 0):
        raise ValueError('Production blocked: higher-basis calibration is not certified.')
    covered = set(resources['element_covered_ids'])
    eligible, statuses = [], []
    for row in rows:
        identifier = row['candidate_id']
        if row['sun'] is not True:
            status = 'not_sun' if row['sun'] is False else 'sun_unknown'
        elif identifier not in covered:
            status = 'blocked_missing_pbe19_resources'
        else:
            status = 'eligible'
            eligible.append(identifier)
        statuses.append(dict(candidate_id=identifier, status=status))
    if len(rows) != 128 or len({r['candidate_id'] for r in rows}) != 128:
        raise ValueError('Expected all128 unique original candidates.')
    return eligible, statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--physics-manifest', type=Path, required=True)
    parser.add_argument('--u-probe', type=Path, required=True)
    args = parser.parse_args()
    job = os.environ['SLURM_JOB_ID']
    output = args.selection/f'openmx-dispatch-{job}'
    output.mkdir(exist_ok=False)
    rows = json.loads((args.assessment/'candidates.json').read_text())
    calibration = json.loads((args.calibration/'result.json').read_text())
    resources = json.loads((args.calibration/'resources.json').read_text())
    physics = json.loads(args.physics_manifest.read_text())
    probe = json.loads((args.u_probe/'result.json').read_text())
    if probe.get('passed') is not True or probe['physics_manifest_sha256'] != digest(args.physics_manifest):
        raise ValueError('Production blocked: MP-derived Hubbard-U probe not validated.')
    eligible, statuses = select_rows(rows, calibration, resources)
    for record in statuses:
        if record['status'] == 'eligible' and record['candidate_id'] not in physics['records']:
            record['status'] = 'blocked_missing_MP_physics'
    eligible = [i for i in eligible if i in physics['records']]
    manifest = dict(denominator=128, candidate_ids=eligible, statuses=statuses,
                    calibration_job=calibration['job_id'], basis_level=1,
                    physics_manifest_sha256=digest(args.physics_manifest),
                    method='OpenMX3.9 with MP-derived Ueff/moments; PAO projectors differ from VASP PAW')
    manifest_path = (output/'manifest.json').resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    if not eligible:
        print('NO_ELIGIBLE_SUN_CANDIDATES', flush=True)
        return
    export = ('ALL,SMOACS_OPENMX_SELECTION='+str(manifest_path)
              + ',SMOACS_OPENMX_BASIS_LEVEL=1,SMOACS_OPENMX_MESH_INCREMENT=0'
              + ',SMOACS_OPENMX_PHYSICS='+str(args.physics_manifest.resolve())
              + ',SMOACS_OPENMX_ID=,SMOACS_OPENMX_REUSE=')
    submitted = subprocess.check_output(['sbatch', '--parsable', '--array=0-'+str(len(eligible)-1)+'%4',
                       '--export='+export, 'experiments/smoacs/openmx/submit.sbatch'], text=True).strip()
    if not re.fullmatch(r'\d+(?:;[^\s]+)?', submitted):
        raise ValueError('Unexpected sbatch job identifier.')
    array_id = submitted.split(';')[0]
    final = subprocess.check_output(['sbatch', '--parsable', '--dependency=afterany:'+array_id,
                  '--export=ALL,SMOACS_REFERENCE_SELECTION='+str(args.selection),
                  'experiments/smoacs/references/assess.sbatch'], text=True).strip()
    manifest.update(array_job=array_id, final_assessment_job=final)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
