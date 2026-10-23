"""Actual FeO+U input/SCF capability probe, not a material benchmark result."""
import json
import os
from pathlib import Path
import re
import subprocess

from formats import input_text, parse_output, validate_log
from resources import select_species, digest


def main():
    from ase import Atoms
    manifest = Path(os.environ['SMOACS_OPENMX_PHYSICS']).resolve()
    physics = json.loads(manifest.read_text())['probe_FeO']
    if abs(physics['hubbards']['Fe']['U_eff_eV']-5.3) > 1e-8:
        raise ValueError('Paper MP FeO Ueff is not the expected5.3eV.')
    folder = Path('results/smoacs/openmx-206')/('u-probe-'+os.environ['SLURM_JOB_ID'])
    folder.mkdir(exist_ok=False)
    data = Path('/opt/openmx3.9/DFT_DATA19')
    species = {e:select_species(e, data) for e in ('Fe','O')}
    atoms = Atoms('FeO', scaled_positions=[[0,0,0],[.5,.5,.5]],
                  cell=[[0,2.15,2.15],[2.15,0,2.15],[2.15,2.15,0]], pbc=True)
    (folder/'input.dat').write_text(input_text(atoms, species, str(data), [3,3,3], relax=False, physics=physics))
    with (folder/'calc.log').open('w') as log, (folder/'stderr.txt').open('w') as err:
        subprocess.run(['srun','--mpi=pmi2','--cpu-bind=cores','-n',os.environ['SLURM_NTASKS'],
                        '/opt/openmx3.9/source/openmx','input.dat'], cwd=folder.resolve(), stdout=log, stderr=err, check=True)
    status = validate_log((folder/'calc.log').read_text(), False)
    text = (folder/'calc.out').read_text()
    parsed = parse_output(text)
    values = re.findall(r'^\s*Uhub\.\s+([\d.Ee+\-]+)', text, re.M)
    if not values or abs(float(values[-1])) < 1e-8:
        raise ValueError('No nonzero Hubbard energy contribution in actual output.')
    result = dict(passed=True, purpose='synthetic FeO U-input/SCF probe only',
                  job_id=os.environ['SLURM_JOB_ID'], physics=physics, physics_manifest_sha256=digest(manifest),
                  input_sha256=digest(folder/'input.dat'), output_sha256=digest(folder/'calc.out'),
                  Hubbard_energy_Ha=float(values[-1]), **status, **parsed)
    (folder/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
