"""Derive compound-specific U and initial moments using the paper MP generator."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings_for_structure(structure):
    from atomate2.vasp.sets.mp import MPGGARelaxSetGenerator
    inputs = MPGGARelaxSetGenerator().get_input_set(structure, potcar_spec=True)
    incar = inputs.incar
    symbols = inputs.poscar.site_symbols
    uvalues = incar.get('LDAUU', [0]*len(symbols))
    jvalues = incar.get('LDAUJ', [0]*len(symbols))
    angular = incar.get('LDAUL', [-1]*len(symbols))
    if incar.get('LDAU', False) and incar.get('LDAUTYPE', 2) != 2:
        raise ValueError('Only Dudarev MP U-J can be mapped to OpenMX Type1.')
    hubbards = {}
    for e,u,j,l in zip(symbols, uvalues, jvalues, angular):
        effective = float(u)-float(j) if incar.get('LDAU', False) else 0.0
        if effective and int(l) not in (0,1,2,3):
            raise ValueError('Unsupported Hubbard angular channel.')
        hubbards[e] = dict(U_eff_eV=effective, angular_momentum=int(l))
    moments = {}
    values = incar.get('MAGMOM', [0.6]*len(inputs.poscar.structure))
    if len(values) != len(inputs.poscar.structure):
        raise ValueError('Unexpected MP initial moment array.')
    for site, moment in zip(inputs.poscar.structure, values):
        e = site.specie.symbol
        if isinstance(moment, (list, tuple)):
            raise ValueError('Noncollinear MP moments require an explicit protocol.')
        if e in moments and abs(moments[e]-float(moment)) > 1e-8:
            raise ValueError('Site-specific magnetic order cannot be mapped by element.')
        moments[e] = float(moment)
    return dict(hubbards=hubbards, initial_moments_muB=moments,
                source='paper MPGGARelaxSetGenerator',
                projector_mapping='OpenMX dual, first radial orbital of target l, others zero',
                projector_equivalent_to_PAW=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit through Slurm.')
    from pymatgen.core import Structure, Lattice
    root = Path('results/smoacs/postprocess-206')
    inventory = json.loads((root/'inventory-249/inventory.json').read_text())
    records, errors = {}, {}
    for candidate in inventory['candidates']:
        identifier = candidate['candidate_id']
        folder = root/'relaxation'/Path(identifier).stem
        record = json.loads((folder/'result.json').read_text())
        if record['status'] != 'converged':
            continue
        try:
            if record['input_sha256'] != candidate['sha256'] or digest(folder/'relaxed.vasp') != record['relaxed_sha256']:
                raise ValueError('Proposal provenance mismatch.')
            result = settings_for_structure(Structure.from_file(folder/'relaxed.vasp'))
            result.update(original_sha256=candidate['sha256'], mattersim_sha256=record['relaxed_sha256'])
            records[identifier] = result
        except Exception as error:
            errors[identifier] = repr(error)
    output = dict(records=records, errors=errors, denominator=128,
                  versions={p:importlib.metadata.version(p) for p in ('atomate2','pymatgen')},
                  role='paper compound-specific electronic-structure inputs')
    probe = Structure(Lattice([[0,2.15,2.15],[2.15,0,2.15],[2.15,2.15,0]]), ['Fe','O'], [[0,0,0],[.5,.5,.5]])
    output['probe_FeO'] = settings_for_structure(probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Do not overwrite an existing physics manifest.')
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(ready=len(records), errors=errors)), flush=True)


if __name__ == '__main__':
    main()
