"""Build a private, identified PBE54 cache and audit the author's DFT input sets."""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
from inventory import sha256


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit through Slurm.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from importlib.metadata import version
    from pymatgen.io.vasp.inputs import Potcar
    inventory = json.loads(args.inventory.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    library = args.output/'psp/POT_GGA_PAW_PBE_54'
    library.mkdir(parents=True, exist_ok=True)
    exported = []
    for symbol, records in inventory['available_pbe54'].items():
        exported_this = False
        for record in records:
            if sha256(record['source']) != record['sha256_file']:
                raise ValueError('Installed potential file changed.')
            for potential in Potcar.from_file(record['source']):
                functionals, symbols = potential.identify_potcar(mode='data')
                if symbol in symbols and 'PBE_54' in functionals:
                    target = library/('POTCAR.'+symbol+'.gz')
                    with gzip.open(target, 'wt') as stream:
                        stream.write(potential.data)
                    exported.append(dict(symbol=symbol, source=record['source'],
                                         source_sha256=record['sha256_file'],
                                         cache_sha256=sha256(target)))
                    exported_this = True
                    break
            if exported_this:
                break
    os.environ['PMG_VASP_PSP_DIR'] = str((args.output/'psp').resolve())
    # Set the already-imported pymatgen settings explicitly as well.
    from pymatgen.core import SETTINGS
    SETTINGS['PMG_VASP_PSP_DIR'] = os.environ['PMG_VASP_PSP_DIR']
    SETTINGS['PMG_DEFAULT_FUNCTIONAL'] = 'PBE_54'
    from pymatgen.core import Structure
    from paper_dft import make_paper_maker, leaf_jobs, stage_makers
    maker = make_paper_maker()
    (args.output/'maker.json').write_text(json.dumps(maker.as_dict(), indent=2, default=str)+'\n')
    ready = []
    for candidate in inventory['candidates']:
        missing = sorted(set(candidate['potcar_symbols'])-{r['symbol'] for r in exported})
        row = dict(candidate_id=candidate['candidate_id'], missing_potcars=missing)
        if not missing:
            structure = Structure.from_file(candidate['source'])
            from pymatgen.io.vasp.sets import MPRelaxSet
            potentials = MPRelaxSet(structure, user_potcar_functional='PBE_54').potcar
            if len(potentials) != len(candidate['potcar_symbols']):
                raise RuntimeError('Incomplete candidate POTCAR construction.')
            stage_settings = []
            for stage in stage_makers(maker):
                inputs = stage.input_set_generator.get_input_set(structure)
                stage_settings.append(dict(name=stage.name,
                    generator=type(stage.input_set_generator).__name__,
                    potcar_functional=stage.input_set_generator.potcar_functional,
                    incar=dict(inputs.incar)))
            row['stage_settings'] = stage_settings
            flow = maker.make(structure)
            row['flow_name'] = flow.name
            row['job_count'] = len(list(leaf_jobs(flow)))
            ready.append(row)
    report = dict(python_packages={n:version(n) for n in ['atomate2','pymatgen','jobflow','custodian','mp-api','emmet-core']},
                  exported_pbe54=exported, candidate_denominator=128,
                  eligible_candidates=ready, missing_symbols=inventory['missing_pbe54'])
    (args.output/'preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
