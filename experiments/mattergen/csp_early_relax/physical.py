"""Strict crystal matching invariant to equivalent supercell representations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from pymatgen.analysis.structure_matcher import StructureMatcher


def match(first,second):
    matcher=StructureMatcher(ltol=.02,stol=.05,angle_tol=1,primitive_cell=True,scale=False,attempt_supercell=True)
    rms=matcher.get_rms_dist(first,second)
    return dict(physical_strict_match=rms is not None,
        primitive_normalized_rms=float(rms[0]) if rms is not None else None,
        primitive_normalized_max=float(rms[1]) if rms is not None else None)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main(root):
    from pymatgen.core import Structure
    base=json.loads((root/'report.json').read_text());data=json.loads((root/'complete.json').read_text())
    if not base['audit']['passed'] or 'endpoint_support' not in base:raise ValueError('Requires completed geometry-aware audit.')
    from cost_plot import draw
    draw(root,base,data)
    points={(r['trajectory'],r['arm']):r['result'] for r in data['relaxations']}
    summaries={}
    for arm in ('mean800','sample800'):
        cases=[]
        for row in base['pairs'][arm]['cases']:
            item=dict(row)
            if row['status']=='compared':
                i=row['trajectory'];a,b=points[i,arm],points[i,'mean1000']
                first,second=[Structure(p['cell'],p['numbers'],p['frac']) for p in (b,a)]
                item.update(match(first,second))
                item['physical_match_and_1meV']=item['physical_strict_match'] and abs(item['delta_energy_eV_atom'])<=.001
            cases.append(item)
        valid=[c for c in cases if c['status']=='compared']
        summaries[arm]=dict(total_parents=24,valid_pairs=len(valid),physical_strict_match=sum(c['physical_strict_match'] for c in valid),
            physical_match_and_1meV=sum(c['physical_match_and_1meV'] for c in valid),
            energy_within_1meV=sum(abs(c['delta_energy_eV_atom'])<=.001 for c in valid),
            simulation_cell_strict_match=sum(c['strict_match'] for c in valid),cases=cases)
    result=dict(passed=True,profile=dict(ltol=.02,stol=.05,angle_tol=1,primitive_cell=True,scale=False,attempt_supercell=True),
        source_sha256={n:sha(root/n) for n in ('report.json','complete.json')},
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ('physical.py','cost_plot.py')},pairs=summaries)
    (root/'physical-matches.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Physical crystal comparison after paired relaxation','',
        'This adds a strict physical-crystal comparison to the existing force/geometry/cost audit. '
        'The original fixed-simulation-cell matcher is retained separately: different supercell representations can describe the same infinite crystal. '
        'Here primitive reduction is enabled, volume rescaling disabled, and ltol0.02/stol0.05/angle1deg are unchanged. '
        'Normalized RMS is dimensionless; it is not an Angstrom displacement. Invalid endpoint geometries remain excluded from matched-pair energies and remain in all24-parent denominators.','',
        '| Early branch vs mean1000 | Valid pairs /24 | Strict physical crystal matches | Physical match + <=1meV/atom | Energy within1meV/atom | Strict simulation-cell matches |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,p in summaries.items():
        lines.append(f'| {arm} | {p["valid_pairs"]}/24 | {p["physical_strict_match"]} | {p["physical_match_and_1meV"]} | {p["energy_within_1meV"]} | {p["simulation_cell_strict_match"]} |')
    lines+=['','| Parent | System | Early branch | Status | Delta E (meV/atom) | Strict physical match | Normalized primitive RMS |',
        '|---|---|---|---|---:|---|---:|']
    for arm,p in summaries.items():
        for c in p['cases']:
            lines.append(f'| {c["trajectory"]} | {c["system"]} | {arm} | {c["status"]} | {1000*c["delta_energy_eV_atom"] if "delta_energy_eV_atom" in c else None} | {c.get("physical_strict_match")} | {c.get("primitive_normalized_rms")} |')
    lines+=['','See [full force, geometry and cost report](REPORT.md) for optimizer settings, cumulative timing, failures and source hashes.','']
    (root/'PHYSICAL_COMPARISON.md').write_text('\n'.join(lines))
    excluded={'evidence.tar.gz','evidence.tar.gz.sha256','verified-evidence.tar.gz','verified-evidence.tar.gz.sha256'}
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in excluded]
    with tarfile.open(root/'verified-evidence.tar.gz','w:gz') as archive:
        for path in files:archive.add(path,arcname=str(path.relative_to(root)))
    (root/'verified-evidence.tar.gz.sha256').write_text(sha(root/'verified-evidence.tar.gz')+'  verified-evidence.tar.gz\n')
    print(json.dumps(dict(pairs={a:{k:v for k,v in p.items() if k!='cases'} for a,p in summaries.items()},archive_sha256=sha(root/'verified-evidence.tar.gz'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(args.run.resolve())
