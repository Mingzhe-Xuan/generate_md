"""Validate an existing full reference pool before canonical serial continuation."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from fetch import digest


def validate(run,selection,checkpoint_sha):
    run=Path(run).resolve();selection=Path(selection).resolve()
    if not selection.is_relative_to(run/'references'):raise ValueError('Reference selection outside this task.')
    summary=json.loads((selection/'summary.json').read_text())
    if summary['inventory_sha256']!=digest(run/'reference-scope.json'):raise ValueError('Reference scope hash mismatch.')
    if summary['snapshot']!='2026-04-13' or 'published Zenodo20149017' not in summary['reference_filter']:
        raise ValueError('Reference selection protocol differs.')
    manifest=json.loads((selection/'mattersim-inventory.json').read_text());candidates=manifest['candidates']
    ids=[c['candidate_id'] for c in candidates]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate reference IDs.')
    reused=[]
    for candidate in candidates:
        if digest(candidate['source'])!=candidate['sha256']:raise ValueError('Reference input changed.')
        folder=selection/'mattersim-relaxation'/Path(candidate['candidate_id']).stem;path=folder/'result.json'
        if not path.exists():continue
        record=json.loads(path.read_text())
        if record['input_sha256']!=candidate['sha256'] or record['checkpoint_sha256']!=checkpoint_sha:
            raise ValueError('Cached reference provenance differs.')
        if record['status'] not in ('converged','unconverged','failed'):raise ValueError('Incomplete cached reference record.')
        if record['status']=='converged':
            if not record['converged'] or not 0<=record['max_filter_force']<=.01 or digest(folder/'relaxed.vasp')!=record['relaxed_sha256']:
                raise ValueError('Invalid cached converged reference.')
        reused.append(record['status'])
    return dict(mode='canonical_serial',references=len(candidates),cached=len(reused),missing=len(candidates)-len(reused),
        cached_status_counts=dict(Counter(reused)),manifest_sha256=digest(selection/'mattersim-inventory.json'),
        checkpoint_sha256=checkpoint_sha,parallel_gate='Failed16-control convergence/trajectory equivalence; parallel production is not used.',
        profile='Unchanged reference BFGS/ExpCellFilter/FixSymmetry, fmax0.01,400steps, original default maxstep.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--selection',required=True,type=Path)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    source=Path(__file__).resolve().parents[1]/'smoacs/postprocess/sources.json'
    report=validate(args.run,args.selection,json.loads(source.read_text())['mattersim']['checkpoint_sha256'])
    report['job_id']=os.environ['SLURM_JOB_ID']
    (args.selection/'reference-execution.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.selection/('reference-execution-'+report['job_id']+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
