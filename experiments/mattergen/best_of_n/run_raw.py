"""Relax the exact paired inputs without CF, using the same persistent GPU B workers."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]/'smoacs'
sys.path.insert(0,str(ROOT/'alternating'))
from rpc import read_response


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm.')
    root=args.run.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    init=json.loads((root/'cf20/initialization.json').read_text())
    if init['branch_manifest_sha256']!=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Paired CF inputs differ.')
    ids=[r['candidate_id'] for r in manifest['candidates']]
    if [r['candidate_id'] for r in init['selected']]!=ids:
        raise ValueError('Paired initial candidate order differs.')
    out=root/'raw-evaluation'
    if (out/'inventory-249/inventory.json').exists():
        return
    out.mkdir(exist_ok=True)
    workers=[]
    logs=[]
    model='e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5'
    try:
        for index in range(2):
            log=(out/f'worker-{index}.err').open('w')
            logs.append(log)
            worker=subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(ROOT/'alternating/worker.py'),
                '--checkpoint',str(Path.cwd()/'results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth')],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            workers.append(worker)
            ready=read_response(worker.stdout,out/f'worker-{index}.stdout')
            if not ready['ready'] or ready['checkpoint_sha256']!=model:
                raise ValueError('B checkpoint differs from paired CF arm.')
        tasks=[dict(candidate_id=r['candidate_id'],source=r['source'],output=str(out/'relaxation'/Path(r['candidate_id']).stem)) for r in manifest['candidates']]
        for index,worker in enumerate(workers):
            worker.stdin.write(json.dumps(dict(tasks=tasks[index::2]))+'\n')
            worker.stdin.flush()
        returned=[]
        for index,worker in enumerate(workers):
            returned.extend(read_response(worker.stdout,out/f'worker-{index}.stdout')['results'])
        records={r['candidate_id']:r for r in returned}
        if set(records)!=set(ids):
            raise ValueError('Missing no-CF relaxation outputs.')
    finally:
        for worker in workers:
            if worker.poll() is None:
                try:
                    worker.stdin.write('{"stop":true}\n')
                    worker.stdin.flush()
                    worker.wait(timeout=20)
                except (BrokenPipeError,subprocess.TimeoutExpired):
                    worker.terminate()
                    worker.wait(timeout=10)
        for log in logs:
            log.close()
    from pymatgen.core import Structure
    candidates=[]
    csv_rows=[]
    for index,r in enumerate(manifest['candidates']):
        source=Path(r['source'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=r['sha256'] or records[r['candidate_id']]['input_sha256']!=r['sha256']:
            raise ValueError('No-CF B input differs from the exact paired perturbation.')
        s=Structure.from_file(source)
        gap=init['initial_hard_gap'][index]
        ef=init['initial_hard_eform'][index]
        candidates.append(dict(candidate_id=r['candidate_id'],source=r['source'],sha256=r['sha256'],num_atoms=len(s),
            formula=s.composition.formula,elements=sorted({site.specie.symbol for site in s}),original_gap=gap,original_eform=ef,original_valid='True'))
        csv_rows.append(dict(original_fnames=r['candidate_id'],bandgap_onehot=gap,ef_onehot=ef,
            is_neutral_cmn='True',minbond_less_than_0p5='True',valid_structure='True'))
    with (out/'result.csv').open('w',newline='') as stream:
        fields=list(csv_rows[0])
        fields[4]='minbond_less_than_0.5'
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in csv_rows:
            row['minbond_less_than_0.5']=row.pop('minbond_less_than_0p5')
            writer.writerow(row)
    (out/'inventory-249').mkdir(exist_ok=True)
    inventory=dict(denominator=len(candidates),role='Same perturbed inputs without CF',candidates=candidates,
        source_csv_sha256=hashlib.sha256((out/'result.csv').read_bytes()).hexdigest())
    (out/'inventory-249/inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(raw_B_records=len(records))),flush=True)


if __name__=='__main__':
    main()
