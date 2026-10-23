"""Rerelax original A200 outputs with the identical alternating B numerical profile."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from run import read_response


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Submit through Slurm.')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    original=Path('results/smoacs/postprocess-206/inventory-249/inventory.json')
    inventory=json.loads(original.read_text())
    (output/'inventory-249').mkdir();(output/'inventory-249/inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    shutil.copy2('results/smoacs/optimization/section21-Struct-AtomicMask-grad-206/result.csv',output/'result.csv')
    tasks=[dict(candidate_id=c['candidate_id'],source=c['source'],output=str(output/'relaxation'/Path(c['candidate_id']).stem)) for c in inventory['candidates']]
    workers=[];handles=[]
    try:
        for i in range(4):
            log=(output/('worker-'+str(i)+'.err')).open('w');handles.append(log)
            workers.append(subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),
                str(Path(__file__).with_name('worker.py').resolve()),'--checkpoint',
                str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve())],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')))
        ready=[read_response(w.stdout,output/('worker-'+str(i)+'.stdout')) for i,w in enumerate(workers)]
        if not all(r['ready'] for r in ready):raise RuntimeError('Matched B workers not ready.')
        for i,w in enumerate(workers):w.stdin.write(json.dumps(dict(tasks=tasks[i::4]))+'\n');w.stdin.flush()
        results=[]
        for i,w in enumerate(workers):results.extend(read_response(w.stdout,output/('worker-'+str(i)+'.stdout'))['results'])
        if len(results)!=len(tasks):raise ValueError('Matched B result count differs.')
        (output/'matched-profile.json').write_text(json.dumps(dict(b_maxstep=.05,b_limit=1600,fmax=.01,
            denominator=len(tasks),converged=sum(r['status']=='converged' for r in results)),indent=2)+'\n')
    finally:
        for w in workers:
            if w.poll() is None:
                try:w.stdin.write('{"stop":true}\n');w.stdin.flush();w.wait(timeout=20)
                except Exception:w.terminate();w.wait(timeout=10)
        for h in handles:h.close()


if __name__=='__main__':main()
