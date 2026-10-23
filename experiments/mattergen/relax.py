"""Apply the current candidate B profile through persistent MatterSim workers."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    if not 1<=args.workers<=int(os.environ['SLURM_CPUS_PER_TASK']):p.error('Worker count exceeds allocation.')
    root=Path(__file__).resolve().parents[1]/'smoacs';spec=importlib.util.spec_from_file_location('mattergen_worker_rpc',root/'alternating/run.py')
    import sys
    sys.path.insert(0,str(root/'alternating'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    inventory=json.loads((args.run/'inventory-249/inventory.json').read_text());candidates=inventory['candidates']
    if len(candidates)!=inventory['denominator']:raise ValueError('B cohort missing inputs.')
    tasks=[dict(candidate_id=c['candidate_id'],source=c['source'],output=str(args.run/'relaxation'/Path(c['candidate_id']).stem)) for c in candidates]
    workers=[];logs=[];records=[]
    try:
        for i in range(args.workers):
            log=(args.run/('worker-'+str(i)+'.err')).open('w');logs.append(log)
            worker=subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(root/'alternating/worker.py'),
                '--checkpoint',str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').absolute())],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            workers.append(worker)
            if not module.read_response(worker.stdout,args.run/('worker-'+str(i)+'.stdout'))['ready']:raise RuntimeError('MatterSim worker not ready.')
        for i,worker in enumerate(workers):worker.stdin.write(json.dumps(dict(tasks=tasks[i::len(workers)]))+'\n');worker.stdin.flush()
        for i,worker in enumerate(workers):records.extend(module.read_response(worker.stdout,args.run/('worker-'+str(i)+'.stdout'))['results'])
        if len(records)!=len(tasks) or {r['candidate_id'] for r in records}!={t['candidate_id'] for t in tasks}:raise ValueError('B IDs differ.')
        scope=[dict(candidate_id=c['candidate_id'],elements=c['elements']) for c in candidates
               if next(r for r in records if r['candidate_id']==c['candidate_id'])['status']=='converged']
        (args.run/'reference-scope.json').write_text(json.dumps(dict(denominator=len(scope),candidates=scope))+'\n')
    finally:
        for worker in workers:
            if worker.poll() is None:
                try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=20)
                except Exception:worker.terminate();worker.wait(timeout=10)
        for log in logs:log.close()


if __name__=='__main__':main()
