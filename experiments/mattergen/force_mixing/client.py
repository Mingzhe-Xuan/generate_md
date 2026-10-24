"""Paired stateless force queries inside the parent Slurm GPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

from protocol import pose_hash

ROOT=Path(__file__).resolve().parents[2]/'smoacs'
sys.path.insert(0,str(ROOT/'alternating'))
from rpc import read_response


class ForceClient:
    def __init__(self,output,workers=2):
        self.output=output
        self.workers=[]
        self.logs=[]
        self.ready=[]
        self.closed=False
        try:
            for i in range(workers):
                log=(output/f'force-worker-{i}.err').open('w')
                self.logs.append(log)
                worker=subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(Path(__file__).with_name('worker.py')),
                    '--checkpoint',str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').absolute())],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                self.workers.append(worker)
                self.ready.append(read_response(worker.stdout,output/f'force-worker-{i}.stdout'))
                if not self.ready[-1].get('ready'):
                    raise RuntimeError('Force worker initialization failed.')
            if len({r['checkpoint_sha256'] for r in self.ready})!=1:
                raise ValueError('Physical models differ.')
            (output/'force-worker-metadata.json').write_text(json.dumps(self.ready,indent=2)+'\n')
        except BaseException:
            self.close()
            raise

    def evaluate(self,ids,cells,frac,numbers,slices,active=None):
        h,s=cells.detach().cpu().tolist(),frac.detach().cpu().tolist()
        z=numbers.detach().cpu().tolist()
        selected=list(range(len(ids))) if active is None else [i for i,v in enumerate(active.detach().cpu().tolist()) if v]
        tasks=[]
        for i in selected:
            task=dict(candidate_id=ids[i],cell=h[i],frac=s[slices[i]],numbers=z[slices[i]])
            task['pose_sha256']=pose_hash(task)
            tasks.append(task)
        for i,worker in enumerate(self.workers):
            worker.stdin.write(json.dumps(dict(tasks=tasks[i::len(self.workers)]),allow_nan=False)+'\n')
            worker.stdin.flush()
        result=[]
        for i,worker in enumerate(self.workers):
            result.extend(read_response(worker.stdout,self.output/f'force-worker-{i}.stdout')['results'])
        byid={r['candidate_id']:r for r in result}
        if len(result)!=len(tasks) or set(byid)!={t['candidate_id'] for t in tasks}:
            raise ValueError('Force response cohort changed.')
        for task in tasks:
            row=byid[task['candidate_id']]
            if row['status']!='completed':
                raise RuntimeError('Force calculation failed; retain checkpoint: '+json.dumps(row))
            if row['pose_sha256']!=task['pose_sha256'] or row['checkpoint_sha256']!=self.ready[0]['checkpoint_sha256']:
                raise ValueError('Stale force response or wrong model.')
        return byid

    def close(self):
        if self.closed:
            return
        self.closed=True
        for worker in self.workers:
            if worker.poll() is None:
                try:
                    worker.stdin.write('{"stop":true}\n')
                    worker.stdin.flush()
                    worker.wait(timeout=20)
                except (BrokenPipeError,subprocess.TimeoutExpired):
                    worker.terminate()
                    worker.wait(timeout=10)
        for log in self.logs:
            log.close()
