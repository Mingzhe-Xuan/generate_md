"""Two counted Slurm GPU physics workers shared by derivatives and full quenches."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'smoacs/alternating'))
from rpc import read_response


class Mechanics:
    def __init__(self,root):
        self.root=root;self.processes=[];self.logs=[];self.ready=[];self.serial=0;self.NFE=0;self.seconds=0.
        for i in range(2):
            log=(root/f'physics-{i}.err').open('a');self.logs.append(log)
            process=subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(Path(__file__).with_name('mechanics.py')),
                '--checkpoint',str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve())],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            self.processes.append(process);meta=read_response(process.stdout,root/f'physics-{i}.stdout')
            if not meta.get('ready'):raise RuntimeError('Physics worker not ready.')
            self.ready.append(meta)
        (root/'mechanics-metadata.json').write_text(json.dumps(self.ready,indent=2)+'\n')

    def request(self,tasks):
        def call(i):
            subset=tasks[i::2]
            if not subset:return []
            p=self.processes[i];p.stdin.write(json.dumps(dict(tasks=subset),allow_nan=False)+'\n');p.stdin.flush()
            return read_response(p.stdout,self.root/f'physics-{i}.stdout')['results']
        with ThreadPoolExecutor(max_workers=2) as pool:groups=list(pool.map(call,range(2)))
        byid={r['candidate_id']:r for group in groups for r in group}
        if len(byid)!=len(tasks) or set(byid)!={r['id'] for r in tasks}:raise ValueError('Physics response coverage changed.')
        result=[byid[t['id']] for t in tasks]
        for row in result:
            self.NFE+=row['NFE'];self.seconds+=row['elapsed_seconds']
            if row.get('infrastructure_failure'):raise RuntimeError('Physics infrastructure failure: '+row['error'])
        return result

    def points(self,geom,Q):
        tasks=[]
        for q in Q:
            cell,frac=geom.pose(q);self.serial+=1
            tasks.append(dict(id=f'point-{self.serial}',action='point',numbers=geom.numbers.tolist(),cell=cell.tolist(),frac=frac.tolist()))
        rows=self.request(tasks)
        if any(r['status']!='completed' for r in rows):raise ValueError('Physical derivative point failed: '+str(rows))
        return rows

    def gradients(self,geom,Q):
        return np.array([geom.energy_gradient(q,r['forces_eV_A'],r['stress_eV_A3']) for q,r in zip(Q,self.points(geom,Q))])

    def close(self):
        for p in self.processes:
            if p.poll() is None:
                try:p.stdin.write('{"stop":true}\n');p.stdin.flush();p.wait(timeout=30)
                except (BrokenPipeError,subprocess.TimeoutExpired):p.terminate();p.wait(timeout=20)
        for log in self.logs:log.close()
