"""Counted isolated worker; all child compute remains within its Slurm GPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys


class Physics:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'smoacs/alternating'))
        from rpc import read_response
        self.read = read_response
        self.log = (root/'worker.err').open('w')
        # Resolving the venv executable symlink selects the system interpreter.
        self.process = subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),
            str(Path(__file__).with_name('physics_worker.py')), '--checkpoint',
            str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        self.metadata = self.read(self.process.stdout, root/'worker.stdout')
        if not self.metadata.get('ready'):
            raise RuntimeError('Physics worker not ready.')
        self.NFE = 0
        self.requests = 0

    def request(self, tasks):
        self.process.stdin.write(json.dumps(dict(tasks=tasks), allow_nan=False)+'\n')
        self.process.stdin.flush()
        response = self.read(self.process.stdout, self.root/'worker.stdout')
        rows = response['results']
        if [r['id'] for r in rows] != [t['id'] for t in tasks]:
            raise ValueError('Physics response coverage changed.')
        self.NFE += sum(r['NFE'] for r in rows)
        self.requests += len(tasks)
        with (self.root/'ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(tasks=tasks, results=rows), allow_nan=False)+'\n')
        if any(r.get('infrastructure_failure') for r in rows):
            raise RuntimeError('Physics infrastructure failure; inspect ledger.')
        return rows

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop":true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=20)
        self.log.close()
