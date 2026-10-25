"""Replay only B for four raw and corresponding full-CF200 saved structures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if not root.is_relative_to((Path.cwd()/'results/mattergen').resolve()):raise ValueError('Use current task results.')
    parent=Path('results/mattergen/gap2p5-task128-20261024/full128').resolve()
    cf=Path('results/mattergen/alignn-full-361/cf200/neutral').resolve()
    ids=[f'mattergen-{k}.vasp' for k in ('0000','0001','0008','0087')]
    raw_scores=load(parent/'cross-prediction/predictions.json')
    cf_scores=load(cf/'evaluation-cycle1/cross-prediction/predictions.json')
    optimized={r['candidate_id']:r for r in load(cf/'results.json')['cycles'][0]['candidates']}
    tasks=[];manifest=[]
    for arm in ('direct','CF200'):
        for key in ids:
            source=parent/'structures'/key if arm=='direct' else Path(optimized[key]['A']['source'])
            expected_folder=parent/'relaxation'/Path(key).stem if arm=='direct' else cf/'evaluation-cycle1/relaxation'/Path(key).stem
            expected=load(expected_folder/'result.json')
            scores=raw_scores if arm=='direct' else cf_scores
            initial=next(r for r in scores if r['candidate_id']==key and r['stage']=='before')
            final=next(r for r in scores if r['candidate_id']==key and r['stage']=='after')
            if sha(source)!=expected['input_sha256'] or initial['input_sha256']!=sha(source):raise ValueError('Original B input changed.')
            if expected['status']!='converged':raise ValueError('Select converged replay controls.')
            folder=root/arm/Path(key).stem
            tasks.append(dict(candidate_id=key,source=str(source),output=str(folder),save_trajectory=True))
            manifest.append(dict(arm=arm,candidate_id=key,source=str(source),input_sha256=sha(source),output=str(folder),
                original_result=expected,original_before=initial,original_after=final))
    (root/'manifest.json').write_text(json.dumps(dict(ids=ids,cases=manifest,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
    smoacs=Path(__file__).resolve().parents[2]/'smoacs'
    sys.path.insert(0,str(smoacs/'alternating'))
    from rpc import read_response
    with (root/'worker.err').open('w') as log:
        worker=subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(smoacs/'alternating/worker.py'),
            '--checkpoint',str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve())],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        try:
            ready=read_response(worker.stdout,root/'worker.stdout')
            if not ready['ready']:raise ValueError('Worker not ready.')
            (root/'worker-metadata.json').write_text(json.dumps(ready,indent=2)+'\n')
            worker.stdin.write(json.dumps(dict(tasks=tasks))+'\n');worker.stdin.flush()
            result=read_response(worker.stdout,root/'worker.stdout')
            if len(result['results'])!=8:raise ValueError('Missing replay attempts.')
            (root/'replay-results.json').write_text(json.dumps(result,indent=2)+'\n')
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)


if __name__=='__main__':main()
