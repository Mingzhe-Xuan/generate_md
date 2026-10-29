"""Inspect a failed native/FP64 comparison without changing its acceptance gate."""
import argparse
import os
from pathlib import Path
import numpy as np
from ase.io import read
from csp_model import CSP,sha
from probe import save
from relations import centered_basis,relative_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    args.output.mkdir(parents=True,exist_ok=False)
    atoms=read(args.source,format='vasp');numbers=atoms.numbers;cell=atoms.cell.array;cart=atoms.positions
    model=CSP();q=centered_basis(len(numbers));rng=np.random.default_rng(20261052)
    direction=q@rng.normal(size=q.shape[1]);direction/=np.linalg.norm(direction)
    rows=[];vectors={}
    for t in (.001,.05,.15,.30):
        values={}
        for dtype in ('single','double'):
            center=model.score(numbers,cell,cart,t,dtype=dtype)
            plus=model.score(numbers,cell,cart+.001*direction.reshape(-1,3),t,dtype=dtype)
            minus=model.score(numbers,cell,cart-.001*direction.reshape(-1,3),t,dtype=dtype)
            values[dtype]=center
            vectors[dtype+'_'+str(t)]=np.stack([r['score'] for r in (center,plus,minus)])
        first,second=(values[k] for k in ('single','double'))
        rows.append(dict(t=t,relative=relative_error(first['score'],second['score']),
            fp32_norm=float(np.linalg.norm(first['score'])),fp64_norm=float(np.linalg.norm(second['score'])),
            max_abs_error=float(np.max(np.abs(first['score']-second['score']))),
            same_graph=first['graph_hash']==second['graph_hash'],
            fp32_graph=first['graph_hash'],fp64_graph=second['graph_hash'],
            std32=first['std'],std64=second['std']))
        print(rows[-1],flush=True)
    np.savez_compressed(args.output/'vectors.npz',numbers=numbers,cell=cell,cart=cart,direction=direction,**vectors)
    save(args.output/'complete.json',dict(source=str(args.source),source_sha256=sha(args.source),
        job_id=os.environ['SLURM_JOB_ID'],results=rows,model=model.metadata,
        note='Diagnostic only: the native-field relative acceptance threshold has not changed.'))


if __name__=='__main__':main()
