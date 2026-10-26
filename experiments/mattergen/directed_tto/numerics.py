"""Diagnose native CF arithmetic and local directional derivative convergence."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from geometry import Geometry
from directions import normalize
from settings import SEED
from cf_gradient import GradientScorer


def promote(scorer,dtype):
    for model in scorer.models:
        model.to(dtype=dtype)
        if hasattr(model,'atom_feat'):model.atom_feat=model.atom_feat.to(dtype=dtype)


def score(scorer,geom,q,dtype,backward=False):
    torch=scorer.torch;device=next(scorer.models[0].parameters()).device
    with scorer.operators(backward),torch.enable_grad() if backward else torch.no_grad():
        variable=torch.tensor(np.asarray(q),dtype=dtype,device=device,requires_grad=backward)
        cell,cart=geom.torch_pose(variable)
        atomic=torch.nn.functional.one_hot(torch.tensor(geom.numbers,dtype=torch.long,device=device)-1,num_classes=98).to(dtype)
        gap,ef=scorer.author.Crystalformer_prediction(*scorer.models,atomic,cart,torch.zeros(geom.n,dtype=torch.long,device=device),cell[None],torch.tensor([geom.n],device=device))
        loss=.5*(gap.reshape(-1)[0]-2.5)**2
        gradient=torch.autograd.grad(loss,variable)[0].detach().cpu().double().numpy() if backward else None
    if hasattr(scorer,'metadata'):
        scorer.metadata['CF_calls']+=1
        if backward:scorer.metadata['CF_backward_calls']+=1
    return dict(gap=float(gap.detach().reshape(-1)[0]),eform=float(ef.detach().reshape(-1)[0]),loss=float(loss.detach()),gradient=gradient)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('GPU diagnostic must use Slurm.')
    from probe import common_seed,load,save,sha,close
    from pymatgen.core import Structure
    plan=load(a.run/'plan.json');output=a.run/('numerics-'+os.environ['SLURM_JOB_ID']+'.json')
    if output.exists():raise ValueError('Refuse to overwrite numerical evidence.')
    scorer=GradientScorer();records={};gradients={};epsilon=(.004,.002,.001,.0005,.00025,.000125,1e-5,1e-6)
    for precision,dtype in [('float32',scorer.torch.float32),('float64',scorer.torch.float64)]:
        promote(scorer,dtype)
        for case in plan['cases'][:4]:
            parent=case['parent_id'];index=int(Path(parent).stem.split('-')[1]);s=Structure.from_file(case['B']['relaxed_path'])
            geom=Geometry(s.atomic_numbers,s.lattice.matrix,s.frac_coords);q,_=common_seed(geom,index,plan['search_seed'])
            ad=score(scorer,geom,q,dtype,True);native=score(scorer,geom,q,dtype);g=geom.project(ad.pop('gradient'))
            gradients[precision,parent]=g
            rng=np.random.default_rng([SEED,index,777]);rows=[]
            for block in ('atomic','strain','both'):
                v=geom.project(rng.normal(size=geom.size))
                if block=='atomic':v[-6:]=0
                if block=='strain':v[:-6]=0
                v=normalize(v);pred=float(g@v)
                for eps in epsilon:
                    plus=score(scorer,geom,q+eps*v,dtype)['loss'];minus=score(scorer,geom,q-eps*v,dtype)['loss']
                    fd=(plus-minus)/(2*eps)
                    rows.append(dict(block=block,epsilon=eps,FD=fd,predicted=pred,passed=close(fd,pred,.001,.02),
                                     right_derivative=(plus-native['loss'])/eps,left_derivative=(native['loss']-minus)/eps))
            entry=dict(parent_id=parent,source_sha256=sha(case['B']['relaxed_path']),precision=precision,score=ad,
                       adapter_forward_error=max(abs(ad[k]-native[k]) for k in ('gap','eform')),checks=rows)
            if precision=='float64':
                first=gradients['float32',parent]
                entry['float32_vs_64']=dict(gradient_cosine=float(first@g/(np.linalg.norm(first)*np.linalg.norm(g))),
                    gradient_relative_error=float(np.linalg.norm(first-g)/np.linalg.norm(g)),
                    gap_error=abs(records['float32',parent]['score']['gap']-ad['gap']))
            records[precision,parent]=entry
            save(output,dict(CF=scorer.metadata,cases=list(records.values()),role='Numerical diagnostic only; not a passed scientific native gate.'))
            print(json.dumps({k:v for k,v in entry.items() if k!='checks'}),flush=True)


if __name__=='__main__':main()
