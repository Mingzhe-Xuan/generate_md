"""Frozen native MatterGen embeddings and common-noise ALIGNN control values."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from csp_model import CSP,sha
from property_model import Properties
from relations import relative_error


def embedding(model,numbers,cell,cart,t):
    captured=[]
    handle=model.native.model.gemnet.register_forward_hook(lambda m,args,out:captured.append(out.node_embeddings))
    try:field=model.score(numbers,cell,cart,t,dtype='single')
    finally:handle.remove()
    if len(captured)!=1 or captured[0].ndim!=2 or len(captured[0])!=len(numbers) or not captured[0].is_cuda:
        raise ValueError('Unexpected native node-embedding interface.')
    nodes=captured[0].detach().cpu().double().numpy()
    if not np.isfinite(nodes).all():raise ValueError('Nonfinite embedding.')
    return nodes.mean(0),field


def composition_cell(numbers,cell):
    fractions=np.bincount(numbers,minlength=99)[1:99]/len(numbers)
    return np.r_[fractions,np.log(len(numbers)),np.log(np.linalg.det(cell)/len(numbers))]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    root=args.run.resolve()
    if (root/'feature-plan.json').exists():raise ValueError('Preserve previous feature attempt.')
    selection=json.loads((root/'selection.json').read_text());policy=selection['policy']
    model=CSP();properties=Properties(root);started=time.perf_counter()
    plan=dict(selection_sha256=sha(root/'selection.json'),policy=policy,backbone=model.metadata,
        properties=properties.metadata,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha(Path(__file__)),labels_enter_backbone=False)
    (root/'feature-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    clean_features=[];noisy_features=[];baseline=[];clean_alignn=[];noisy_alignn=[];manifests=[]
    for index,case in enumerate(selection['cases']):
        numbers=np.array(case['numbers']);cell=np.array(case['cell']);frac=np.array(case['frac']);cart=frac@cell
        seed=int.from_bytes(hashlib.sha256(f"20261056|{case['split']}|{case['id']}|noise".encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed)
        b=composition_cell(numbers,cell);cf=[];nf=[];pa=[];ca=[];noise=[];coordinates=[];standards=[];graphs=[]
        clean_property,_=properties.evaluate(numbers,cell,cart)
        for t in policy['times']:
            clean,field=embedding(model,numbers,cell,cart,t)
            if index==0 and t==policy['times'][0]:
                repeat=model.score(numbers,cell,cart,t,dtype='single')
                if relative_error(field['score'],repeat['score'])>1e-5:raise ValueError('Feature hook altered native score/replay.')
            gaussian=rng.normal(size=frac.shape);gaussian-=gaussian.mean(0)
            displacement=field['std']*gaussian
            noisy_cart=(frac+displacement)@cell
            noisy,noisy_field=embedding(model,numbers,cell,noisy_cart,t)
            prediction,_=properties.evaluate(numbers,cell,noisy_cart)
            cf.append(np.r_[b,clean]);nf.append(np.r_[b,noisy]);ca.append(clean_property['values']);pa.append(prediction['values'])
            noise.append(displacement);coordinates.append(noisy_cart);standards.append(field['std']);graphs.append([field['graph_hash'],noisy_field['graph_hash']])
        folder=root/'parents'/case['split'];folder.mkdir(parents=True,exist_ok=True)
        path=folder/(case['id']+'.npz')
        np.savez_compressed(path,numbers=numbers,cell=cell,frac=frac,noisy_cart=coordinates,fractional_noise=noise,std=standards,
            clean_features=cf,noisy_features=nf,composition_cell=b,clean_alignn=ca,noisy_alignn=pa)
        manifests.append(dict(id=case['id'],split=case['split'],vectors=str(path.relative_to(root)),sha256=sha(path),graphs=graphs,noise_seed=seed))
        clean_features.append(cf);noisy_features.append(nf);baseline.append(np.repeat(b[None],len(policy['times']),axis=0))
        clean_alignn.append(ca);noisy_alignn.append(pa)
        if (index+1)%32==0:
            (root/'feature-progress.json').write_text(json.dumps(dict(completed=index+1,total=len(selection['cases']),seconds=time.perf_counter()-started))+'\n')
            print('FEATURES',index+1,len(selection['cases']),flush=True)
    path=root/'features.npz'
    np.savez_compressed(path,clean_features=clean_features,noisy_features=noisy_features,baseline=baseline,
        clean_alignn=clean_alignn,noisy_alignn=noisy_alignn,labels=[c['labels'] for c in selection['cases']],
        ids=[c['id'] for c in selection['cases']],splits=[c['split'] for c in selection['cases']])
    result=dict(execution_complete=True,parents=len(manifests),features_sha256=sha(path),parent_vectors=manifests,
        model=model.metadata,properties=properties.metadata,seconds=time.perf_counter()-started,
        source_commit=plan['commit'],feature_plan_sha256=sha(root/'feature-plan.json'))
    (root/'feature-complete.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
