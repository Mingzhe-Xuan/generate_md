"""Same-input hook versus unobserved and independent-wrapper native replay."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from csp_model import CSP,sha
from relations import relative_error
from extract import embedding


def wrapper_embedding(model,numbers,cell,cart,t):
    gemnet=model.native.model.gemnet;original=gemnet.forward;captured=[]
    def observed(*args,**kw):
        out=original(*args,**kw);captured.append(out.node_embeddings);return out
    gemnet.forward=observed
    try:field=model.score(numbers,cell,cart,t,dtype='single')
    finally:gemnet.forward=original
    if len(captured)!=1:raise ValueError('Unexpected wrapper call count.')
    return captured[0].detach().cpu().double().numpy().mean(0),field


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    args.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads((args.source/'selection.json').read_text());case=selection['cases'][0]
    model=CSP();numbers=np.array(case['numbers']);cell=np.array(case['cell']);frac=np.array(case['frac']);cart=frac@cell
    rng=np.random.default_rng(20261059);rows=[];raw={}
    for i,t in enumerate(selection['policy']['times']):
        _,base=embedding(model,numbers,cell,cart,t)
        noise=rng.normal(size=frac.shape);noise-=noise.mean(0)
        for domain,coordinates in [('clean',cart),('noisy',(frac+base['std']*noise)@cell)]:
            modes={}
            for mode in ('plain','hook','wrapper'):
                scores=[];features=[];graphs=[]
                for _ in range(3):
                    if mode=='plain':field=model.score(numbers,cell,coordinates,t,dtype='single')
                    else:
                        feature,field=(embedding if mode=='hook' else wrapper_embedding)(model,numbers,cell,coordinates,t)
                        features.append(feature)
                    scores.append(field['score']);graphs.append(field['graph_hash'])
                prefix=f'{i}_{domain}_{mode}'
                raw[prefix+'_scores']=scores
                if features:raw[prefix+'_features']=features
                modes[mode]=dict(graphs=graphs,score_norms=[float(np.linalg.norm(s)) for s in scores],
                    score_replay_relative=[relative_error(scores[0],s) for s in scores[1:]],
                    score_replay_absolute=[float(np.linalg.norm(scores[0]-s)) for s in scores[1:]],
                    feature_replay_relative=[relative_error(features[0],f) for f in features[1:]] if features else None)
            plain=raw[f'{i}_{domain}_plain_scores'][0];hook=raw[f'{i}_{domain}_hook_scores'][0];wrapper=raw[f'{i}_{domain}_wrapper_scores'][0]
            rows.append(dict(t=t,domain=domain,modes=modes,hook_plain_relative=relative_error(hook,plain),
                hook_plain_absolute=float(np.linalg.norm(hook-plain)),wrapper_plain_relative=relative_error(wrapper,plain),
                hook_wrapper_feature_relative=relative_error(raw[f'{i}_{domain}_hook_features'][0],raw[f'{i}_{domain}_wrapper_features'][0])))
    np.savez_compressed(args.output/'vectors.npz',numbers=numbers,cell=cell,cart=cart,**raw)
    report=dict(selection_sha256=sha(args.source/'selection.json'),first_parent=case,script_sha256=sha(Path(__file__)),
        rows=rows,model=model.metadata,vectors_sha256=sha(args.output/'vectors.npz'))
    (args.output/'complete.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
