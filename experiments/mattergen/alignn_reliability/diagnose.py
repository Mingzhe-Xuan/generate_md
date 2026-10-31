"""Posthoc native/single-forward checks on published-output discrepancies."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from core import sha, save


def discrepancy_ids(ids, prediction, archived, threshold=.01):
    if len(ids) != len(prediction): raise ValueError('ID/prediction mismatch.')
    return [str(k) for k,p in zip(ids,prediction) if abs(float(p)-archived[str(k)])>threshold]


def main(root):
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'forward_inverse'))
    from property_model import Properties
    prep=json.loads((root/'prepare.json').read_text())
    with np.load(root/'features.npz',allow_pickle=False) as d:data={k:d[k] for k in d.files}
    test=data['role']=='test'; wanted=discrepancy_ids(data['ids'][test],data['raw_prediction'][test],prep['published_test_predictions'])
    samples={r['id']:r for r in (json.loads(l) for l in (root/'samples.jsonl').read_text().splitlines())}
    loader=root/'diagnostic-loader';loader.mkdir();properties=Properties(loader)
    rows=[]
    for key in wanted:
        sample=samples[key];index=int(np.flatnonzero(data['ids']==key)[0]);cell=np.array(sample['cell']);frac=np.array(sample['frac'])
        values,_=properties.evaluate(np.array(sample['numbers']),cell,frac@cell,native=True)
        control=properties.predictor.native_controls[-1]
        single=float(values['values'][0]);batch=float(data['prediction'][index])
        if abs(single-batch)>1e-5:raise ValueError('Single/batched inference disagreement: '+key)
        rows.append(dict(id=key,current_single_gap=single,current_batch_gap=batch,
            archived_raw_gap=prep['published_test_predictions'][key],DFT_gap=float(data['truth'][index]),
            batch_single_error=abs(single-batch),native_controls=control))
    save(root/'discrepancy-diagnostic.json',dict(passed=True,selection='Posthoc abs(raw current - archived)>0.01eV; no classifier retuning',
        samples=len(rows),rows=rows,features_sha256=sha(root/'features.npz'),
        code_sha256=sha(Path(__file__))))
    print(json.dumps(dict(samples=len(rows),max_batch_single=max((r['batch_single_error'] for r in rows),default=0),
        max_native_original_graph_error=max((r['native_controls']['original_graph_prediction_max_errors'][0] for r in rows),default=0))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(a.run.absolute())
