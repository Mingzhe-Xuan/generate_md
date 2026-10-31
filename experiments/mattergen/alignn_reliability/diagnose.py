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


def topology_hash(graph):
    import hashlib
    g,_,images=graph;src,dst=g.edges(order='eid')
    a=np.column_stack([src.numpy(),dst.numpy(),images.numpy()]).astype(np.int64)
    a=a[np.lexsort(a.T[::-1])]
    return hashlib.sha256(a.tobytes()).hexdigest()


def main(root):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'forward_inverse'))
    from property_model import Properties
    prep=json.loads((root/'prepare.json').read_text())
    with np.load(root/'features.npz',allow_pickle=False) as d:data={k:d[k] for k in d.files}
    test=data['role']=='test'; wanted=discrepancy_ids(data['ids'][test],data['raw_prediction'][test],prep['published_test_predictions'])
    samples={r['id']:r for r in (json.loads(l) for l in (root/'samples.jsonl').read_text().splitlines())}
    attempt=len(list(root.glob('diagnostic-loader*')))+1
    loader=root/f'diagnostic-loader-{attempt:02d}';loader.mkdir();properties=Properties(loader)
    rows=[]
    for key in wanted:
        sample=samples[key];index=int(np.flatnonzero(data['ids']==key)[0]);cell=np.array(sample['cell']);frac=np.array(sample['frac'])
        cells=torch.tensor(cell[None],device='cuda',dtype=torch.float64)
        fractions=torch.tensor(frac,device='cuda',dtype=torch.float64)
        numbers=torch.tensor(sample['numbers'],device='cuda')
        onehot=torch.nn.functional.one_hot(numbers-1,98).double();sizes=torch.tensor([len(numbers)],device='cuda')
        with torch.no_grad():
            exact_graphs=properties.predictor.graphs(cells,fractions,onehot,sizes)
            exact_values=properties.predictor.predict(cells,fractions,onehot,sizes,graphs=exact_graphs,native_control=True)
        exact_control=properties.predictor.native_controls[-1]
        single=float(exact_values[0][0]);batch=float(data['prediction'][index])
        if abs(single-batch)>1e-5:raise ValueError('Exact-input single/batched inference disagreement: '+key)
        values,roundtrip_graphs=properties.evaluate(np.array(sample['numbers']),cell,frac@cell,native=True)
        control=properties.predictor.native_controls[-1]
        reconstructed=torch.tensor(frac@cell,device='cuda',dtype=torch.float64)@torch.linalg.inv(cells[0])
        rows.append(dict(id=key,current_single_gap=single,current_batch_gap=batch,
            archived_raw_gap=prep['published_test_predictions'][key],DFT_gap=float(data['truth'][index]),
            batch_single_error=abs(single-batch),exact_input_native_controls=exact_control,
            cart_roundtrip_gap=float(values['values'][0]),roundtrip_prediction_delta=float(values['values'][0])-single,
            max_fractional_roundtrip_delta=float((reconstructed-fractions).abs().max()),
            exact_topology_hash=topology_hash(exact_graphs[0]),roundtrip_topology_hash=topology_hash(roundtrip_graphs[0]),
            native_controls=control))
    save(root/'discrepancy-diagnostic.json',dict(passed=True,exact_input_consistency_passed=True,
        representation_sensitivity_cases=[r['id'] for r in rows if abs(r['roundtrip_prediction_delta'])>1e-5],
        selection='Posthoc abs(raw current - archived)>0.01eV; no classifier retuning',
        samples=len(rows),rows=rows,features_sha256=sha(root/'features.npz'),
        code_sha256=sha(Path(__file__))))
    print(json.dumps(dict(samples=len(rows),max_batch_single=max((r['batch_single_error'] for r in rows),default=0),
        max_native_original_graph_error=max((r['native_controls']['original_graph_prediction_max_errors'][0] for r in rows),default=0),
        max_roundtrip_prediction_delta=max((abs(r['roundtrip_prediction_delta']) for r in rows),default=0))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(a.run.absolute())
