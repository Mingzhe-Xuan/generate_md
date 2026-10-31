"""Same-forward pre-fc features of the verified frozen ALIGNN inference pipeline."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from core import sha, save

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'forward_inverse'))
from property_model import Properties


def state_hash(model):
    import hashlib
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def batches(rows, size=8, atoms=512):
    selected = []; count = 0
    for row in rows:
        if selected and (len(selected) >= size or count+len(row['numbers']) > atoms):
            yield selected; selected = []; count = 0
        selected.append(row); count += len(row['numbers'])
    if selected: yield selected


def main(root):
    import torch
    torch.set_num_threads(2)
    if (root/'features.npz').exists(): raise ValueError('Refuse existing feature output.')
    prep = json.loads((root/'prepare.json').read_text())
    if sha(root/'samples.jsonl') != prep['samples_sha256']: raise ValueError('Dataset changed.')
    rows = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    loader = root/'feature-loader'; loader.mkdir()
    properties = Properties(loader); predictor = properties.predictor; model = predictor.models[0]
    if model.fc.in_features != 256 or model.fc.out_features != 1: raise ValueError('Unexpected final feature shape.')
    before = state_hash(model); captured = []; features = []; gaps = []; raw_gaps = []; eforms = []
    def hook(layer, inputs):
        if inputs[0].ndim != 2 or inputs[0].shape[1] != 256: raise ValueError('Wrong fc input.')
        captured.append(inputs[0].detach())
    handle = model.fc.register_forward_pre_hook(hook)
    control_ids = {rows[0]['id'], rows[5000]['id'], rows[-1]['id']}
    started = time.perf_counter(); max_reconstruct = 0.; processed = 0
    try:
        for index, selected in enumerate(batches(rows)):
            sizes = torch.tensor([len(r['numbers']) for r in selected], device='cuda')
            cells = torch.tensor([r['cell'] for r in selected], device='cuda', dtype=torch.float64)
            frac = torch.tensor(np.concatenate([r['frac'] for r in selected]), device='cuda', dtype=torch.float64)
            numbers = torch.tensor(np.concatenate([r['numbers'] for r in selected]), device='cuda')
            onehot = torch.nn.functional.one_hot(numbers-1, 98).double()
            captured.clear()
            with torch.no_grad():
                graphs = predictor.graphs(cells, frac, onehot, sizes)
                outputs = predictor.predict(cells, frac, onehot, sizes, graphs=graphs,
                    native_control=bool(control_ids & {r['id'] for r in selected}))
            if len(captured) != 1 or len(captured[0]) != len(selected): raise ValueError('Hook call/row coverage changed.')
            h = captured[0]
            with torch.no_grad(): raw = (h@model.fc.weight.T+model.fc.bias).reshape(-1)
            error = float((raw.clamp_min(0)-outputs[0]).abs().max()); max_reconstruct = max(max_reconstruct, error)
            if error > 1e-9: raise ValueError('Same-forward reconstruction failed.')
            features.append(h.cpu().numpy()); raw_gaps.extend(raw.cpu().tolist())
            gaps.extend(outputs[0].cpu().tolist()); eforms.extend(outputs[1].cpu().tolist())
            processed += len(selected)
            if index % 20 == 0:
                save(root/'feature-progress.json', dict(rows=processed, total=len(rows), batches=index+1,
                    seconds=time.perf_counter()-started, max_reconstruction_error=max_reconstruct))
                print((root/'feature-progress.json').read_text(), flush=True)
    finally: handle.remove()
    after = state_hash(model)
    if before != after or any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError('Frozen predictor changed.')
    x = np.concatenate(features)
    if x.shape != (9239, 256) or not np.isfinite(x).all(): raise ValueError('Feature coverage/finite failure.')
    np.savez_compressed(root/'features.npz', h=x, prediction=np.array(gaps), raw_prediction=np.array(raw_gaps),
        eform_prediction=np.array(eforms), truth=np.array([r['truth'] for r in rows]),
        ids=np.array([r['id'] for r in rows]), role=np.array([r['role'] for r in rows]),
        group=np.array([r['group'] for r in rows]), original=np.array([r['original'] for r in rows]))
    test = np.array([r['original'] == 'test' for r in rows]); archived=np.array([prep['published_test_predictions'][r['id']] for r in rows if r['original']=='test'])
    delta = np.array(raw_gaps)[test]-archived
    properties.metadata['calls'] = predictor.forward_calls
    save(root/'feature-complete.json', dict(passed=True, rows=len(rows), dimension=256,
        samples_sha256=prep['samples_sha256'], features_sha256=sha(root/'features.npz'),
        weights_before=before, weights_after=after, frozen=True, alignn=properties.metadata,
        native_controls=predictor.native_controls, max_same_forward_reconstruction_error=max_reconstruct,
        current_test_mae=float(np.mean(np.abs(np.array(gaps)[test]-np.array([r['truth'] for r in rows])[test]))),
        archived_prediction_differences=dict(mae=float(np.abs(delta).mean()), max=float(np.abs(delta).max()),
            fraction_over_0_01=float(np.mean(np.abs(delta)>.01))),
        seconds=time.perf_counter()-started, device=torch.cuda.get_device_name(),
        torch=torch.__version__, batch_size_cap=8, batch_atom_cap=512,
        graph_builds=predictor.graph_builds,
        code_sha256={p.name: sha(p) for p in HERE.glob('*.py')}))
    print((root/'feature-complete.json').read_text(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True); a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Use Slurm.')
    main(a.run.absolute())
