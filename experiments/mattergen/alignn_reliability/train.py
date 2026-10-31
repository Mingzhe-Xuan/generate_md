"""Train only tiny correctness classifiers; never change the property predictor."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from core import GAP_SHA, THRESHOLDS, sha, save, labels, standardize_fit, metrics, calibrate
from head import Head, sigmoid


def selection(y, logit):
    from sklearn.metrics import average_precision_score
    p = sigmoid(logit)
    score = average_precision_score(1-y, 1-p) if len(np.unique(y)) == 2 else -float(np.mean((p-y)**2))
    loss = float(np.mean(np.logaddexp(0, logit)-y*logit))
    return float(score), -loss


def linear_fit(x, y, vx, vy, source):
    from sklearn.linear_model import LogisticRegression
    mean, scale = standardize_fit(x); xx = (x-mean)/scale; vv = (vx-mean)/scale
    records = []; best = None
    if len(np.unique(y)) == 1:
        p = np.clip(y.mean(), 1e-6, 1-1e-6)
        return Head(dict(kind=np.array('constant'), input=np.array(source), mean=mean, scale=scale,
            bias=np.array(np.log(p/(1-p))), predictor_sha256=np.array(GAP_SHA))), []
    for c in (.01, .1, 1., 10.):
        model = LogisticRegression(C=c, max_iter=3000, solver='lbfgs', random_state=42).fit(xx, y)
        score = selection(vy, model.decision_function(vv)); records.append(dict(C=c, validation_selection=list(score), iterations=int(model.n_iter_[0])))
        if best is None or score > best[0]: best = (score, c, model)
    model = best[2]
    return Head(dict(kind=np.array('linear'), input=np.array(source), mean=mean, scale=scale,
        weight=model.coef_[0].copy(), bias=model.intercept_[0].copy(), C=np.array(best[1]),
        predictor_sha256=np.array(GAP_SHA))), records


def mlp_fit(x, y, vx, vy):
    import torch
    from torch import nn
    mean, scale = standardize_fit(x)
    xx = torch.tensor((x-mean)/scale, dtype=torch.float32); yy = torch.tensor(y, dtype=torch.float32)
    vv = torch.tensor((vx-mean)/scale, dtype=torch.float32)
    best = None; records = []
    for lr in (3e-4, 1e-3):
        torch.manual_seed(42)
        net = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 1))
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
        local = None; stale = 0
        for epoch in range(200):
            net.train()
            for ix in torch.randperm(len(xx)).split(128):
                optimizer.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(net(xx[ix]).flatten(), yy[ix])
                if not torch.isfinite(loss): raise ValueError('Nonfinite classifier loss.')
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
            net.eval()
            with torch.no_grad(): score = selection(vy, net(vv).flatten().numpy())
            if local is None or score > local[0]:
                local = (score, epoch+1, {k: v.detach().clone() for k, v in net.state_dict().items()}); stale = 0
            else: stale += 1
            if stale >= 25: break
        records.append(dict(lr=lr, epochs=epoch+1, chosen_epoch=local[1], validation_selection=list(local[0])))
        if best is None or local[0] > best[0]: best = (local[0], lr, local[1], local[2])
    state = best[3]
    return Head(dict(kind=np.array('mlp'), input=np.array('h'), mean=mean, scale=scale,
        w1=state['0.weight'].numpy(), b1=state['0.bias'].numpy(),
        w2=state['3.weight'].numpy(), b2=state['3.bias'].numpy(),
        lr=np.array(best[1]), epochs=np.array(best[2]), predictor_sha256=np.array(GAP_SHA))), records


def main(root):
    import torch
    import sklearn
    torch.set_num_threads(2); started = time.perf_counter()
    if (root/'heads-complete.json').exists(): raise ValueError('Refuse completed head training.')
    info = json.loads((root/'feature-complete.json').read_text())
    if sha(root/'features.npz') != info['features_sha256']: raise ValueError('Features changed.')
    with np.load(root/'features.npz', allow_pickle=False) as d: data = {k: d[k] for k in d.files}
    role = data['role']; ix = {r: np.flatnonzero(role == r) for r in ('meta_train','meta_val','calibration','test')}
    if len(ix['test']) != 4239 or sum(len(ix[r]) for r in ix if r != 'test') != 5000:
        raise ValueError('Meta/test partition coverage changed.')
    if any(len(ix[r]) == 0 for r in ix): raise ValueError('Empty partition.')
    for a, b in [('meta_train','meta_val'),('meta_train','calibration'),('meta_val','calibration')]:
        if set(data['group'][ix[a]]) & set(data['group'][ix[b]]): raise ValueError('Composition overlap in meta split.')
    train, valid, cal, test = (ix[k] for k in ('meta_train','meta_val','calibration','test'))
    h = data['h']; raw = data['raw_prediction']; error = np.abs(data['prediction']-data['truth'])
    out = root/'heads'; out.mkdir(); results = []; probabilities = {}; decisions = []
    meta_groups = set(data['group'][np.flatnonzero(role != 'test')])
    unseen = np.array([g not in meta_groups for g in data['group'][test]])
    for threshold in THRESHOLDS:
        y = labels(data['truth'], data['prediction'], threshold)
        candidates = {}
        baseline, records = linear_fit(raw[train, None], y[train], raw[valid, None], y[valid], 'raw_prediction')
        candidates['prediction_only'] = (baseline, records)
        linear, records = linear_fit(h[train], y[train], h[valid], y[valid], 'h'); candidates['feature_linear'] = (linear, records)
        mlp, records = mlp_fit(h[train], y[train], h[valid], y[valid]); candidates['feature_mlp'] = (mlp, records)
        choices = []
        for name, (head, records) in candidates.items():
            stem = f'{name}-tau{threshold}'
            val_logits = head.logits(h[valid], raw[valid]); quality = selection(y[valid], val_logits)
            calibration = calibrate(head.logits(h[cal], raw[cal]), y[cal])
            head.arrays.update(cal_a=np.array(calibration['a']), cal_b=np.array(calibration['b']), threshold=np.array(threshold))
            file = out/(stem+'.npz'); head.save(file); restored = Head.load(file)
            p = restored.predict_proba(h[test], raw[test])
            if not np.allclose(p, head.predict_proba(h[test], raw[test]), atol=1e-12, rtol=0): raise ValueError('Head serialization mismatch.')
            probabilities[stem] = p
            report = dict(threshold=threshold, model=name, file=str(file.relative_to(root)), sha256=sha(file),
                parameters=int(sum(v.size for k,v in head.arrays.items() if k in ('weight','bias','w1','b1','w2','b2'))),
                search=records, validation_selection=list(quality), calibration=calibration,
                validation=metrics(y[valid], restored.predict_proba(h[valid], raw[valid]), error[valid]),
                test=metrics(y[test], p, error[test]),
                test_unseen_meta_composition=metrics(y[test][unseen], p[unseen], error[test][unseen]),
                test_predicted_gap_over_0_1=metrics(y[test][data['prediction'][test]>.1], p[data['prediction'][test]>.1], error[test][data['prediction'][test]>.1]),
                test_nonmetal_truth=metrics(y[test][data['truth'][test]>0], p[data['truth'][test]>0], error[test][data['truth'][test]>0]))
            results.append(report)
            if name.startswith('feature_'): choices.append((quality, name, str(file.relative_to(root))))
        chosen = max(choices)
        decisions.append(dict(threshold=threshold, model=chosen[1], file=chosen[2], selected_using='meta_val AP_bad then negative BCE only'))
        print(json.dumps(dict(threshold=threshold, chosen=chosen[1], reports=[dict(model=r['model'], auroc=r['test']['auroc'], ap_bad=r['test']['ap_bad']) for r in results if r['threshold']==threshold])), flush=True)
    np.savez_compressed(root/'test-probabilities.npz', **probabilities, ids=data['ids'][test])
    save(root/'heads-complete.json', dict(passed=True, thresholds=THRESHOLDS, feature_source_sha256=info['features_sha256'],
        partition_counts={k: len(v) for k,v in ix.items()}, chosen=decisions, reports=results,
        unseen_test_compositions=int(unseen.sum()), original_training_ids_excluded=True,
        sklearn=sklearn.__version__, torch=torch.__version__, seconds=time.perf_counter()-started,
        warning='ID test is frozen; original validation was used for base selection. Calibration is empirical, no guarantee for noisy/OOD/RL-adapted candidates.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True); a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Use Slurm.')
    main(a.run.absolute())
