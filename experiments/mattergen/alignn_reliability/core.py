"""Shared label, split and evaluation contracts for frozen ALIGNN reliability."""
from collections import Counter
import hashlib
import json
from math import gcd
from functools import reduce
from pathlib import Path
import numpy as np

GAP_SHA = '8e9309fcb435a88481ccbc945d16cb1343f9ac24d8198607f753ce59b687242e'
THRESHOLDS = (.1, .2, .5)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def composition(elements):
    counts = Counter(elements); divisor = reduce(gcd, counts.values())
    return '|'.join(f'{e}:{counts[e]//divisor}' for e in sorted(counts))


def role(group, original):
    if original == 'test': return 'test'
    if original != 'val': raise ValueError('Original ALIGNN training IDs cannot supervise reliability.')
    value = int(hashlib.sha256(('20261031|'+group).encode()).hexdigest()[:12], 16) % 10000
    return 'meta_train' if value < 7000 else ('meta_val' if value < 8500 else 'calibration')


def labels(truth, prediction, threshold):
    a, b = np.asarray(truth), np.asarray(prediction)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid truth/prediction arrays.')
    return (np.abs(a-b) <= threshold+1e-12).astype(np.int64)


def standardize_fit(x):
    mean = np.mean(x, axis=0); scale = np.std(x, axis=0)
    return mean, np.where(scale > 1e-12, scale, 1.)


def metrics(y, probability, error):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y); p = np.asarray(probability); error = np.asarray(error)
    if len(y) == 0: return dict(n=0)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)): raise ValueError('Invalid confidence.')
    result = dict(n=len(y), correct_fraction=float(y.mean()), mae=float(error.mean()),
        brier=float(np.mean((p-y)**2)), auroc=None, ap_bad=None,
        logloss=float(-np.mean(y*np.log(np.clip(p, 1e-12, 1))+(1-y)*np.log(np.clip(1-p, 1e-12, 1)))))
    if len(np.unique(y)) == 2:
        result.update(auroc=float(roc_auc_score(y, p)), ap_bad=float(average_precision_score(1-y, 1-p)))
    for cutoff in (.5, .7, .9):
        keep = p >= cutoff
        result[f'accept_{cutoff}'] = dict(n=int(keep.sum()), coverage=float(keep.mean()),
            correct_fraction=float(y[keep].mean()) if keep.any() else None,
            mae=float(error[keep].mean()) if keep.any() else None)
    order = np.argsort(-p, kind='stable'); curve = []
    for fraction in (.1, .25, .5, .75, 1.):
        n = max(1, int(np.ceil(len(y)*fraction))); ix = order[:n]
        curve.append(dict(coverage=n/len(y), n=n, correct_fraction=float(y[ix].mean()), mae=float(error[ix].mean())))
    result['coverage_curve'] = curve
    return result


def calibrate(logit, y):
    """Positive-slope Platt calibration on a separate calibration partition."""
    from scipy.optimize import minimize
    from scipy.special import expit
    z = np.asarray(logit, dtype=float); y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2: return dict(a=1., b=0., status='single_class_identity')
    def objective(v):
        a = np.exp(v[0]); u = a*z+v[1]; delta = expit(u)-y
        loss = np.mean(np.logaddexp(0, u)-y*u)+1e-5*np.sum(v**2)
        grad = np.array([np.mean(delta*a*z), np.mean(delta)])+2e-5*v
        return loss, grad
    fit = minimize(objective, np.zeros(2), jac=True, method='L-BFGS-B', bounds=[(-6, 6), (-20, 20)])
    if not fit.success: raise RuntimeError('Calibration failed: '+fit.message)
    return dict(a=float(np.exp(fit.x[0])), b=float(fit.x[1]), status='fitted')
