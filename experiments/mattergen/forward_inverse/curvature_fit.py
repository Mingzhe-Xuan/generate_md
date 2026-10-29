"""Leave-one-chemical-system-out tests of curvature laws; no per-test-case fitting."""
import numpy as np
from relations import harmonic_curvature


METHODS = ('zero', 'noise', 'scaled_noise', 'scaled_H', 'blurred_H')


def prediction(record, method, parameter=None):
    if method == 'zero':return np.zeros_like(record['observed'])
    if method == 'noise':return np.linalg.inv(record['C'])
    if method == 'scaled_noise':return parameter*np.linalg.inv(record['C'])
    if method == 'scaled_H':return parameter*record['H']
    if method == 'blurred_H':return harmonic_curvature(record['H'],record['C'],parameter)
    raise ValueError('Unknown curvature model: '+method)


def error(record, estimate):
    return float(np.sum((record['observed']-estimate)**2)/max(np.sum(record['observed']**2),1e-30))


def fit(records, method, beta_grid=None):
    if method in ('zero','noise'):return None
    if not records:raise ValueError('Empty training fold.')
    if method in ('scaled_noise','scaled_H'):
        numerator=denominator=0.
        for record in records:
            base=prediction(record,method,1.)
            weight=1/max(np.sum(record['observed']**2),1e-30)
            numerator+=weight*np.sum(record['observed']*base)
            denominator+=weight*np.sum(base**2)
        return float(max(0.,numerator/max(denominator,1e-30)))
    if method == 'blurred_H':
        beta_grid=np.logspace(-5,6,221) if beta_grid is None else np.asarray(beta_grid)
        losses=[np.mean([error(r,prediction(r,method,beta)) for r in records]) for beta in beta_grid]
        return float(beta_grid[int(np.argmin(losses))])
    raise ValueError('Unknown model.')


def heldout_systems(records, method):
    groups=sorted({r['system'] for r in records})
    if len(groups)<3:raise ValueError('Need at least three chemical systems for held-out fits.')
    folds=[];errors=[]
    for group in groups:
        train=[r for r in records if r['system']!=group]
        test=[r for r in records if r['system']==group]
        parameter=fit(train,method)
        test_errors=[error(r,prediction(r,method,parameter)) for r in test]
        errors.extend(test_errors)
        folds.append(dict(system=group,parameter=parameter,training_systems=sorted({r['system'] for r in train}),
            test_cases=[r['id'] for r in test],errors=test_errors))
    return dict(method=method,groups=len(groups),cases=len(records),folds=folds,
                R2_zero_baseline=float(1-np.mean(errors)),relative_errors=errors)
