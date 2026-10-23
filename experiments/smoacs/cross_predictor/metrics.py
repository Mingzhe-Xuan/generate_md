"""Agreement between predictors; no ground-truth accuracy is implied."""
import math
import numpy as np


def index_rows(rows, ids):
    indexed={r['candidate_id']:r for r in rows}
    if len(indexed)!=len(rows) or len(set(ids))!=len(ids) or set(indexed)!=set(ids):
        raise ValueError('Duplicate or mismatched candidate IDs.')
    return indexed


def classify(gap, energy, valid):
    if gap is None or energy is None:
        return dict(gap=None, eform=None, joint=None)
    if not all(math.isfinite(v) for v in (gap,energy)):
        raise ValueError('Nonfinite prediction.')
    gap_ok=2.3<=gap<=2.7
    ef_ok=energy<-.5
    return dict(gap=gap_ok,eform=ef_ok,joint=gap_ok and ef_ok and valid)


def numerical(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid paired predictions.')
    if not len(x):return dict(n=0)
    d=y-x
    corr=rank_corr=None
    if len(x)>1 and np.ptp(x)>1e-12 and np.ptp(y)>1e-12:
        from scipy.stats import spearmanr
        corr=float(np.corrcoef(x,y)[0,1])
        rank_corr=float(spearmanr(x,y).statistic)
    return dict(n=len(x),bias_alignn_minus_crystalformer=float(d.mean()),mae_between_predictors=float(np.abs(d).mean()),
                rmse_between_predictors=float(np.sqrt(np.mean(d*d))),pearson=corr,spearman=rank_corr,
                absolute_difference_quantiles=dict(zip(('min','p25','median','p75','p95','max'),
                    map(float,np.quantile(np.abs(d),[0,.25,.5,.75,.95,1])))))


def agreement(rows):
    result=dict(denominator=len(rows),classification={},numerical={})
    for metric in ('gap','eform','joint'):
        counts=dict(both_pass=0,crystalformer_only=0,alignn_only=0,neither_pass=0,unknown=0)
        cf_pass=al_pass=cf_unknown=al_unknown=0
        for row in rows:
            a,b=row['crystalformer_flags'][metric],row['alignn_flags'][metric]
            cf_pass+=a is True;al_pass+=b is True;cf_unknown+=a is None;al_unknown+=b is None
            label='unknown' if a is None or b is None else ('both_pass' if a and b else
                'crystalformer_only' if a else 'alignn_only' if b else 'neither_pass')
            counts[label]+=1
        known=len(rows)-counts['unknown']
        counts.update(crystalformer_pass=cf_pass,alignn_pass=al_pass,
            crystalformer_unknown=cf_unknown,alignn_unknown=al_unknown,
            known_pair_count=known,
            classification_agreement=(counts['both_pass']+counts['neither_pass'])/known if known else None,
            confirmed_retention_of_crystalformer_pass=counts['both_pass']/cf_pass if cf_pass else None)
        result['classification'][metric]=counts
    complete=[r for r in rows if r['status']=='completed']
    for prop in ('gap','eform'):
        result['numerical'][prop]=numerical([r['crystalformer_'+prop] for r in complete],
                                          [r['alignn_'+prop] for r in complete])
    return result
