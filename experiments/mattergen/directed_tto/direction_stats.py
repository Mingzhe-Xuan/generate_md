"""Parent-paired post-quench direction diagnostics without Best-of-N selection."""
import numpy as np
from settings import FAMILIES, ARMS, SCALES, interval


def contrast(delta):
    values=np.asarray(delta,dtype=float)
    return dict(parents=len(values),mean_delta=float(values.mean()),CI95=interval(values),
                better=int(sum(values<-.001)),worse=int(sum(values>.001)),ties_1meV=int(sum(abs(values)<=.001)))


def summarize_directions(attempts,parents,fallback_errors):
    if len(parents)!=16 or len(set(parents))!=16:raise ValueError('Require16 unique frozen parents.')
    fallback=np.asarray(fallback_errors,dtype=float)
    if fallback.shape!=(16,) or not np.isfinite(fallback).all():raise ValueError('Invalid shared fallback scores.')
    keyed={(r['parent_id'],r['family'],r['arm'],r['scale']):r for r in attempts}
    expected={(p,f,a,s) for p in parents for f in FAMILIES for a in ARMS for s in SCALES}
    if set(keyed)!=expected or len(attempts)!=len(expected):raise ValueError('Missing/duplicated direction attempts.')
    output={}
    for family in FAMILIES:
        values={};eligible={}
        for arm in ARMS:
            rows=[[keyed[p,family,arm,s] for s in SCALES] for p in parents]
            mask=np.array([[bool(r['eligible']) for r in row] for row in rows]);eligible[arm]=mask
            errors=np.array([[abs(float(r['point']['gap'])-2.5) if r['eligible'] else fallback[i]
                              for r in row] for i,row in enumerate(rows)])
            if not np.isfinite(errors).all():raise ValueError('Nonfinite eligible prediction.')
            values[arm]=errors
        primary={arm:dict(MAE=float(v.mean()),eligible_attempts=int(eligible[arm].sum()),
                         fallback_attempts=int((~eligible[arm]).sum())) for arm,v in values.items()}
        primary['descent_vs_controls']={a:contrast((values['descent']-values[a]).mean(1)) for a in ('random','reverse')}
        primary['descent_vs_shared_zero_step']=contrast(values['descent'].mean(1)-fallback)
        sensitivity={}
        for arm in ('random','reverse'):
            mask=eligible['descent']&eligible[arm];deltas=values['descent']-values[arm]
            grouped=np.array([deltas[i][mask[i]].mean() for i in range(16) if mask[i].any()])
            sensitivity[arm]=dict(paired_attempts=int(mask.sum()),parents=int(sum(mask.any(1))),
                                  parent_mean_contrast=contrast(grouped) if len(grouped) else None)
        output[family]=dict(all_attempts_shared_fallback=primary,eligible_pairs_only=sensitivity,
            per_scale={str(s):{a:float(values[a][:,k].mean()) for a in ARMS} for k,s in enumerate(SCALES)})
    return dict(note='Descriptive unselected proposals, not Best4. Equal parent weights: average scales within each parent before parent bootstrap. Every failed/ineligible attempt retains its budget and shared baseline/seed fallback; eligible-only sensitivity reports changing coverage. Negative contrasts favor descent. No primary acceptance change.',families=output)
