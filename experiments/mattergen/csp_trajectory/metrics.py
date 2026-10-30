"""Descriptive monotonicity statistics with explicit missing-step boundaries."""
import numpy as np

KEYS = ('cos_score_force', 'cos_drift_force')


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):
        return dict(n=0,mean=None,min=None,max=None,q05=None,q25=None,median=None,q75=None,q95=None)
    return dict(n=len(a),mean=float(a.mean()),min=float(a.min()),max=float(a.max()),
        **dict(zip(('q05','q25','median','q75','q95'),map(float,np.quantile(a,[.05,.25,.5,.75,.95])))))


def sequence(steps,values):
    pairs=sorted((int(s),float(v)) for s,v in zip(steps,values) if v is not None and np.isfinite(v))
    if len({s for s,v in pairs}) != len(pairs):
        raise ValueError('Duplicate step.')
    deltas=[];drawdown=0.;peak=None;last=None
    for s,v in pairs:
        if last is None or s != last[0]+1:
            peak=v
        else:
            deltas.append(v-last[1]);peak=max(peak,v);drawdown=max(drawdown,peak-v)
        last=(s,v)
    d=np.asarray(deltas)
    return dict(valid_points=len(pairs),adjacent_pairs=len(d),
        drops={str(t):int(np.sum(d < -t)) for t in (0.,.01,.05)},
        largest_single_drop=float(max(0.,-d.min())) if len(d) else None,
        maximum_contiguous_drawdown=float(drawdown) if pairs else None,
        first_step=pairs[0][0] if pairs else None,last_step=pairs[-1][0] if pairs else None)


def analyze(rows):
    rows=[r for r in rows if r['stage']=='predictor']
    ids=sorted({r['trajectory'] for r in rows})
    by={i:sorted((r for r in rows if r['trajectory']==i),key=lambda r:r['step']) for i in ids}
    paths=[]
    for i in ids:
        item=dict(trajectory=i,system=by[i][0]['system'],batch=by[i][0]['batch'])
        for key in KEYS:
            item[key]=sequence([r['step'] for r in by[i]],[r.get(key) for r in by[i]])
            blocks=[];steps=[]
            for start in range(0,1000,25):
                block=[r.get(key) for r in by[i] if start<=r['step']<start+25]
                if len(block)==25 and all(v is not None and np.isfinite(v) for v in block):
                    blocks.append(float(np.mean(block)));steps.append(start//25)
            item[key]['blocks25']=sequence(steps,blocks)
        paths.append(item)
    summary={}
    for key in KEYS:
        p=[r[key] for r in paths]
        summary[key]=dict(trajectories=len(p),paths_with_adjacent_pairs=sum(x['adjacent_pairs']>0 for x in p),
            adjacent_pairs=sum(x['adjacent_pairs'] for x in p),
            drops={str(t):sum(x['drops'][str(t)] for x in p) for t in (0.,.01,.05)},
            paths_with_drop={str(t):sum(x['drops'][str(t)]>0 for x in p) for t in (0.,.01,.05)},
            paths_with_block25_drop_005=sum(x['blocks25']['drops']['0.05']>0 for x in p),
            maximum_drawdown=distribution([x['maximum_contiguous_drawdown'] for x in p if x['maximum_contiguous_drawdown'] is not None]))
    means=[]
    for s in range(1000):
        frame=[r for r in rows if r['step']==s]
        means.append(dict(step=s,**{key:distribution([r[key] for r in frame if r.get(key) is not None]) for key in KEYS}))
    cohorts=[]
    for start in (800,900,950):
        item=dict(start=start,end=999)
        for key in KEYS:
            eligible=[i for i in ids if len([r for r in by[i] if start<=r['step']<=999 and r.get(key) is not None])==1000-start]
            curve=[float(np.mean([by[i][s][key] for i in eligible])) for s in range(start,1000)] if eligible else []
            item[key]=dict(trajectories=eligible,n=len(eligible),mean_curve=curve,
                first=curve[0] if curve else None,last=curve[-1] if curve else None,
                sequence=sequence(range(start,1000),curve))
        cohorts.append(item)
    return dict(paths=paths,summary=summary,per_step=means,fixed_cohorts=cohorts)
