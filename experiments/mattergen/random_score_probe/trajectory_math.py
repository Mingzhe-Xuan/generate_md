"""Physical support and trajectory-level summaries for unconditional sampling."""
from collections import Counter
import numpy as np

STEPS = [0,100,250,500,650,750,800,850,900,925,950,975,990,995,999]


def project(v):
    v=np.asarray(v,dtype=float).reshape(-1,3)
    if not np.isfinite(v).all():
        raise ValueError('Nonfinite vector.')
    return v-v.mean(0)


def comparison(score_fractional,cell,force):
    force=project(force)
    score=project(np.asarray(score_fractional)@np.linalg.inv(cell).T)
    drift=project(np.asarray(score_fractional)@cell)
    def cosine(a,b):
        na,nb=np.linalg.norm(a),np.linalg.norm(b)
        return float(np.sum(a*b)/(na*nb)) if na>1e-10 and nb>1e-8 else None
    result=dict(force_norm=float(np.linalg.norm(force)),score_norm=float(np.linalg.norm(score)),
                drift_norm=float(np.linalg.norm(drift)),cos_score_force=cosine(score,force),
                cos_drift_force=cosine(drift,force))
    for key,v in [('score',score),('drift',drift)]:
        atoms=[cosine(a,b) for a,b in zip(v,force)]
        valid=[a for a in atoms if a is not None]
        result[key+'_atom_valid']=len(valid)
        result[key+'_atom_mean_cosine']=float(np.mean(valid)) if valid else None
        result[key+'_atom_positive_fraction']=float(np.mean(np.array(valid)>0)) if valid else None
        result[key+'_energy_directional_derivative']=-float(np.sum(force*v))/max(np.linalg.norm(v),1e-30)
    return result


def eligibility(numbers,cell,max_z=94,mask_z=101):
    numbers=np.asarray(numbers)
    cell=np.asarray(cell)
    reasons=[]
    if np.any(numbers==mask_z):reasons.append('masked_atom_types')
    if np.any((numbers<1)|((numbers>max_z)&(numbers!=mask_z))):reasons.append('unsupported_atom_types')
    if not np.isfinite(cell).all() or np.linalg.det(cell)<=1e-6 or np.linalg.svd(cell,compute_uv=False).min()<.3:
        reasons.append('invalid_or_nearly_singular_cell')
    elif np.linalg.det(cell)/len(numbers)<2:
        reasons.append('density_guard_volume_per_atom_below_2A3')
    return reasons


def summarize(rows):
    output=[]
    keys=sorted({(r['stage'],r['step']) for r in rows},key=lambda k:(k[1],0 if k[0]=='prior' else 1))
    for stage,step in keys:
        selected=[r for r in rows if (r['stage'],r['step'])==(stage,step)]
        summary=dict(stage=stage,step=step,t=selected[0]['t'],total=len(selected),
                     statuses=dict(Counter(r['status'] for r in selected)),
                     all_types_known=sum(r['masked_atoms']==0 for r in selected),
                     masked_atom_fraction_mean=float(np.mean([r['masked_atoms']/r['num_atoms'] for r in selected])))
        for key in ('cos_score_force','cos_drift_force'):
            valid=[r[key] for r in selected if r.get(key) is not None]
            if valid:
                # One snapshot per trajectory in this stage: bootstrap independent trajectories.
                a=np.array(valid)
                rng=np.random.default_rng([20261050,step,0 if key=='cos_score_force' else 1])
                boot=a[rng.integers(len(a),size=(4000,len(a)))].mean(1)
                summary[key]=dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                    q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),
                    positive_fraction=float(np.mean(a>0)),CI95=np.quantile(boot,[.025,.975]).tolist())
            else:summary[key]=dict(n=0,mean=None,CI95=None)
        output.append(summary)
    return output
