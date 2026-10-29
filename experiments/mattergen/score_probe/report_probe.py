"""Hash/coverage audited held-out fits and descriptive direction statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_math import unit, heldout, paired_parent_interval


def stats(values):
    a = np.array([x for x in values if x is not None], dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                positive_fraction=float((a>0).mean()), q25=float(np.quantile(a,.25)), q75=float(np.quantile(a,.75))) if len(a) else dict(n=0)


def report(root):
    complete = json.loads((root/'complete.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    if not complete['passed'] or complete['policy'] != plan['policy']:
        raise ValueError('Native failure or policy mismatch.')
    rows = complete['states']
    policy = complete['policy']
    parents = sorted({r['parent_id'] for r in rows})
    keys = {(r['parent_id'],r['anchor'],r['t'],r['replica']) for r in rows}
    expected = {(p,a,t,r) for p in parents for a in policy['anchors'] for t in policy['times'] for r in range(policy['replicas'])}
    if len(parents) != 16 or keys != expected or len(rows) != len(expected):
        raise ValueError('Missing/duplicate parent-state coverage.')
    if set(parents) != {r['parent_id'] for r in plan['inputs']}:
        raise ValueError('Parent selection changed.')
    vectors = []
    for row in rows:
        path = root/row['vector_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['vector_sha256']:
            raise ValueError('Vector artifact changed.')
        with np.load(path, allow_pickle=False) as data:
            item = {k:data[k].copy() for k in ('F','G','restoration','conditional','unconditional','delta','guided',
                    'drift_conditional','drift_unconditional','drift_delta','drift_guided')}
        if any(not np.isfinite(v).all() for v in item.values()):
            raise ValueError('Nonfinite vectors.')
        np.testing.assert_allclose(item['delta'],item['conditional']-item['unconditional'],atol=1e-12)
        np.testing.assert_allclose(item['guided'],item['unconditional']+2*item['delta'],atol=1e-12)
        np.testing.assert_allclose(item['drift_delta'],item['drift_conditional']-item['drift_unconditional'],atol=1e-12)
        np.testing.assert_allclose(item['drift_guided'],item['drift_unconditional']+2*item['drift_delta'],atol=1e-12)
        vectors.append(dict(row,**item))
    groups = []
    configurations = [('force',['F']),('property',['G']),('joint',['F','G']),
                      ('noise',['restoration']),('noise_force',['restoration','F']),
                      ('noise_joint',['restoration','F','G'])]
    for anchor in policy['anchors']:
        for t in policy['times']:
            selected = [r for r in vectors if r['anchor']==anchor and r['t']==t]
            detail = dict(anchor=anchor,t=t,states=len(selected),noise_RMS_A=stats([r['noise_RMS_A'] for r in selected]),
                          cos_F_G=stats([r['cos_F_G'] for r in selected]),targets={})
            for target in ('unconditional','conditional','delta','guided','drift_unconditional','drift_conditional','drift_delta','drift_guided'):
                # A near-zero score has no directional target; retain it as explicitly missing.
                eligible = [dict(r,target=r[target]) for r in selected if np.linalg.norm(r[target])>1e-10]
                fits = {name:heldout(eligible,features) for name,features in configurations} if len({r['parent_id'] for r in eligible})>=3 else {}
                if fits:
                    fits['joint_over_force'] = paired_parent_interval(fits['joint'],fits['force'])
                    fits['property_over_noise_force'] = paired_parent_interval(fits['noise_joint'],fits['noise_force'])
                    # Fixed coordinate permutations destroy vector alignment while preserving every norm.
                    # This is a negative alignment control, not a composition-exchangeable property test.
                    rng=np.random.default_rng([policy['seed'],int(t*1000),0 if anchor=='raw' else 1])
                    shuffled=[dict(r,G=unit(r['G'])[rng.permutation(len(r['G']))]) for r in eligible]
                    fits['permuted_property_joint']=heldout(shuffled,['F','G'])
                detail['targets'][target]=dict(valid_states=len(eligible),unknown_states=len(selected)-len(eligible),
                    cosine_force=stats([r['cosine'][target]['force'] for r in selected]),
                    cosine_property=stats([r['cosine'][target]['property'] for r in selected]),
                    local_explained=stats([r['local_projection'].get(target,{}).get('explained_fraction') for r in selected]),fits=fits)
            groups.append(detail)
    output = dict(passed=True,parents=16,states=len(rows),groups=groups,elapsed_seconds=complete['elapsed_seconds'],
                  NFE=complete['NFE'],native_checks=complete['native_checks'],
                  limitations=['Fixed cell and species; position-only conditional slice, not full sampling trajectory.',
                    'Same fine-tuned checkpoint with condition enabled/disabled; disabled branch is not mattergen_base.',
                    'No fitted potential, physical score decomposition, causal attribution or post-relaxation gain established.',
                    'ALIGNN gradients at noisy states remain model derivatives, not validated physical bandgap gradients.',
                    'R2 uses a zero-vector baseline and unit directions; no per-coordinate intercept.',
                    '16 independent parents; noise states are correlated. Bootstrap is descriptive and does not refit folds.'])
    (root/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    lines=['# MatterGen score probe','',f'16 parents / {len(rows)} states. Time {complete["elapsed_seconds"]:.1f}s; MS NFE {complete["NFE"]}.',
           '', 'Direction-only leave-one-parent-out fits; zero-vector R2 baseline. Coefficients can have either sign.', '',
           '| Anchor | t | target | cos(force) | cos(property) | F R2 | G R2 | F+G R2 | property R2 gain | noise+F+G R2 |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        for target in ('unconditional','delta','drift_unconditional','drift_delta'):
            r=group['targets'][target]
            f=r['fits']
            if not f:
                continue
            lines.append(f'| {group["anchor"]} | {group["t"]} | {target} | {r["cosine_force"].get("mean",float("nan")):.4f} | {r["cosine_property"].get("mean",float("nan")):.4f} | '+
                         ' | '.join(f'{f[k]["heldout_R2_zero_baseline"]:.4f}' for k in ('force','property','joint'))+
                         f' | {f["joint_over_force"]["R2_gain"]:.4f} | {f["noise_joint"]["heldout_R2_zero_baseline"]:.4f} |')
    lines += ['',*output['limitations']]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    plot(root,groups)
    return output


def plot(root,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for j,anchor in enumerate(('raw','relaxed')):
        selected=[r for r in groups if r['anchor']==anchor]
        times=[r['t'] for r in selected]
        for target,style in [('unconditional','-o'),('delta','--s')]:
            for name in ('force','property'):
                axes[0,j].plot(times,[r['targets'][target]['cosine_'+name].get('mean',np.nan) for r in selected],style,label=f'{target} vs {name}')
        for key in ('force','joint','noise_joint'):
            axes[1,j].plot(times,[r['targets']['delta']['fits'].get(key,{}).get('heldout_R2_zero_baseline',np.nan) for r in selected],'-o',label=key)
        axes[0,j].set_title(anchor+' anchors')
        axes[0,j].set_ylabel('Mean cosine')
        axes[1,j].set_ylabel('Delta score: held-out R2')
        axes[1,j].set_xlabel('Diffusion time (position-only corruption)')
        for ax in axes[:,j]:
            ax.axhline(0,color='grey',lw=.6)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle('MatterSim force / ALIGNN target-loss descent probing MatterGen score')
    fig.tight_layout()
    fig.savefig(root/'probe.png',dpi=180)
    fig.savefig(root/'probe.svg')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    report(args.run.resolve())
