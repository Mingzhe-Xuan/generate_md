"""Slurm CPU audit and descriptive report of dense pure-CSP trajectories."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tarfile
import numpy as np
from metrics import analyze,KEYS


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((root/'complete.json').read_text())
    plan=json.loads((root/'composition_plan.json').read_text())
    if not result['passed'] or len(result['paths'])!=24 or len(result['states'])!=24024:
        raise ValueError('Incomplete run.')
    rows={r['id']:r for r in result['states']}
    if len(rows)!=24024:raise ValueError('Duplicate states.')
    arrays={};mapping={r['system']:{int(z):n for z,n in r['atomic_counts'].items()} for r in plan['compositions']}
    for path in result['paths']:
        p=root/path['path']
        if sha(p)!=path['sha256'] or sha(root/path['terminal_path'])!=path['terminal_sha256']:
            raise ValueError('Array checksum mismatch.')
        with np.load(p,allow_pickle=False) as d:arrays[path['trajectory']]={k:d[k].copy() for k in d.files}
        d=arrays[path['trajectory']]
        if Counter(map(int,d['numbers']))!=mapping[path['system']]:raise ValueError('Composition mismatch.')
        if list(d['stage'])!=['prior']+['predictor']*1000 or list(d['step'])!=[0]+list(range(1000)):
            raise ValueError('Frame schedule mismatch.')
        if not all(np.isfinite(d[k]).all() for k in ('cell','frac','score_fractional','t')):
            raise ValueError('Nonfinite arrays.')
        with np.load(root/path['terminal_path']) as terminal:
            if not np.array_equal(terminal['numbers'],d['numbers']):raise ValueError('Terminal species changed.')
        for k in range(1001):
            row=rows[f'{path["trajectory"]}:{k}']
            if (row['frame'],row['step'],row['stage'],row['system'],row['batch'])!=(k,int(d['step'][k]),str(d['stage'][k]),path['system'],path['batch']):
                raise ValueError('Metadata-frame mismatch.')
    observed=set();max_error=0.;nfe=0
    with (root/'physics/ledger.jsonl').open() as f:
        for line in f:
            ledger=json.loads(line)
            for task,point in zip(ledger['tasks'],ledger['results']):
                key=task['id'];row=rows[key];d=arrays[row['trajectory']];k=row['frame']
                if key in observed or key!=point['id'] or task['action']!='point':raise ValueError('Invalid ledger coverage.')
                observed.add(key);nfe+=point['NFE']
                for field in ('numbers','cell','frac'):
                    expected=d[field] if field=='numbers' else d[field][k]
                    if not np.array_equal(task[field],expected):raise ValueError('Ledger geometry mismatch.')
                if point['status']!='completed':
                    if row['status']!='force_failed':raise ValueError('Failed point status mismatch.')
                    continue
                if not np.array_equal(point['numbers'],task['numbers']) or not np.allclose(point['cell'],task['cell'],atol=1e-12,rtol=0) or not np.allclose(point['frac'],task['frac'],atol=1e-12,rtol=0):
                    raise ValueError('Single point changed geometry/species.')
                force=np.asarray(point['forces_eV_A']);force=force-force.mean(axis=0)
                lattice=d['cell'][k];sf=d['score_fractional'][k]
                # Independent coordinate conversion: solve L * s_cart^T = s_frac^T.
                score=np.linalg.solve(lattice,sf.T).T
                drift=np.einsum('ni,ij->nj',sf,lattice)
                for field,vector in zip(KEYS,(score,drift)):
                    vector=vector-vector.mean(axis=0)
                    a,b=np.linalg.norm(vector),np.linalg.norm(force)
                    cosine=float(np.dot(vector.ravel(),force.ravel())/(a*b)) if a>1e-10 and b>1e-8 else None
                    stored=row[field]
                    if (cosine is None)!=(stored is None):raise ValueError('Stationary status mismatch.')
                    if cosine is not None:
                        error=abs(cosine-stored);max_error=max(max_error,error)
                        if error>1e-10:raise ValueError('Raw cosine audit failed.')
    if len(observed)!=result['requests'] or nfe!=result['NFE']:raise ValueError('Force accounting mismatch.')
    for row in rows.values():
        if row['id'] not in observed and (row['status']!='unknown' or not row['reasons'] or any(row[k] is not None for k in KEYS)):
            raise ValueError('Unknown state invalidly counted.')
    if sum(a['score_calls'] for a in result['sampling_audit'])!=6000 or result['native_forward_calls']!=6000:
        raise ValueError('Native forward accounting mismatch.')
    controls=json.loads((root/'force-controls.json').read_text())
    control_points={}
    with (root/'force-controls/ledger.jsonl').open() as f:
        for line in f:
            entry=json.loads(line)
            for task,point in zip(entry['tasks'],entry['results']):
                if task['id'] in control_points or task['id']!=point['id'] or point['status']!='completed':
                    raise ValueError('Force control ledger invalid.')
                control_points[task['id']]=(task,point)
    for check in controls['checks']:
        if 'epsilon' not in check:continue
        i=check['trajectory'];eps=check['epsilon'];d=arrays[i];cell=d['cell'][-1];cart=d['frac'][-1]@cell
        direction=np.arange(cart.size,dtype=float).reshape(cart.shape)**2
        direction-=direction.mean(0);direction/=np.linalg.norm(direction)
        base=control_points[f'base:{i}'][1]
        expected=-float(np.sum(np.asarray(base['forces_eV_A'])*direction))
        endpoints=[]
        for sign in (1,-1):
            task,point=control_points[f'{i}:{eps}:{sign}']
            if not np.allclose(np.asarray(task['frac'])@cell,cart+sign*eps*direction,atol=1e-12,rtol=0):
                raise ValueError('Force control displacement mismatch.')
            endpoints.append(point['energy_eV'])
        fd=(endpoints[0]-endpoints[1])/(2*eps)
        if abs(fd-check['FD'])>1e-12 or abs(expected-check['expected'])>1e-12 or not check['passed'] or abs(fd-expected)>.005+.02*max(abs(fd),abs(expected)):
            raise ValueError('Force energy-derivative audit failed.')
    if not controls['passed'] or len(control_points)!=controls['requests']:
        raise ValueError('Incomplete force controls.')
    analysis=analyze(result['states'])
    analysis.update(audit=dict(passed=True,array_paths=24,states=24024,ledger_points=len(observed),NFE=nfe,
        maximum_cosine_error=max_error,native_batched_forward_calls=6000),
        statuses=dict(Counter(r['status'] for r in rows.values())),
        exclusions=dict(Counter(reason for r in rows.values() for reason in r['reasons'])),
        elapsed_seconds=result['elapsed_seconds'],force_controls=controls,
        input_sha256={name:sha(root/name) for name in ('complete.json','plan.json','composition_plan.json','physics/ledger.jsonl','force-controls.json')})
    (root/'report.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    colors=['#1967a3','#d05d21'];labels=['Cartesian score / force','Coordinate denoising direction / force']
    x=np.arange(1000)
    for key,color,label in zip(KEYS,colors,labels):
        means=np.array([r[key]['mean'] if r[key]['mean'] is not None else np.nan for r in analysis['per_step']])
        q25=np.array([r[key]['q25'] if r[key]['q25'] is not None else np.nan for r in analysis['per_step']])
        q75=np.array([r[key]['q75'] if r[key]['q75'] is not None else np.nan for r in analysis['per_step']])
        axes[0,0].plot(x,means,color=color,label=label,lw=1)
        axes[0,0].fill_between(x,q25,q75,color=color,alpha=.12)
        axes[0,1].plot(x,[r[key]['n'] for r in analysis['per_step']],color=color,label=label)
        cohort=analysis['fixed_cohorts'][0][key]
        axes[1,0].plot(range(800,1000),cohort['mean_curve'] or [np.nan]*200,color=color,label=f'{label} (n={cohort["n"]})',lw=1)
        dd=[r[key]['maximum_contiguous_drawdown'] for r in analysis['paths'] if r[key]['maximum_contiguous_drawdown'] is not None]
        axes[1,1].hist(dd,bins=np.linspace(0,2,21),alpha=.5,color=color,label=label)
    axes[0,0].set(title='Observed-state mean and path IQR (changing coverage)',xlabel='Predictor step',ylabel='Cosine similarity',ylim=(-1,1))
    axes[0,1].set(title='Eligible nonstationary states',xlabel='Predictor step',ylabel='Paths with defined cosine',ylim=(0,25))
    axes[1,0].set(title='Fixed complete cohort: last 200 predictor inputs',xlabel='Predictor step',ylabel='Mean cosine',ylim=(-1,1))
    axes[1,1].set(title='Maximum drawdown within contiguous observed segments',xlabel='Cosine decrease from preceding peak',ylabel='Number of paths')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Pure MatterGen CSP: GSS 12 systems, 24 paths in 3 native coupled batches')
    for suffix in ('png','svg'):fig.savefig(root/f'trajectories.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(14,12),sharex=True,sharey=True,constrained_layout=True)
    systems=sorted({r['system'] for r in result['paths']})
    for ax,system in zip(axes.flat,systems):
        for path in [p for p in result['paths'] if p['system']==system]:
            frame=sorted([r for r in rows.values() if r['trajectory']==path['trajectory'] and r['stage']=='predictor'],key=lambda r:r['step'])
            for key,color in zip(KEYS,colors):
                values=np.array([np.nan if r[key] is None else r[key] for r in frame])
                ax.plot(x,values,color=color,alpha=.18,lw=.5)
                block=np.array([np.mean(values[s:s+25]) for s in range(0,1000,25)])
                ax.plot(np.arange(12,1000,25),block,color=color,lw=1)
        ax.set(title=system,ylim=(-1,1));ax.grid(alpha=.2)
    fig.supxlabel('Predictor step (faint: actual inputs; solid: complete 25-step block means)')
    fig.supylabel('Cosine similarity');fig.suptitle('Blue: Cartesian score; orange: coordinate denoising direction')
    for suffix in ('png','svg'):fig.savefig(root/f'systems.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# Pure CSP denoising alignment on GSS','',
        'Twelve GSS systems, two random paths each; native random coordinates/cell, fixed composition, 1000 predictor/corrector steps. '
        'Three native batches of eight paths; paths within each adaptive-corrector batch are coupled. '
        'No property condition, force guidance, relaxation or reference-geometry initialization. Training overlap is unknown; '
        'this is an observational mechanism experiment, not an unseen CSP recovery benchmark.','',
        f'GPU elapsed: {result["elapsed_seconds"]:.1f} s. Captured 24,024 states; '
        f'{len(observed):,} MatterSim single points. Statuses: {analysis["statuses"]}.',
        'Unknown geometries and stationary directions are excluded from cosine denominators; no missing value is replaced by zero. '
        'The guards are pair distance >=0.5 A, volume/atom >=2 A^3, cell minimum singular value >=0.3 A and supported species.','',
        '| Direction | Paths with adjacent valid steps | Any decrease | Decrease >0.01 | Decrease >0.05 | Downward pairs / all adjacent pairs | Median max drawdown |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,label in zip(KEYS,labels):
        s=analysis['summary'][key]
        lines.append(f'| {label} | {s["paths_with_adjacent_pairs"]} | {s["paths_with_drop"]["0.0"]} | {s["paths_with_drop"]["0.01"]} | {s["paths_with_drop"]["0.05"]} | {s["drops"]["0.0"]}/{s["adjacent_pairs"]} | {s["maximum_drawdown"]["median"]:.4f} |')
    lines+=['','Only genuinely adjacent predictor steps enter these counts. Missing steps break drawdown segments.',
        'A positive overall trend is not equivalent to monotonicity of an individual stochastic path.','',
        '| Fixed cohort | Direction | Paths complete throughout window | Mean at first step | Mean at step999 | Mean-curve downward transitions >0.01 |',
        '|---|---|---:|---:|---:|---:|']
    for cohort in analysis['fixed_cohorts']:
        for key,label in zip(KEYS,labels):
            c=cohort[key]
            lines.append(f'| {cohort["start"]}–999 | {label} | {c["n"]} | {c["first"]} | {c["last"]} | {c["sequence"]["drops"]["0.01"]} |')
    lines+=['','The Cartesian score is a transformed covector, `s_frac @ inv(cell).T`; the coordinate deterministic '
        'denoising direction is `s_frac @ cell`. Both are centered, as is the physical force. Only coordinate alignment is measured; '
        'the lattice/stress direction is not included. Native stochastic corrector and predictor noise remain active.','',
        f'Independent raw-array/force-ledger audit passed; maximum cosine reconstruction error {max_error:.3g}. '
        '6000 batched native forwards, 1001 snapshots per path, all ordered species preserved, observer RNG unchanged.','',
        '![Trajectory summary](trajectories.png)','','![System paths](systems.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in ('evidence.tar.gz','evidence.tar.gz.sha256')]
    with tarfile.open(root/'evidence.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=str(p.relative_to(root)))
    (root/'evidence.tar.gz.sha256').write_text(sha(root/'evidence.tar.gz')+'  evidence.tar.gz\n')
    print(json.dumps(dict(audit=analysis['audit'],summary=analysis['summary'],archive_sha256=sha(root/'evidence.tar.gz'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(args.run.resolve())
