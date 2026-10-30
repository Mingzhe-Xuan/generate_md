"""Audited energy-versus-step plots for the original dense fixed-composition run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

WINDOWS=((0,99),(200,299),(450,549),(750,849),(900,999))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def per_atom(energy,n):
    if n<1 or not np.isfinite(energy):raise ValueError('Invalid energy or atom count.')
    return float(energy)/n


def record_energy(row,n):
    value=row.get('energy_eV')
    return None if value is None else per_atom(value,n)


def block_medians(values,size=25,minimum=20):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)%size or not 1<=minimum<=size:
        raise ValueError('Expected complete step grid and valid block settings.')
    blocks=values.reshape(-1,size);counts=np.isfinite(blocks).sum(1)
    medians=np.array([float(np.nanmedian(b)) if n>=minimum else np.nan for b,n in zip(blocks,counts)])
    return np.arange(len(blocks))*size+(size-1)/2,medians,counts


def fixed_cohort(values,start):
    values=np.asarray(values,dtype=float)
    ids=np.flatnonzero(np.isfinite(values[:,start:]).all(1))
    return ids,values[ids,start:]


def changes(values,start=0,end=None):
    a=np.asarray(values,dtype=float)[start:end];d=np.diff(a);d=d[np.isfinite(d)]
    return dict(adjacent_pairs=len(d),increases=int(np.sum(d>0)),decreases=int(np.sum(d<0)),
        increases_over_1meV=int(np.sum(d>.001)),decreases_over_1meV=int(np.sum(d<-.001)))


def audit_load(source):
    data=json.loads((source/'complete.json').read_text(encoding='utf-8'))
    old_report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    for name,digest in old_report['input_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Original audited input changed: '+name)
    if not data['passed'] or len(data['paths'])!=24 or len(data['states'])!=24024:
        raise ValueError('Incomplete original cohort.')
    paths=sorted(data['paths'],key=lambda r:r['trajectory']);arrays={}
    if [p['trajectory'] for p in paths]!=list(range(24)):raise ValueError('Path IDs changed.')
    for p in paths:
        if sha(source/p['path'])!=p['sha256']:raise ValueError('Trajectory array checksum changed.')
        with np.load(source/p['path'],allow_pickle=False) as d:arrays[p['trajectory']]={k:d[k].copy() for k in d.files}
        d=arrays[p['trajectory']]
        if list(d['stage'])!=['prior']+['predictor']*1000 or list(d['step'])!=[0]+list(range(1000)):
            raise ValueError('Unexpected dense observation schedule.')
    rows={r['id']:r for r in data['states']}
    if len(rows)!=24024:raise ValueError('Duplicate states.')
    observed=set();nfe=0
    with (source/'physics/ledger.jsonl').open(encoding='utf-8') as f:
        for line in f:
            ledger=json.loads(line)
            if len(ledger['tasks'])!=len(ledger['results']):raise ValueError('Mismatched ledger.')
            for task,point in zip(ledger['tasks'],ledger['results']):
                key=task['id']
                if key in observed or point['id']!=key or task['action']!='point':raise ValueError('Invalid single-point coverage.')
                observed.add(key);row=rows[key];d=arrays[row['trajectory']];frame=row['frame']
                for field in ('numbers','cell','frac'):
                    expected=d[field] if field=='numbers' else d[field][frame]
                    if not np.array_equal(task[field],expected):raise ValueError('Single-point geometry differs from captured state.')
                if point['status']=='completed':
                    if row.get('energy_eV')!=point['energy_eV'] or not np.isfinite(point['energy_eV']):
                        raise ValueError('Raw single-point energy audit failed.')
                elif row.get('energy_eV') is not None:raise ValueError('Failed point has a plotted energy.')
                nfe+=point['NFE']
    if len(observed)!=data['requests'] or nfe!=data['NFE']:raise ValueError('Single-point accounting changed.')
    energy=np.full((24,1000),np.nan);prior=np.full(24,np.nan);csv_rows=[]
    for p in paths:
        i=p['trajectory'];d=arrays[i];n=len(d['numbers'])
        for k in range(1001):
            row=rows[f'{i}:{k}']
            if row['frame']!=k or row['step']!=int(d['step'][k]) or row['stage']!=str(d['stage'][k]):
                raise ValueError('State indexing differs from arrays.')
            value=row.get('energy_eV')
            if value is not None:
                if row['id'] not in observed:raise ValueError('Energy without original model call.')
                value=record_energy(row,n)
            elif row['id'] not in observed and not row['reasons']:raise ValueError('Unknown energy has no reason.')
            if row['stage']=='predictor':energy[i,row['step']]=np.nan if value is None else value
            else:prior[i]=np.nan if value is None else value
            csv_rows.append(dict(trajectory=i,system=p['system'],batch=p['batch'],stage=row['stage'],step=row['step'],
                t=row['t'],num_atoms=n,status=row['status'],reasons=';'.join(row['reasons']),
                energy_eV=row.get('energy_eV'),energy_eV_atom=value))
    if not np.isfinite(energy[:,-1]).all():raise ValueError('All paths need a final observed reference energy.')
    audit=dict(passed=True,trajectories=24,all_states=24024,ledger_points=len(observed),NFE=nfe,
        predictor_energies=int(np.isfinite(energy).sum()),predictor_missing=int(np.isnan(energy).sum()),
        prior_energies=int(np.isfinite(prior).sum()),new_model_calls=0)
    return paths,energy,csv_rows,audit


def summarize(paths,energy):
    from scipy.stats import spearmanr
    results=[]
    for p in paths:
        i=p['trajectory'];values=energy[i];centers,median,counts=block_medians(values)
        valid=np.isfinite(median)
        rho=float(spearmanr(centers[valid],median[valid]).statistic) if valid.sum()>1 else None
        windows=[]
        for start,end in WINDOWS:
            a=values[start:end+1];count=int(np.isfinite(a).sum())
            windows.append(dict(start=start,end=end,n=count,median_eV_atom=float(np.nanmedian(a)) if count>=50 else None))
        row=dict(trajectory=i,system=p['system'],batch=p['batch'],num_finite=int(np.isfinite(values).sum()),
            end_observed_energy_eV_atom=float(values[-1]),windows=windows,
            block25_spearman=rho,peak_block_center_step=float(centers[np.nanargmax(median)]) if valid.any() else None,
            adjacent=changes(values),late_adjacent=changes(values,900,1000))
        first,last=windows[0]['median_eV_atom'],windows[-1]['median_eV_atom']
        row['late_minus_early_median_eV_atom']=None if first is None or last is None else last-first
        results.append(row)
    comparable=[r for r in results if r['late_minus_early_median_eV_atom'] is not None]
    early_later=[r for r in results if r['windows'][0]['median_eV_atom'] is not None and r['windows'][1]['median_eV_atom'] is not None]
    aggregate=dict(paths_with_early_late_windows=len(comparable),late_median_lower=sum(r['late_minus_early_median_eV_atom']<0 for r in comparable),
        paths_with_negative_block_spearman=sum(r['block25_spearman'] is not None and r['block25_spearman']<0 for r in results),
        early_200_299_median_higher_than_0_99=sum(r['windows'][1]['median_eV_atom']>r['windows'][0]['median_eV_atom'] for r in early_later),
        early_window_comparable=len(early_later),
        adjacent={k:sum(r['adjacent'][k] for r in results) for k in results[0]['adjacent']},
        late_adjacent={k:sum(r['late_adjacent'][k] for r in results) for k in results[0]['late_adjacent']})
    return results,aggregate


def plot(output,paths,energy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    relative=energy-energy[:,-1,None];steps=np.arange(1000)
    fig,axes=plt.subplots(5,3,figsize=(15,18),constrained_layout=True)
    ax=axes[0,0];q=np.nanquantile(relative,[.25,.5,.75],axis=0)
    ax.fill_between(steps,q[0],q[2],color='#597b99',alpha=.25,label='Path IQR')
    ax.plot(steps,q[1],color='#223f58',lw=1,label='Median of available paths')
    ax.axhline(0,color='k',lw=.6);ax.set_yscale('symlog',linthresh=.1)
    ax.set(title='Across24 paths: relative energy (changing coverage)',ylabel='E(step) - E(last observed), eV/atom',xlabel='Predictor step')
    ax.legend(fontsize=7)
    ids,late=fixed_cohort(relative,800);ax=axes[0,1]
    for a in late:ax.plot(np.arange(800,1000),a,color='#8ba2b6',alpha=.18,lw=.6)
    q=np.quantile(late,[.25,.5,.75],axis=0)
    ax.fill_between(np.arange(800,1000),q[0],q[2],color='#597b99',alpha=.25)
    ax.plot(np.arange(800,1000),q[1],color='#223f58',lw=1)
    ax.axhline(0,color='k',lw=.6);ax.set_yscale('symlog',linthresh=.01)
    ax.set(title=f'Late window: fixed complete cohort (n={len(ids)})',ylabel='Relative energy, eV/atom',xlabel='Predictor step')
    ax=axes[0,2];ax.plot(steps,np.isfinite(energy).sum(0),color='#355f82',lw=.8)
    ax.set(title='Energy coverage (unknown states stay missing)',xlabel='Predictor step',ylabel='Paths with evaluated energy',ylim=(0,25))
    systems=sorted({p['system'] for p in paths});colors=['#1967a3','#d05d21']
    minimum=float(np.nanmin(energy));maximum=float(np.nanmax(energy))
    for ax,system in zip(axes[1:].flat,systems):
        group=[p for p in paths if p['system']==system]
        for p,color in zip(group,colors):
            a=energy[p['trajectory']];center,med,count=block_medians(a)
            ax.plot(steps,a,color=color,alpha=.16,lw=.45)
            ax.plot(center,med,color=color,lw=1.4,label=f'Path{p["trajectory"]:02d}')
        ax.set(title=system,xlabel='Predictor step',ylabel='MatterSim E/N (eV/atom)',xlim=(0,999),ylim=(minimum-1,maximum*1.05))
        ax.set_yscale('symlog',linthresh=1.);ax.axhline(0,color='k',lw=.5,alpha=.5);ax.legend(fontsize=7,loc='upper right')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('MatterGen pure CSP denoising: MatterSim energy on ORIGINAL24 GSS paths\n'
        'Faint: actual single-point energies; solid:25-step medians (at least20 valid states/block)',fontsize=15)
    fig.supxlabel('Native predictor INPUT states0..999; no relaxation. Relative reference: own last observed input, not an energy minimum.\n'
        'Symlog axes retain negative energies and large early spikes. Paths within each native batch are coupled.',fontsize=10)
    for suffix in ('png','svg','pdf'):fig.savefig(output/f'energy-trajectories.{suffix}',dpi=190)
    plt.close(fig)
    return ids.tolist()


def main(source,output):
    paths,energy,csv_rows,audit=audit_load(source)
    results,aggregate=summarize(paths,energy)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'energies.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    cohort=plot(output,paths,energy)
    report=dict(audit=audit,aggregate=aggregate,paths=results,late_fixed_cohort=cohort,
        source=str(source),source_sha256={n:sha(source/n) for n in ('complete.json','report.json','plan.json','physics/ledger.jsonl')},
        code_sha256=sha(__file__),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        data_role='Original dense CSP run. Do not mix with the later800/1000 paired replay, which is numerically different.',
        reference='Energy per atom relative to own predictor-input999; not native returned mean1000 and not a relaxed minimum.',
        block_policy=dict(size=25,minimum_valid=20),windows=WINDOWS)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# MatterSim energy during original pure-CSP denoising','',
        'GSS12systems, two paths each, fixed compositions; original full1000-step native trajectory. '
        'All plotted values are existing MatterSim5M single points, with no relaxation or new model calls. '
        'This dense original run is distinct from the later800/1000 paired replay.','',
        f'Audit: {audit["ledger_points"]} original single points, {audit["predictor_energies"]} predictor-state energies, '
        f'{audit["predictor_missing"]} missing predictor states, {audit["prior_energies"]} separate prior energies. '
        'Every point energy and geometry was checked against the raw ledger/NPZ arrays and existing checksums.','',
        f'Late-window median (900–999) is lower than early-window median (0–99) in '
        f'{aggregate["late_median_lower"]}/{aggregate["paths_with_early_late_windows"]} paths. '
        f'All-step25-block Spearman correlation is negative for {aggregate["paths_with_negative_block_spearman"]}/24 paths. '
        f'The200–299 median exceeds the0–99 median in {aggregate["early_200_299_median_higher_than_0_99"]}/'
        f'{aggregate["early_window_comparable"]} comparable paths.','',
        'Adjacent raw-state changes (missing steps break comparisons): '+json.dumps(aggregate['adjacent'])+'.',
        'Last100-step changes: '+json.dumps(aggregate['late_adjacent'])+'.','',
        '| Path | System | Median E/N0–99 | Median E/N200–299 | Median E/N450–549 | Median E/N750–849 | Median E/N900–999 | Block25 Spearman |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        v=[w['median_eV_atom'] for w in row['windows']]
        lines.append('| '+str(row['trajectory'])+' | '+row['system']+' | '+' | '.join('NA' if x is None else f'{x:.5f}' for x in v)+f' | {row["block25_spearman"]:.4f} |')
    lines+=['','Absolute energies from different compositions have different baselines; compare within-path changes or the relative overview. '
        'The relative zero is the final observed predictor input, not a relaxed energy minimum. '
        'Solid curves use only blocks with at least20/25 valid states, with gaps retained. '
        'Early highly distorted states are ML-potential predictions; these curves alone do not validate physical barrier heights. '
        'The position and cell both evolve, and native stochastic updates remain active. '
        'These are descriptive trajectories from three coupled native batches, without independent-path significance claims.','',
        '![Energy trajectories](energy-trajectories.png)','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(audit=audit,aggregate=aggregate,late_fixed_cohort=cohort,output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.source.resolve(),args.output.resolve())
