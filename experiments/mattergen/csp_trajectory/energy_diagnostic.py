"""Separate energy-plot gaps, nonlinear axes and genuine denoising changes."""
import argparse
import json
from pathlib import Path
import numpy as np
from energy_trajectory import audit_load,block_medians,sha

NATIVE=('mattergen/diffusion/sampling/pc_sampler.py',
    'mattergen/diffusion/sampling/predictors.py',
    'mattergen/diffusion/sampling/predictors_correctors.py',
    'mattergen/diffusion/wrapped/wrapped_predictors_correctors.py',
    'mattergen/common/diffusion/predictors_correctors.py',
    'mattergen/common/diffusion/corruption.py',
    'mattergen/diffusion/corruption/sde_lib.py')


def sigma_fractional(step,n=8):
    t=1.-np.asarray(step,dtype=float)*.001
    return .01*(5./.01)**t/n**(1/3)


def omitted_blocks(energy):
    energy=np.asarray(energy,dtype=float)
    return np.isfinite(energy.reshape(energy.shape[0],-1,25)).sum(2)<20


def largest_window_drop(values,width=50,minimum=40):
    a=np.asarray(values,dtype=float);candidates=[]
    for k in range(width,len(a)-width+1):
        first,second=a[k-width:k],a[k:k+width]
        if np.isfinite(first).sum()>=minimum and np.isfinite(second).sum()>=minimum:
            candidates.append((float(np.nanmedian(second)-np.nanmedian(first)),k))
    if not candidates:return None
    delta,k=min(candidates)
    return dict(center_step=k,median_change_eV_atom=delta,width=width,minimum_valid=minimum,
        role='Largest difference between adjacent window medians; not a physical transition or exact changepoint. Earliest tie.')


def main(source,native,output):
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=json.loads((source/'plan.json').read_text())
    pinned=original['checkpoint']['source_files']
    for name in NATIVE:
        if sha(native/name)!=pinned[name]:raise ValueError('Native source differs: '+name)
    config=Path(__file__).resolve().parents[0]/'../csp/model_config.yaml'
    spec=json.loads(config.with_name('sources.json').read_text())
    if sha(config)!=spec['config_sha256']:raise ValueError('CSP noise configuration changed.')
    cfg=yaml.safe_load(config.read_text())['lightning_module']['diffusion_module']['corruption']['sdes']['pos']
    if cfg['sigma_max']!=5. or cfg.get('sigma_min',.01)!=.01:raise ValueError('Unexpected positional noise schedule.')
    paths,energy,_,audit=audit_load(source)
    data=json.loads((source/'complete.json').read_text());distance=np.full_like(energy,np.nan);volume=np.full_like(energy,np.nan)
    for p in paths:
        with np.load(source/p['path']) as d:volume[p['trajectory']]=np.linalg.det(d['cell'][1:])/len(d['numbers'])
    for r in data['states']:
        if r['stage']=='predictor' and r.get('energy_eV') is not None:
            distance[r['trajectory'],r['step']]=r['minimum_pair_distance_A']
    relative=energy-energy[:,-1,None];omitted=omitted_blocks(energy)
    windows=[]
    for start in (400,450,500,550,600,750,900):
        stop=start+50;row=dict(start=start,end=stop-1)
        for name,a in [('energy',energy),('relative',relative),('minimum_pair',distance),('volume_per_atom',np.where(np.isfinite(energy),volume,np.nan))]:
            v=np.nanmedian(a[:,start:stop],axis=1)
            row[name+'_median_of_path_window_medians']=float(np.median(v))
        windows.append(row)
    drops=[dict(trajectory=p['trajectory'],system=p['system'],**largest_window_drop(energy[p['trajectory']])) for p in paths]
    before=np.nanmedian(energy[:,450:500],axis=1);after=np.nanmedian(energy[:,550:600],axis=1)
    paired_change=dict(before=[450,499],after=[550,599],paths=24,paths_decreased=int(np.sum(after<before)),
        median_paired_change_eV_atom=float(np.median(after-before)),
        minimum_observations_before=int(np.isfinite(energy[:,450:500]).sum(1).min()),
        minimum_observations_after=int(np.isfinite(energy[:,550:600]).sum(1).min()))
    output.mkdir(parents=True,exist_ok=False)
    steps=np.arange(1000);fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for ax,values,title,label in [(axes[0,0],relative,'Energy trend on a LINEAR vertical axis','E(step) - E(last observed), eV/atom'),
                                 (axes[0,1],distance,'Nearest pair distance in the same evaluated states','Periodic minimum pair distance (A)')]:
        q=np.nanquantile(values,[.25,.5,.75],axis=0)
        ax.fill_between(steps,q[0],q[2],color='#8ca9be',alpha=.35,label='Path IQR')
        ax.plot(steps,q[1],color='#235978',lw=1,label='Median of available paths')
        ax.set(title=title,xlabel='Predictor step',ylabel=label);ax.legend(fontsize=8)
    ax=axes[1,0]
    for n in (6,8,12):ax.plot(steps,sigma_fractional(steps,n),label=f'{n} atoms')
    ax.set(title='Smooth forward positional-noise schedule (not actual update size)',xlabel='Predictor step',ylabel='Unwrapped forward standard deviation (fractional)',yscale='log');ax.legend(fontsize=8)
    ax=axes[1,1];ax.bar(np.arange(40)*25+12,omitted.mean(0),width=23,color='#b96b3c')
    ax.set(title='Why the original solid lines have gaps',xlabel='Predictor step (25-step blocks)',ylabel='Fraction of paths with <20 valid states /25',ylim=(0,1))
    for ax in axes.flat:ax.axvline(500,color='#aa3333',ls='--',lw=.8,label='Step500');ax.grid(alpha=.2)
    fig.suptitle('Energy discontinuity diagnostic: original24 GSS CSP paths; no resampling / relaxation')
    fig.supxlabel('Energy and distance use available evaluated states (coverage changes). Relative zero is the last observed input, not a relaxed minimum.\n'
        'The sampler/noise/corrector sources are SHA-matched to the original run; there is no fixed500-step algorithm switch.',fontsize=9)
    for suffix in ('png','svg','pdf'):fig.savefig(output/f'energy-diagnostic.{suffix}',dpi=190)
    plt.close(fig)
    report=dict(audit=audit,omitted_blocks=int(omitted.sum()),total_blocks=omitted.size,
        omitted_before600=int(omitted[:,:24].sum()),sigma_fractional_8atoms={str(k):float(sigma_fractional(k)) for k in (450,499,500,501,550)},
        windows=windows,largest_window_drops=drops,
        paired_window_change=paired_change,
        native_source_sha256={name:sha(native/name) for name in NATIVE},config_sha256=sha(config),
        source_sha256={name:sha(source/name) for name in ('complete.json','plan.json')},code_sha256=sha(__file__),
        interpretation='Plot gaps follow missing records/block rules; symlog magnifies zero crossings. Actual energies and close contacts also change rapidly. Smooth-noise/local-environment explanation is an inference, not an identified transition or barrier measurement.')
    (output/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(omitted_blocks=report['omitted_blocks'],omitted_before600=report['omitted_before600'],windows=windows,
        sigma_fractional_8atoms=report['sigma_fractional_8atoms'],drop_center_range=[min(r['center_step'] for r in drops),max(r['center_step'] for r in drops)])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--native-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source.resolve(),a.native_source.resolve(),a.output.resolve())
