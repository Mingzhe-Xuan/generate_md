"""Audit finite relaxation-map responses and summarize by whole parent structures."""
import argparse
import json
from pathlib import Path
import numpy as np
from csp_model import sha
from relations import centered
from pullback_math import response


def interval_by_parent(values, seed=20261054):
    if len(values) < 2:return None
    groups=sorted(values)
    sample=np.array([values[g] for g in groups],dtype=float)
    rng=np.random.default_rng(seed)
    medians=np.median(sample[rng.integers(0,len(sample),size=(2000,len(sample)))],axis=1)
    return np.quantile(medians,[.025,.975]).tolist()


def controls_valid(case, direction, property_index, policy):
    checks=[r for r in case['controls'] if r['direction']==direction]
    if len(checks)!=len(policy['FD_steps_A']):return False
    for check in checks:
        if not check['same_graph']:return False
        ad=float(check['AD'][property_index])
        for key in ('FD','fixed_graph_FD'):
            fd=float(check[key][property_index])
            if abs(ad-fd)>policy['FD_abs']+policy['FD_rel']*max(abs(ad),abs(fd)):return False
    return True


def check_endpoint(row, numbers, cell, tolerance):
    if row['status']!='converged':raise ValueError('Unconverged endpoint presented as completed.')
    if not np.array_equal(row['numbers'],numbers):raise ValueError('Species invariant failed.')
    if not np.allclose(row['cell'],cell,rtol=0,atol=1e-10):raise ValueError('Fixed cell invariant failed.')
    force=np.linalg.norm(row['forces_eV_A'],axis=1).max()
    if force>tolerance*(1+1e-6):raise ValueError('Raw force exceeds claimed convergence tolerance.')


def summarize_rows(rows, policy):
    groups=[]
    for direction in policy['directions']:
        for amplitude in policy['amplitudes_RMS_A']:
            for tolerance in policy['tolerances_eV_A']:
                selected=[r for r in rows if r['direction']==direction and r['amplitude_RMS_A']==amplitude and r['tolerance_eV_A']==tolerance]
                for j, prop in enumerate(policy['property_order']):
                    for cohort in ('all_completed','gradient_QC'):
                        eligible=[r for r in selected if cohort=='all_completed' or r['gradient_QC'][j]]
                        defined={r['id']:r['absolute_response_retention'][j] for r in eligible if r['absolute_response_retention'][j] is not None}
                        if len({r['id'] for r in eligible})!=len(eligible):raise ValueError('Repeated parent in group; cluster before aggregation.')
                        raw=[r['raw_FD'][j] for r in eligible];post=[r['relaxed_FD'][j] for r in eligible]
                        groups.append(dict(direction=direction,amplitude_RMS_A=amplitude,tolerance_eV_A=tolerance,
                            property=prop,cohort=cohort,completed_parents=len(selected),eligible_parents=len(eligible),
                            ratio_defined_parents=len(defined),ratio_undefined_parents=len(eligible)-len(defined),
                            median_absolute_response_retention=float(np.median(list(defined.values()))) if defined else None,
                            median_retention_95_bootstrap=interval_by_parent(defined),
                            median_absolute_raw_slope=float(np.median(np.abs(raw))) if raw else None,
                            median_absolute_relaxed_slope=float(np.median(np.abs(post))) if post else None,
                            median_geometry_gain=float(np.median([r['geometry_response_gain'] for r in eligible])) if eligible else None,
                            parents=[r['id'] for r in eligible]))
    return groups


def execute(run, output):
    from ase.geometry import find_mic
    if output.exists():raise ValueError('Use a fresh report directory.')
    plan=json.loads((run/'plan.json').read_text());complete=json.loads((run/'complete.json').read_text())
    policy=plan['policy'];selected=plan['selection']['selected'];cases=complete['cases']
    if not complete['execution_complete'] or [c['id'] for c in cases]!=[c['id'] for c in selected]:
        raise ValueError('Selected structure coverage incomplete.')
    rows=[];coverage=[];maximum=0.
    for case in cases:
        coverage.append(dict(id=case['id'],system=case['system'],status=case['status'],
            direction_status=case.get('direction_status'),pairs=len(case['pairs']),
            completed_pairs=sum(r['status']=='completed' for r in case['pairs']),NFE=case['NFE']))
        if case['status']!='completed':continue
        expected=sum(value=='active' for value in case['direction_status'].values())*len(policy['amplitudes_RMS_A'])*len(policy['tolerances_eV_A'])
        if len(case['pairs'])!=expected:raise ValueError('Missing active-direction pairs.')
        numbers=np.array(case['numbers']);cell=np.array(case['cell'])
        for pair in case['pairs']:
            if pair['status']!='completed':continue
            tolerance=pair['tolerance_eV_A'];reference=case['references'][str(tolerance)]
            for endpoint in pair['physics']+[reference['physics']]:check_endpoint(endpoint,numbers,cell,tolerance)
            carts=[np.array(r['frac'])@cell for r in pair['physics']]
            difference=find_mic(carts[0]-carts[1],cell,pbc=True)[0]
            actual=response(pair['raw_plus']['values'],pair['raw_minus']['values'],
                pair['relaxed_plus']['values'],pair['relaxed_minus']['values'],
                case['original']['gradients'],reference['property']['gradients'],
                case['directions'][pair['direction']],difference,pair['amplitude_RMS_A'])
            for key in actual:
                a=np.array([np.nan if x is None else x for x in np.atleast_1d(actual[key])],dtype=float)
                b=np.array([np.nan if x is None else x for x in np.atleast_1d(pair['response'][key])],dtype=float)
                if not np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True):raise ValueError('Response reconstruction failed: '+key)
                if np.isfinite(a).any():maximum=max(maximum,float(np.max(np.abs(a[np.isfinite(a)]-b[np.isfinite(a)]))))
            target=policy['target_gap_eV']
            raw_values=np.array(case['original']['values']);post_values=np.array(reference['property']['values'])
            raw_plus=np.array(pair['raw_plus']['values']);post_plus=np.array(pair['relaxed_plus']['values'])
            rows.append(dict(actual,id=case['id'],system=case['system'],direction=pair['direction'],
                amplitude_RMS_A=pair['amplitude_RMS_A'],tolerance_eV_A=tolerance,
                gradient_QC=[controls_valid(case,pair['direction'],j,policy) for j in range(2)],
                endpoint_RMS_to_reference_A=pair['endpoint_RMS_to_reference_A'],
                raw_gap_loss_improvement=float(.5*((raw_values[0]-target)**2-(raw_plus[0]-target)**2)),
                relaxed_gap_loss_improvement=float(.5*((post_values[0]-target)**2-(post_plus[0]-target)**2)),
                raw_eform_improvement=float(raw_values[1]-raw_plus[1]),relaxed_eform_improvement=float(post_values[1]-post_plus[1])))
    report=dict(plan_sha256=sha(run/'plan.json'),complete_sha256=sha(run/'complete.json'),
        source_commit=plan['commit'],selected_parents=len(selected),coverage=coverage,rows=rows,
        groups=summarize_rows(rows,policy),max_response_reconstruction_error=maximum,
        raw_audit_passed=True,policy=policy,
        caveats=['Predictor responses, not DFT property accuracy or formation-energy stability certification.',
                 'Fixed-cell finite-amplitude relaxation map, not exact differentiation through the optimizer.',
                 'Ordered MIC distances do not establish crystallographic/basin equivalence.',
                 'Unconverged/zero-gradient/near-zero-raw-response cases retain explicit denominators.',
                 'Bootstrap resamples whole parent/system, not correlated branches; small-cohort intervals are exploratory.'])
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot(report,output)
    print(json.dumps(dict(selected=len(selected),completed=sum(c['status']=='completed' for c in cases),audited_pairs=len(rows))))


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    rows=report['rows'];amplitudes=report['policy']['amplitudes_RMS_A']
    for j,(prop,direction) in enumerate(zip(report['policy']['property_order'],['gap_loss_descent','eform_descent'])):
        for k,tol in enumerate(report['policy']['tolerances_eV_A']):
            selected=[r for r in rows if r['direction']==direction and r['tolerance_eV_A']==tol and r['gradient_QC'][j]]
            axis=axes[j,k]
            for i,amp in enumerate(amplitudes):
                ratios=[r['absolute_response_retention'][j] for r in selected if r['amplitude_RMS_A']==amp and r['absolute_response_retention'][j] is not None]
                if ratios:
                    axis.scatter(np.full(len(ratios),i),np.maximum(ratios,1e-8),s=26,alpha=.7,label=f'{amp} A (n={len(ratios)})')
                    axis.plot([i-.12,i+.12],[max(np.median(ratios),1e-8)]*2,color='black',linewidth=2)
            axis.axhline(1,color='grey',linestyle=':')
            axis.set_yscale('log');axis.set_xticks(range(len(amplitudes)),[str(a) for a in amplitudes])
            axis.set(title=f'{prop}; force tolerance {tol} eV/A',xlabel='Perturbation atomic RMS (angstrom)',ylabel='|Relaxed slope / raw slope|')
            if selected:axis.legend(fontsize=7)
            axis.grid(alpha=.2)
    fig.suptitle('Forward property response retained after relaxation\nProperty-descent directions; gradient-QC cases; each point is one parent')
    fig.savefig(output/'property-response.png',dpi=180);fig.savefig(output/'property-response.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();execute(args.run,args.output)


if __name__=='__main__':main()
