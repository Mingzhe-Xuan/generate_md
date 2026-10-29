"""Independent raw-vector audit of frozen twelve-system score/force planes."""
import argparse
import json
from pathlib import Path
import numpy as np
from csp_model import sha
from relations import relative_error,matrix_diagnostics
from response_report import curl_evidence


def execute(root,source,output):
    if output.exists():raise ValueError('Use a fresh report directory.')
    complete=json.loads((root/'complete.json').read_text());policy=complete['policy']
    original=json.loads((source/'complete.json').read_text())
    if sha(source/'plan.json')!=complete['source_plan_sha256'] or sha(source/'complete.json')!=complete['source_complete_sha256']:
        raise ValueError('Source manifests changed.')
    if [c['id'] for c in complete['cases']]!=[c['id'] for c in original['cases']]:raise ValueError('Incomplete parent coverage.')
    ledger={}
    for line in (root/'physics/ledger.jsonl').read_text().splitlines():
        record=json.loads(line)
        if len(record['tasks'])!=9:raise ValueError('Unexpected physical point count.')
        ledger[record['tasks'][0]['id'].removesuffix('-base')]=record
    rows=[];audit_errors=[]
    for case in complete['cases']:
        row=dict(id=case['id'],system=case['system'],prior_system=case['prior_system'],status=case['status'],times=[])
        if case['status']!='completed':rows.append(row);continue
        point_rows=ledger[case['id']]['results']
        energies=np.array([r['energy_eV'] for r in point_rows[1:]]).reshape(2,2,2)
        all_force=np.array([r['forces_eV_A'] for r in point_rows[1:]]).reshape(2,2,2,-1)
        for time in case['times']:
            path=root/time['vectors']
            if sha(path)!=time['vectors_sha256']:raise ValueError('Vector hash mismatch.')
            with np.load(path) as z:
                plane=z['plane']
                np.testing.assert_allclose(plane.T@plane,np.eye(2),atol=1e-12)
                physical=np.stack([np.stack([plane.T@((p[0]-p[1])/(2*h)) for p in fields],axis=1)
                    for h,fields in zip(policy['steps_A'],all_force)])
                np.testing.assert_allclose(z['physical_forces'],all_force,atol=0,rtol=0)
                np.testing.assert_allclose(physical,case['physical_J'],atol=1e-10,rtol=0)
                for j,h in enumerate(policy['steps_A']):
                    control=case['energy_force_controls'][j]
                    fd=(energies[j,:,0]-energies[j,:,1])/(2*h)
                    ad=-plane.T@np.asarray(point_rows[0]['forces_eV_A']).ravel()
                    np.testing.assert_allclose(fd,control['FD'],atol=1e-10,rtol=0)
                    np.testing.assert_allclose(ad,control['AD'],atol=1e-10,rtol=0)
                details=dict(t=time['t'],dtypes={},fp32_fp64_value_relative_error=time['fp32_fp64_value_relative_error'])
                for dtype in ('single','double'):
                    meta=time['dtypes'][dtype]
                    matrix=z[dtype+'_adjoints']@plane
                    np.testing.assert_allclose(matrix,meta['matrix'],atol=1e-10,rtol=0)
                    modes={}
                    for mode in ('inference','grad_enabled'):
                        fields=z[dtype+'_'+mode+'_fields'].astype(float)
                        finite=np.stack([np.stack([plane.T@((p[0]-p[1])/(2*h)) for p in pairs],axis=1)
                            for h,pairs in zip(policy['steps_A'],fields)])
                        saved=meta['modes'][mode]
                        ad_same=all(g==meta['base_graph'] for g in meta['adjoint_graphs'])
                        graph_ok=[ad_same and all(a==b==meta['base_graph'] for a,b in hashes) for hashes in saved['pair_graphs']]
                        evidence=curl_evidence(matrix,*np.eye(2),finite[:,:,0],finite[:,:,1],graph_ok)
                        if relative_error(z[dtype+'_score'],z[dtype+'_repeat'])>policy['native_replay_relative']:
                            evidence['corroborated_nonzero_curl']=False
                        if evidence['corroborated_nonzero_curl']!=saved['corroborated_nonzero_curl']:
                            raise ValueError('Curl flag differs from raw evidence.')
                        audit_errors.append(abs(evidence['curl_AD']-saved['curl_AD']))
                        modes[mode]=evidence
                    details['dtypes'][dtype]=dict(modes=modes,plane_antisymmetric_fraction=matrix_diagnostics(matrix)['antisymmetric_fraction'],
                        grad_vs_inference_relative_error=meta['grad_vs_inference_relative_error'])
                row['times'].append(details)
        row.update(physical_antisymmetric_fractions=[matrix_diagnostics(j)['antisymmetric_fraction'] for j in physical],
            physical_FD_relative_change=relative_error(*physical),
            energy_force_QC=all(all(r['passed']) for r in case['energy_force_controls']))
        rows.append(row)
    counts=[]
    for cohort in ('all','prior4systems','new8systems'):
        selected=[r for r in rows if cohort=='all' or (r['prior_system'] if cohort=='prior4systems' else not r['prior_system'])]
        for dtype in ('single','double'):
            for mode in ('inference','grad_enabled'):
                points=[t['dtypes'][dtype]['modes'][mode] for r in selected for t in r['times']]
                counts.append(dict(cohort=cohort,dtype=dtype,mode=mode,selected_parents=len(selected),
                    planned_case_times=len(selected)*len(policy['times']),evaluated=len(points),
                    numerical_direction_QC=sum(p['numerical_direction_QC'] for p in points),
                    corroborated_nonzero_curl=sum(p['corroborated_nonzero_curl'] for p in points)))
    report=dict(raw_audit_passed=True,source_commit=complete['source_commit'],experiment_sha256=sha(root/'complete.json'),
        selected_parents=len(rows),policy=policy,rows=rows,counts=counts,
        max_raw_curl_discrepancy=max(audit_errors,default=0.),
        physical_energy_force_QC_parents=sum(r.get('energy_force_QC',False) for r in rows),
        caveats=['Fixed independent random planes and two step sizes; not full-matrix certification.',
                 'FP64 is secondary; failed native equivalence is not upgraded.',
                 'Counts are descriptive fixed-study coverage, not independent population prevalence.',
                 'A failed numerical/graph check is inconclusive, not evidence of conservativity.'])
    output.mkdir(parents=True);(output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot(report,output)
    print(json.dumps(counts))


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    rows=report['rows'];x=np.arange(len(rows))
    for offset,dtype,label in [(-.18,'single','Native FP32'),(.18,'double','FP64 diagnostic')]:
        values=[sum(t['dtypes'][dtype]['modes']['inference']['corroborated_nonzero_curl'] for t in r['times']) for r in rows]
        axes[0].bar(x+offset,values,width=.36,label=label)
    axes[0].set(ylabel='Corroborated nonzero-curl points / 4',ylim=(0,4.6),title='Frozen-window random-plane confirmation: all 12 selected systems')
    axes[0].legend()
    native=[np.median([t['dtypes']['single']['plane_antisymmetric_fraction'] for t in r['times']]) for r in rows]
    physical=[np.median(r['physical_antisymmetric_fractions']) for r in rows]
    axes[1].semilogy(x,native,'o-',label='Native score plane (AD; numerical QC separate)')
    axes[1].semilogy(x,physical,'-',color='C1',alpha=.6)
    good=np.array([r.get('energy_force_QC',False) for r in rows])
    axes[1].semilogy(x[good],np.asarray(physical)[good],'s',color='C1',label='Force plane: energy/force QC passed')
    axes[1].semilogy(x[~good],np.asarray(physical)[~good],'s',color='C1',markerfacecolor='none',label='Force plane: energy/force QC failed')
    axes[1].set(ylabel='Median normalized plane antisymmetry',title='Dimensionless local antisymmetry; different field units')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x,[r['system'] for r in rows],rotation=25);axis.grid(axis='y',alpha=.2)
    fig.savefig(output/'confirmed-curl.png',dpi=180);fig.savefig(output/'confirmed-curl.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    execute(args.run,args.source,args.output)


if __name__=='__main__':main()
