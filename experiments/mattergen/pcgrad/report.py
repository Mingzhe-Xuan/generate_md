"""Compare native PCGrad with immutable matched linear-force pilot results."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())


def helpers():
    path=Path(__file__).resolve().parents[1]/'alignn_templates/report.py'
    spec=importlib.util.spec_from_file_location('pcgrad_report_helpers',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();baseline=args.baseline.resolve()
    old=load(baseline/'comparison.json');ids=old['ids']
    if len(ids)!=4:
        raise ValueError('Require the immutable matched four-case pilot.')
    util=helpers();cases=dict(old['cases']);metrics=dict(old['metrics'])
    protocols={};statuses={};diagnostics={};paired={}
    for weight in (.1,.2,.3):
        folder=root/f'lambda-{weight}';name=f'PCGrad-{weight}'
        protocol,rows,status=util.validate_run(folder,'CF')
        composition=protocol['force_mixing']['composition'];audit=load(folder/'pcgrad-audit.json')
        proof=load(folder/'physics-gradient-probe.json')
        if protocol['ids']!=ids or composition['name']!='PCGrad' or composition!=audit['protocol'] or composition['weight']!=weight:
            raise ValueError('PCGrad method, budget or source IDs changed.')
        if not proof['passed'] or audit['gradient_calls']!=200:
            raise ValueError('Missing physics or actual projection-call proof.')
        traces=[json.loads(line) for line in (folder/'pcgrad-trace.jsonl').read_text().splitlines()]
        if [t['A_step'] for t in traces]!=list(range(1,201)) or any(t['weight']!=weight for t in traces):
            raise ValueError('Missing PCGrad steps or wrong weight.')
        points=[c for t in traces for c in t['cases']]
        endpoint=load(folder/'force-mixing-audit.json')['endpoint']
        counts={};timing={};strict={}
        for model in ('crystalformer','alignn'):
            strict[model]=dict(force_and_eform=0,force_and_joint=0)
        after={r['candidate_id']:r for r in rows if r['stage']=='after'}
        a_rows=load(folder/'results.json')['cycles'][0]['candidates']
        timing['A_phase_seconds']=max(r['a_phase_seconds'] for r in a_rows)
        b_seconds=[];b_steps=[];energy_drop=[]
        for key in ids:
            b=load(folder/'evaluation-cycle1/relaxation'/Path(key).stem/'result.json')
            ok=b['status']=='converged' and b['max_raw_atomic_force']<=.01 and b['max_filter_force']<=.01
            counts[key]=dict(status=b['status'],strict_force=ok,raw_force=b.get('max_raw_atomic_force'),filter_force=b.get('max_filter_force'))
            b_seconds.append(b.get('elapsed_seconds',0.))
            if b['status']=='converged':
                b_steps.append(b['optimizer_steps'])
                energy_drop.append((b['initial_energy_eV']-b['final_energy_eV'])*1000/b['num_atoms'])
            for model in strict:
                strict[model]['force_and_eform']+=int(ok and after[key][model+'_flags']['eform'] is True)
                strict[model]['force_and_joint']+=int(ok and after[key][model+'_flags']['joint'] is True)
        timing.update(sum_B_case_seconds=sum(b_seconds),median_B_steps=statistics.median(b_steps) if b_steps else None,
            median_B_energy_drop_meV_atom=statistics.median(energy_drop) if energy_drop else None)
        for stage,label in [('before','A'),('after','A_B')]:
            selected=[r for r in rows if r['stage']==stage]
            if len(selected)!=4 or {r['candidate_id'] for r in selected}!=set(ids):
                raise ValueError('Missing or duplicated original ID.')
            cases[name+'_'+label]=selected;metrics[name+'_'+label]=util.summarize(selected)
        paired[name]=util.paired(old['cases'][f'lambda-{weight}_A_B'],cases[name+'_A_B'],ids)
        protocols[name]=protocol;statuses[name]=status
        diagnostics[name]=dict(conflict_fraction=sum(c['conflict'] for c in points)/len(points),
            conflicting_structure_steps=audit['conflicting_structure_steps'],recorded_structure_steps=len(points),
            nonzero_projected_pairs=sum(c['projected_cosine'] is not None for c in points),
            post_projection_negative_pairs=sum(c['projected_cosine'] is not None and c['projected_cosine'] < -1e-8 for c in points),
            pre_B_raw_force_median=statistics.median(r['max_raw_force'] for r in endpoint),
            strict_force_count=sum(c['strict_force'] for c in counts.values()),B_cases=counts,strict_properties=strict,timing=timing)
    result=dict(denominator=4,ids=ids,metrics=metrics,cases=cases,protocols=protocols,B_status=statuses,
        diagnostics=diagnostics,paired_to_linear=paired,baseline=str(baseline),
        scope='Same raw four-parent pilot, fixed ordered species, lambda0.1/0.2/0.3, GPU Adam200 cosine, same final MatterSim B. '
        'Symmetric per-structure two-task PCGrad before Adam in scaled geometry variables. CF gap+Ef and raw total MatterSim energy tasks. '
        'Frozen saved-pose CF/ALIGNN scores, all IDs/unknowns retained. No SUN/DFT/MD; four-case feasibility pilot, not full-set evidence.')
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# PCGrad four-case comparison','',result['scope'],'',
        '| Method | CF gap | CF Ef | CF joint | CF MAE/eV | AL gap | AL Ef | AL joint | AL MAE/eV | Unknown CF/AL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,m in metrics.items():
        cf,al=m['crystalformer'],m['alignn']
        def value(v):return 'unknown' if v is None else f'{v:.6f}'
        lines.append(f"| {name} | {cf['gap_pass']}/4 | {cf['eform_pass']}/4 | {cf['joint_pass']}/4 | {value(cf['MAE'])} | {al['gap_pass']}/4 | {al['eform_pass']}/4 | {al['joint_pass']}/4 | {value(al['MAE'])} | {cf['unknown']}/{al['unknown']} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(metrics=metrics,B_status=statuses,diagnostics=diagnostics)),flush=True)


if __name__=='__main__':main()
