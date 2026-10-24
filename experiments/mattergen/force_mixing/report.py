"""Paired fixed-species force blending versus existing CF200 and direct B."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())


def helpers():
    path=Path(__file__).resolve().parents[1]/'alignn_templates/report.py'
    spec=importlib.util.spec_from_file_location('force_comparison_helpers',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    root=args.run.resolve()
    util=helpers()
    raw=load(args.source/'cross-prediction/predictions.json')
    groups=(None,) if args.pilot else ('neutral','unmasked')
    methods={}
    protocols={}
    statuses={}
    diagnostics={}
    for label,parent in [('CF200',args.baseline.resolve())]+[(f'lambda-{v}',root/f'lambda-{v}') for v in ('0.1','0.2','0.3')]:
        methods[label]=[]
        statuses[label]=Counter()
        diagnostics[label]={}
        for cohort in groups:
            folder=parent if cohort is None else parent/cohort
            protocol,rows,status=util.validate_run(folder,'CF')
            protocols[label+'/'+str(cohort)]=protocol
            methods[label].extend(rows)
            statuses[label].update(status)
            if label=='CF200':
                continue
            weight=float(label.split('-')[1])
            if protocol['force_mixing']['weight']!=weight:
                raise ValueError('Mixed-force coefficient differs from requested branch.')
            physics=load(folder/'force-mixing-audit.json')
            proof=load(folder/'physics-gradient-probe.json')
            if not proof['passed'] or physics['weight']!=weight or not 1<=physics['gradient_calls']<=200:
                raise ValueError('Native force mixing budget/probe was not verified.')
            trace=[json.loads(line) for line in (folder/'force-trace.jsonl').read_text().splitlines()]
            if [r['A_step'] for r in trace]!=list(range(1,physics['gradient_calls']+1)) or any(r['weight']!=weight for r in trace):
                raise ValueError('Missing or different mixed-force steps.')
            cosines=[c['coordinate_force_cosine'] for row in trace for c in row['cases'] if c['coordinate_force_cosine'] is not None]
            ratios=[c['lambda_F1_over_F2'] for row in trace for c in row['cases'] if c['lambda_F1_over_F2'] is not None]
            diagnostics[label][str(cohort)]=dict(gradient_calls=physics['gradient_calls'],
                coordinate_force_opposition_fraction=sum(v<0 for v in cosines)/len(cosines) if cosines else None,
                median_lambda_F1_over_F2=statistics.median(ratios) if ratios else None,
                maximum_pre_B_raw_force=max(r['max_raw_force'] for r in physics['endpoint']))
    ids=protocols['CF200/'+str(groups[0])]['ids'] if args.pilot else [r['candidate_id'] for r in load(args.source/'inventory-249/inventory.json')['candidates']]
    cases={label:[r for r in raw if r['stage']==stage and r['candidate_id'] in ids] for label,stage in [('MatterGen_raw','before'),('MatterGen_B','after')]}
    for method,rows in methods.items():
        for label,stage in (('A','before'),('A_B','after')):
            selected=[r for r in rows if r['stage']==stage]
            if len(selected)!=len(ids) or {r['candidate_id'] for r in selected}!=set(ids):
                raise ValueError('Method does not retain original IDs exactly once.')
            cases[method+'_'+label]=selected
    result=dict(ids=ids,denominator=len(ids),metrics={k:util.summarize(v) for k,v in cases.items()},cases=cases,
        protocols=protocols,B_status={k:dict(v) for k,v in statuses.items()},force_diagnostics=diagnostics,
        paired_to_CF200={method:util.paired(cases['CF200_A_B'],cases[method+'_A_B'],ids) for method in methods if method!='CF200'},
        paired_to_direct_B={method:util.paired(cases['MatterGen_B'],cases[method+'_A_B'],ids) for method in methods},
        scope='Raw original fixed-species MatterGen starts; F2 + lambda F1 BEFORE Adam200 cosine. '
              'F2 is raw negative total MatterSim energy gradient, F1 negative original CF gap+Ef loss gradient. '
              'Coordinates and cell stress pulled back consistently; no force/RMS/peratom rescaling. Same final B, frozen CF+ALIGNN. No SUN/DFT/MD.')
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Force-mixing comparison','',result['scope'],'',
        '| Method | CF gap | CF Ef | CF joint | CF MAE/eV | AL gap | AL Ef | AL joint | AL MAE/eV | Unknown CF/AL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,m in result['metrics'].items():
        cf,al=m['crystalformer'],m['alignn'];n=m['denominator']
        def value(v):return 'unknown' if v is None else f'{v:.6f}'
        lines.append(f"| {name} | {cf['gap_pass']}/{n} | {cf['eform_pass']}/{n} | {cf['joint_pass']}/{n} | {value(cf['MAE'])} | {al['gap_pass']}/{n} | {al['eform_pass']}/{n} | {al['joint_pass']}/{n} | {value(al['MAE'])} | {cf['unknown']}/{al['unknown']} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(metrics=result['metrics'],B_status=result['B_status'],force_diagnostics=diagnostics)),flush=True)


if __name__=='__main__':main()
