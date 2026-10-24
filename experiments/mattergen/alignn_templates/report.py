"""Fair fixed-species ALIGNN200/CF200 comparison with both frozen scorers."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def summarize(rows):
    result=dict(denominator=len(rows))
    for model in ('crystalformer','alignn'):
        errors=[abs(r[model+'_gap']-2.5) for r in rows if r.get(model+'_gap') is not None]
        result[model]=dict(known=len(errors),unknown=len(rows)-len(errors),
            MAE=sum(errors)/len(errors) if errors else None,
            RMSE=math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None,
            **{flag+'_pass':sum(r[model+'_flags'][flag] is True for r in rows) for flag in ('gap','eform','joint')})
    return result


def paired(first,second,ids):
    a={r['candidate_id']:r for r in first}
    b={r['candidate_id']:r for r in second}
    if set(a)!=set(ids) or set(b)!=set(ids):
        raise ValueError('Paired methods must retain every original structure.')
    answer={}
    for model in ('crystalformer','alignn'):
        known=[i for i in ids if a[i].get(model+'_gap') is not None and b[i].get(model+'_gap') is not None]
        answer[model]=dict(ids=known,n=len(known),
            first_MAE=sum(abs(a[i][model+'_gap']-2.5) for i in known)/len(known) if known else None,
            second_MAE=sum(abs(b[i][model+'_gap']-2.5) for i in known)/len(known) if known else None)
    return answer


def validate_run(path,predictor):
    protocol=load(path/'protocol.json')
    if protocol['steps_per_A']!=200 or protocol['lr_period']!=200 or protocol['cycles']!=1:
        raise ValueError('Expected exactly one200-update cosine phase then B.')
    if not load(path/'gpu-audit.json')['passed'] or not load(path/'fixed-species-audit.json')['passed']:
        raise ValueError('Missing native GPU/fixed-species proof.')
    if predictor=='ALIGNN' and (protocol.get('optimization_predictor')!='ALIGNN' or not load(path/'alignn-adapter-audit.json')['passed']):
        raise ValueError('Actual optimization predictor was not ALIGNN.')
    if predictor=='CF' and protocol.get('optimization_predictor','CF')!='CF':
        raise ValueError('CF control has a different optimization predictor.')
    data=load(path/'results.json')
    for row in data['cycles'][0]['candidates']:
        if not 0<=row['a_updates']<=200:
            raise ValueError('Invalid actual A update budget.')
        if row['a_updates']<200 and row.get('reason',{}).get('status')!='A_failed':
            raise ValueError('Missing explicit failure reason for a short A branch.')
    protocol=dict(protocol,actual_A_update_counts=dict(Counter(r['a_updates'] for r in data['cycles'][0]['candidates'])))
    metadata=load(path/'evaluation-cycle1/cross-prediction/metadata.json')
    if not metadata['crystalformer_control']['control_passed'] or len(metadata['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in metadata['controls']):
        raise ValueError('Frozen scorer control failed.')
    return protocol,load(path/'evaluation-cycle1/cross-prediction/predictions.json'),dict(Counter(r.get('b_result',{}).get('status','not_eligible') for r in data['cycles'][0]['candidates']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    root=args.run.resolve()
    source=args.source.resolve()
    inventory=load(source/'inventory-249/inventory.json')
    raw=load(source/'cross-prediction/predictions.json')
    protocols={}
    predictions={}
    statuses={}
    arms=('alignn200',) if args.pilot else ('alignn200','cf200')
    for arm in arms:
        predictions[arm]=[]
        statuses[arm]=Counter()
        for cohort in (None,) if args.pilot else ('neutral','unmasked'):
            path=root/arm if cohort is None else root/arm/cohort
            p,rows,status=validate_run(path,'ALIGNN' if arm=='alignn200' else 'CF')
            protocols[arm+'/'+str(cohort)]=p
            predictions[arm].extend(rows)
            statuses[arm].update(status)
    ids=next(iter(protocols.values()))['ids'] if args.pilot else [r['candidate_id'] for r in inventory['candidates']]
    cases={}
    for name,stage in (('MatterGen_raw','before'),('MatterGen_B','after')):
        cases[name]=[r for r in raw if r['candidate_id'] in ids and r['stage']==stage]
    for arm,rows in predictions.items():
        for stage,label in (('before','A'),('after','A_B')):
            selected=[r for r in rows if r['stage']==stage]
            if len(selected)!=len(ids) or {r['candidate_id'] for r in selected}!=set(ids):
                raise ValueError('Missing or duplicated original structure.')
            cases[arm+'_'+label]=selected
    result=dict(ids=ids,denominator=len(ids),metrics={name:summarize(rows) for name,rows in cases.items()},
        cases=cases,B_status={k:dict(v) for k,v in statuses.items()},protocols=protocols,
        paired_to_direct_B={arm:paired(cases['MatterGen_B'],cases[arm+'_A_B'],ids) for arm in arms},
        scope='Fixed original ordered elements; raw MatterGen starts;200-step Adam cosine; same B profile. '
            'ALIGNN is the optimization model for alignn200 and CF is held-out; roles reverse for cf200. '
            'Both model scores are frozen predictions, not DFT. No SUN/DFT/MD.',
        source_inventory_sha256=hashlib.sha256((source/'inventory-249/inventory.json').read_bytes()).hexdigest())
    if not args.pilot:
        result['paired_CF200_ALIGNN200']=paired(cases['cf200_A_B'],cases['alignn200_A_B'],ids)
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Fixed-element predictor-swap experiment','',result['scope'],'',
        '| Method | CF gap | CF Ef | CF joint | CF MAE/eV | ALIGNN gap | ALIGNN Ef | ALIGNN joint | ALIGNN MAE/eV | Unknown CF/AL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,s in result['metrics'].items():
        cf,al=s['crystalformer'],s['alignn']
        def value(v):
            return 'unknown' if v is None else f'{v:.6f}'
        n=s['denominator']
        lines.append(f"| {name} | {cf['gap_pass']}/{n} | {cf['eform_pass']}/{n} | {cf['joint_pass']}/{n} | {value(cf['MAE'])} | {al['gap_pass']}/{n} | {al['eform_pass']}/{n} | {al['joint_pass']}/{n} | {value(al['MAE'])} | {cf['unknown']}/{al['unknown']} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(metrics=result['metrics'],B_status=result['B_status'])),flush=True)


if __name__=='__main__':
    main()
