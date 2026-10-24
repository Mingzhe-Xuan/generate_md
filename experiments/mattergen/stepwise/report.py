"""Strict paired pilot comparison; no references or new predictor inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def select(rows, ids, stage):
    selected=[r for r in rows if r['stage']==stage]
    indexed={r['candidate_id']:r for r in selected}
    if len(indexed)!=len(selected) or not set(ids)<=set(indexed) or len(ids)!=len(set(ids)):
        raise ValueError('Missing or duplicated comparison IDs.')
    return [indexed[key] for key in ids]


def summarize(rows):
    result={}
    for model in ('crystalformer','alignn'):
        for metric in ('gap','eform','joint'):
            values=[r[model+'_flags'][metric] for r in rows]
            result[model+'_'+metric]=dict(passed=sum(v is True for v in values),unknown=sum(v is None for v in values),denominator=len(rows))
        errors=[r[model+'_gap']-2.5 for r in rows if r[model+'_gap'] is not None]
        result[model+'_gap_MAE']=sum(abs(x) for x in errors)/len(errors) if errors else None
        result[model+'_gap_RMSE']=math.sqrt(sum(x*x for x in errors)/len(errors)) if errors else None
        result[model+'_known']=len(errors)
    result['structure_valid']=dict(passed=sum(r['structure_valid'] is True for r in rows),
        unknown=sum(r['structure_valid'] is None for r in rows),denominator=len(rows))
    return result


def validated_new_arm(root, cycle, ids):
    data=load(root/'results.json')
    protocol=data['protocol']
    if not load(root/'complete.json')['completed'] or not load(root/'gpu-audit.json')['passed']:
        raise ValueError('Incomplete CUDA arm.')
    if protocol['ids']!=ids or protocol['cycles']!=cycle or protocol['steps_per_A']!=(200 if cycle==1 else 1):
        raise ValueError('Incorrect total budget or input cohort.')
    if protocol['lr_period']!=200 or protocol['template_initialization']['species_policy']!='fixed_per_site_onehot':
        raise ValueError('Incorrect fixed-species or learning-rate protocol.')
    if protocol['lr_cycle_policy']!=('global' if cycle==200 else 'per_phase') or protocol['total_A_budget']!=200:
        raise ValueError('Cosine schedule does not span exactly200 total updates.')
    if not load(root/'fixed-species-audit.json')['passed']:
        raise ValueError('Fixed-species audit missing.')
    completed={key:sum(r['a_updates'] for c in data['cycles'] for r in c['candidates'] if r['candidate_id']==key) for key in ids}
    if any(v>200 for v in completed.values()):
        raise ValueError('A budget exceeded.')
    if [c['cycle'] for c in data['cycles']]!=list(range(1,cycle+1)):
        raise ValueError('Missing or duplicate A/B cycles.')
    alive={key:True for key in ids}
    previous={}
    for c in data['cycles']:
        if [r['candidate_id'] for r in c['candidates']]!=ids:
            raise ValueError('A/B cycle cohort changed.')
        for row in c['candidates']:
            key=row['candidate_id']
            if row['a_updates']<0 or row['a_updates']>protocol['steps_per_A']:
                raise ValueError('Invalid per-phase A update count.')
            if not alive[key] and row['a_updates']:
                raise ValueError('A advanced after a nonconverged B.')
            if row['status']=='completed':
                b=row['b_result']
                if b['status']!='converged' or b['max_filter_force']>.01:
                    raise ValueError('A completed endpoint lacks force-converged B.')
                if cycle==200 and c['cycle']>1 and row['preceding_B_sha256']!=previous[key]:
                    raise ValueError('A input is not linked to the preceding converged B.')
                previous[key]=row['B']['sha256']
            else:
                alive[key]=False
    if any(alive[key] and completed[key]!=200 for key in ids):
        raise ValueError('A surviving endpoint did not complete200 updates.')
    evaluation=root/f'evaluation-cycle{cycle}'
    metadata=load(evaluation/'cross-prediction/metadata.json')
    if not metadata['crystalformer_control']['control_passed'] or len(metadata['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in metadata['controls']):
        raise ValueError('Independent predictor controls failed.')
    return data,completed,load(evaluation/'cross-prediction/predictions.json'),metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--variable200',type=Path,default=Path('results/mattergen/cf-full-347'))
    parser.add_argument('--variable600',type=Path,default=Path('results/mattergen/cf-full-340'))
    parser.add_argument('--fixed600',type=Path,default=Path('results/mattergen/cf-fixed-350'))
    args=parser.parse_args()
    root=args.run
    ids=load(root/'block200/protocol.json')['ids']
    if len(ids)!=4:
        raise ValueError('Four property-blind templates required.')
    baseline=load(args.source/'cross-prediction/predictions.json')
    methods=dict(MatterGen_raw=select(baseline,ids,'before'),MatterGen_B=select(baseline,ids,'after'))
    for name,path in [('variable200_B',args.variable200),('variable600_B',args.variable600),('fixed600_B',args.fixed600)]:
        methods[name]=select(load(path/'evaluation/predictions.json'),ids,'after')
    updates={}
    controls={}
    trace=[]
    for name,cycle in [('block200',1),('stepwise200',200)]:
        data,updates[name],rows,controls[name]=validated_new_arm(root/name,cycle,ids)
        methods[name+'_B']=select(rows,ids,'after')
        for c in data['cycles']:
            selected=c['candidates']
            trace.append(dict(method=name,cycle=c['cycle'],CF=[dict(candidate_id=r['candidate_id'],updates=r['a_updates'],
                status=r['status'],gap=r.get('B',{}).get('gap') if r['status']=='completed' else None,
                eform=r.get('B',{}).get('eform') if r['status']=='completed' else None,
                max_filter_force=r.get('b_result',{}).get('max_filter_force')) for r in selected]))
    if controls['block200']['checkpoint_zip_sha256']!=controls['stepwise200']['checkpoint_zip_sha256']:
        raise ValueError('Predictor checkpoints differ between arms.')
    paired={}
    for comparison in [('block200_B','stepwise200_B'),('MatterGen_B','stepwise200_B')]:
        indexed={m:{r['candidate_id']:r for r in methods[m]} for m in comparison}
        paired['__'.join(comparison)]={}
        for model in ('crystalformer','alignn'):
            common=[key for key in ids if all(indexed[m][key][model+'_gap'] is not None for m in comparison)]
            paired['__'.join(comparison)][model]=dict(n=len(common),ids=common,
                MAE={m:sum(abs(indexed[m][key][model+'_gap']-2.5) for key in common)/len(common) if common else None for m in comparison})
    source_sha=hashlib.sha256((args.source/'inventory-249/inventory.json').read_bytes()).hexdigest()
    if any(load(root/m/'initialization.json')['source_inventory_sha256']!=source_sha for m in ('block200','stepwise200')):
        raise ValueError('New arms started from different raw sources.')
    result=dict(denominator=4,ids=ids,metrics={m:summarize(rows) for m,rows in methods.items()},per_case=methods,
        updates=updates,paired_gap_errors=paired,CF_trajectory=trace,predictor_controls=controls,SUN_computed=False,
        note='Fixed original elements;200 total Adam/cosine updates; one force-converged B after each A versus matched A200/oneB. '
        'Same four size/ID-selected raw templates, mean loss over four. Old full-cohort baselines had separate65/63 mean losses. '
        'Failures remain in denominator; four-case feasibility only. No SUN/DFT. B convergence does not imply CF stationarity.')
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Four-case one-A/one-converged-B pilot','',result['note'],'',
        '| Method | CF gap | CF Ef | CF joint | CF MAE | AL gap | AL Ef | AL joint | AL MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in result['metrics'].items():
        cells=[]
        for model in ('crystalformer','alignn'):
            for metric in ('gap','eform','joint'):
                v=m[model+'_'+metric]
                cells.append(f"{v['passed']}/4 (unknown {v['unknown']})")
            value=m[model+'_gap_MAE']
            cells.append(f'{value:.6f}' if value is not None else 'unknown')
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(denominator=4,updates=updates,paired_gap_errors=paired)),flush=True)


if __name__=='__main__':
    main()
