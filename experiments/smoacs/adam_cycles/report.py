"""Compare per-property, joint and SUN outcomes on identical four-template IDs."""
import argparse
import json
from pathlib import Path

from adam_policy import paired_flags,rate


def paired_report(first,third,ids):
    if set(first)!=set(ids) or set(third)!=set(ids):raise ValueError('Comparison IDs differ.')
    compared={}
    for model in ('crystalformer','alignn'):
        for metric in ('gap','eform','valid','joint'):
            def value(row):
                return row.get('structure_valid') if metric=='valid' else row[model+'_flags'][metric]
            compared[model+'_'+metric]=paired_flags({i:value(first[i]) for i in ids},{i:value(third[i]) for i in ids},ids)
    return compared


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();data=json.loads((args.run/'results.json').read_text());ids=data['protocol']['ids']
    models={};sun={};stats={}
    for cycle in (1,3):
        root=args.run/('evaluation-cycle'+str(cycle))
        rows=[r for r in json.loads((root/'cross-prediction/predictions.json').read_text()) if r['stage']=='after']
        if len(rows)!=4 or {r['candidate_id'] for r in rows}!=set(ids):raise ValueError('Property evaluation IDs differ.')
        models[cycle]={r['candidate_id']:r for r in rows}
        rows=json.loads((root/'SUN/candidates.json').read_text())
        if len(rows)!=4 or {r['candidate_id'] for r in rows}!=set(ids):raise ValueError('SUN evaluation IDs differ.')
        sun[cycle]={r['candidate_id']:r for r in rows}
        stats[cycle]=dict(properties={},SUN={})
        for model in ('crystalformer','alignn'):
            for metric in ('gap','eform','joint'):
                stats[cycle]['properties'][model+'_'+metric]=rate([models[cycle][i][model+'_flags'][metric] for i in ids],4)
        stats[cycle]['properties']['structure_valid']=rate([models[cycle][i].get('structure_valid') for i in ids],4)
        both=[None if any(models[cycle][i][m+'_flags']['joint'] is None for m in ('crystalformer','alignn')) else
              all(models[cycle][i][m+'_flags']['joint'] for m in ('crystalformer','alignn')) for i in ids]
        stats[cycle]['properties']['both_models_joint']=rate(both,4)
        for metric in ('stable','unique','novel','sun'):
            stats[cycle]['SUN'][metric]=rate([sun[cycle][i][metric] for i in ids],4)
        for model in ('crystalformer','alignn'):
            overlap=[]
            for i in ids:
                values=[sun[cycle][i]['sun'],models[cycle][i][model+'_flags']['joint']]
                overlap.append(False if False in values else None if None in values else True)
            stats[cycle]['SUN']['sun_and_'+model+'_joint']=rate(overlap,4)
    paired=paired_report(models[1],models[3],ids)
    for metric in ('stable','unique','novel','sun'):
        paired[metric]=paired_flags({i:sun[1][i][metric] for i in ids},{i:sun[3][i][metric] for i in ids},ids)
    common=[i for i in ids if models[1][i]['status']=='completed' and models[3][i]['status']=='completed']
    completed_paired=paired_report({i:models[1][i] for i in common},{i:models[3][i] for i in common},common) if common else {}
    numerical={}
    for model in ('crystalformer','alignn'):
        numerical[model]=dict(n=len(common),candidate_ids=common,
            gap_absolute_error_cycle1=[abs(models[1][i][model+'_gap']-2.5) for i in common],
            gap_absolute_error_cycle3=[abs(models[3][i][model+'_gap']-2.5) for i in common],
            eform_cycle1=[models[1][i][model+'_eform'] for i in common],eform_cycle3=[models[3][i][model+'_eform'] for i in common])
    result=dict(denominator=4,protocol=data['protocol'],cycle1=stats[1],cycle3=stats[3],paired=paired,
        same_completed_pair_ids=common,completed_paired=completed_paired,numerical=numerical,
        note='Small pilot, no statistical generalization. Same first-cycle prefix, models, thresholds and union reference pool. '
             'Unknowns and failed relaxation remain in the four-template denominator. Fixed200-step A phases are not convergence claims; DFT paused.')
    (args.run/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Adam200 AB: one versus three cycles','',
        '| Metric | Cycle1 passes/4 (unknown) | Cycle3 passes/4 (unknown) | Confirmed change pp | Possible change pp |',
        '|---|---:|---:|---:|---|']
    for section in ('properties','SUN'):
        for metric,a in stats[1][section].items():
            b=stats[3][section][metric]
            delta=100*(b['confirmed_rate']-a['confirmed_rate'])
            bounds=[100*(b['possible_rate_bounds'][0]-a['possible_rate_bounds'][1]),100*(b['possible_rate_bounds'][1]-a['possible_rate_bounds'][0])]
            lines.append('| '+metric+' | '+str(a['pass_count'])+'/4 ('+str(a['unknown_count'])+') | '+
                         str(b['pass_count'])+'/4 ('+str(b['unknown_count'])+') | '+str(delta)+' | '+str(bounds)+' |')
    lines+=['','Same complete property pairs: '+str(common),'',result['note'],'']
    (args.run/'comparison.md').write_text('\n'.join(lines));print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
