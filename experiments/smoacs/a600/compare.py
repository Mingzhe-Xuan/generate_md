"""Paired band-gap errors on all four original template IDs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def errors(rows,ids,model,stage):
    selected=[r for r in rows if r['stage']==stage];indexed={r['candidate_id']:r for r in selected}
    if len(selected)!=len(ids) or set(indexed)!=set(ids):raise ValueError('Evaluation IDs differ or duplicate.')
    values={i:None if indexed[i][model+'_gap'] is None else abs(indexed[i][model+'_gap']-2.5) for i in ids}
    return values


def describe(values):
    known=[v for v in values.values() if v is not None]
    return dict(denominator=len(values),known=len(known),unknown=len(values)-len(known),
        mean_absolute_target_error=statistics.mean(known) if known else None,
        median_absolute_target_error=statistics.median(known) if known else None,
        root_mean_square_target_error=math.sqrt(statistics.mean([v*v for v in known])) if known else None,
        bandgap_pass=sum(v<=.2+1e-12 for v in known),by_id=values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('results/smoacs/adam200-AB3-GPU-20261023'))
    args=parser.parse_args();baseline=json.loads((args.baseline/'protocol.json').read_text());ids=baseline['ids']
    arms={'alternating_A200B_x3':args.baseline/'evaluation-cycle3',
          'A600_single_cosine_B':args.root/'single/evaluation-cycle1',
          'A600_restart200_B':args.root/'restart200/evaluation-cycle1'}
    output=dict(target_eV=2.5,ids=ids,arms={},paired={},note='Compare actual B endpoints; unknowns retained, no joint-convergence claim. '
        'Single600cosine differs in LR restarts as well as B frequency; restart200 arm matches the LR sequence.')
    for name,path in arms.items():
        protocol=json.loads((path.parent/'protocol.json').read_text())
        for key in ('ids','baseline_npz_sha256','a_device','a_dtype','temperature','a_base_rates','gaussian_derivative','b_fmax','b_maxstep','b_max_steps'):
            if protocol[key]!=baseline[key]:raise ValueError('Protocol mismatch '+name+': '+key)
        source=path/'cross-prediction/predictions.json';rows=json.loads(source.read_text())
        output['arms'][name]=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            metrics={model:{stage:describe(errors(rows,ids,model,stage)) for stage in ('before','after')} for model in ('crystalformer','alignn')})
    for model in ('crystalformer','alignn'):
        output['paired'][model]={}
        baseline_errors=output['arms']['alternating_A200B_x3']['metrics'][model]['after']['by_id']
        for arm in ('A600_single_cosine_B','A600_restart200_B'):
            other=output['arms'][arm]['metrics'][model]['after']['by_id']
            paired=[i for i in ids if baseline_errors[i] is not None and other[i] is not None]
            differences={i:other[i]-baseline_errors[i] for i in paired}
            output['paired'][model][arm]=dict(ids=paired,denominator=len(paired),
                mean_alternating=statistics.mean([baseline_errors[i] for i in paired]) if paired else None,
                mean_other=statistics.mean([other[i] for i in paired]) if paired else None,
                mean_error_other_minus_alternating=statistics.mean(differences.values()) if paired else None,
                differences=differences,other_better=sum(v<0 for v in differences.values()),alternating_better=sum(v>0 for v in differences.values()))
    (args.root/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# A600B versus interleaving','', '| Method | CF mean absolute gap error | ALIGNN mean absolute gap error | Known CF/ALIGNN |',
           '|---|---:|---:|---|']
    for name,arm in output['arms'].items():
        metrics=arm['metrics'];cf=metrics['crystalformer']['after'];al=metrics['alignn']['after']
        lines.append('| '+name+' | '+str(cf['mean_absolute_target_error'])+' | '+str(al['mean_absolute_target_error'])+' | '+str(cf['known'])+'/'+str(al['known'])+' |')
    lines+=['',output['note'],''];(args.root/'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(output['paired']),flush=True)


if __name__=='__main__':main()
