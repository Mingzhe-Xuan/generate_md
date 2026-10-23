"""Compare alternating final iterates with original A200 and A200+B outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--sun',type=Path,required=True)
    p.add_argument('--matched-sun',type=Path,required=True)
    args=p.parse_args()
    state=json.loads((args.run/'results.json').read_text())
    pointers=json.loads((args.run/'evaluation.json').read_text())
    cross=json.loads(Path(pointers['cross_summary']).read_text())
    old=json.loads(Path('results/smoacs/cross-predictor-206/prediction-285/predictions.json').read_text())
    sun=json.loads((args.sun/'candidates.json').read_text())
    old_sun=json.loads(Path('results/smoacs/mp-public-reference/selected-261/assessment-267/summary.json').read_text())
    current={r['candidate_id']:r for r in sun}
    predicted=json.loads(Path(pointers['cross_predictions']).read_text())
    after={r['candidate_id']:r for r in predicted if r['stage']=='after'}
    if set(current)!=set(after) or set(current)!={r['candidate_id'] for r in state['candidates']}:
        raise ValueError('Evaluation IDs differ.')
    denominator=len(current)
    baseline=[]
    for stage,label in [('before','original_A200'),('after','original_A200_then_B')]:
        rows=[r for r in old if r['stage']==stage]
        baseline.append(dict(method=label,denominator=len(rows),
            crystalformer_joint=sum(r['crystalformer_flags']['joint'] is True for r in rows),
            alignn_joint=sum(r['alignn_flags']['joint'] is True for r in rows),
            both_models_joint=sum(r['crystalformer_flags']['joint'] is True and r['alignn_flags']['joint'] is True for r in rows),
            sun_pass=old_sun['sun_pass'] if stage=='after' else None))
    final=dict(method='alternating_final_B',denominator=denominator,
        crystalformer_joint=cross['after_full']['classification']['joint']['crystalformer_pass'],
        alignn_joint=cross['after_full']['classification']['joint']['alignn_pass'],
        both_models_joint=cross['after_full']['classification']['joint']['both_pass'],
        stable_pass=sum(r['stable'] is True for r in sun),sun_pass=sum(r['sun'] is True for r in sun),
        sun_unknown=sum(r['sun'] is None for r in sun))
    jointly=[r['candidate_id'] for r in state['candidates'] if r['status']=='jointly_converged']
    matched=json.loads(Path(pointers['matched_cross_summary']).read_text())['after_full']['classification']['joint']
    baseline.append(dict(method='original_A200_then_matched_B',denominator=denominator,
        crystalformer_joint=matched['crystalformer_pass'],alignn_joint=matched['alignn_pass'],both_models_joint=matched['both_pass'],
        sun_pass=json.loads((args.matched_sun/'summary.json').read_text())['sun_pass']))
    report=dict(methods=baseline+[final],alternating_statuses=dict(Counter(r['status'] for r in state['candidates'])),
        joint_converged_count=len(jointly),joint_converged_sun=sum(current[i]['sun'] is True for i in jointly),
        joint_converged_both_models_property=sum(after[i]['crystalformer_flags']['joint'] is True and after[i]['alignn_flags']['joint'] is True for i in jointly),
        protocol=state['protocol'],sun_summary=json.loads((args.sun/'summary.json').read_text()),
        note='Final-iterate properties/SUN include B-converged physical-cycle/outer-limit cases. They are not claimed jointly converged. '
             'Composition changed; reference phases were reselected for final chemical systems. Original baseline protocol differences remain explicit.')
    (args.run/'final-evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Alternating A/B evaluation','',
        '| Method | Denominator | CF joint property | ALIGNN joint property | Both predictors | SUN confirmed |',
        '|---|---:|---:|---:|---:|---:|']
    for row in report['methods']:
        lines.append('| '+' | '.join(str(row.get(k)) for k in ('method','denominator','crystalformer_joint','alignn_joint','both_models_joint','sun_pass'))+' |')
    lines+=['','Joint convergence statuses: '+json.dumps(report['alternating_statuses']), '',report['note'],'']
    (args.run/'final-report.md').write_text('\n'.join(lines))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
