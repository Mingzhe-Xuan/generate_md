"""CF property loss along B, retaining a fixed structure denominator."""
import argparse
import json
from pathlib import Path
import statistics


def monotonicity(values,tolerance=1e-4):
    changes=[b-a for a,b in zip(values,values[1:])]
    return dict(increases=sum(v>tolerance for v in changes),decreases=sum(v < -tolerance for v in changes),
        unchanged=sum(abs(v)<=tolerance for v in changes),nondecreasing=all(v>=-tolerance for v in changes))


def fixed_mae(paths,step):
    return statistics.mean(next(r for r in reversed(path) if r['stage']=='optimizer' and r['B_step']<=step)['gap_absolute_error'] for path in paths)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();root=args.run.resolve()
    data=json.loads((root/'scores.json').read_text());controls=json.loads((root/'replay-controls.json').read_text())
    result=dict(replay_controls_passed=controls['passed'],scorer='Frozen CF',target_gap_eV=2.5,arms={})
    for arm in ('direct','CF200'):
        ids=sorted({r['candidate_id'] for r in data if r['arm']==arm})
        paths=[[r for r in data if r['arm']==arm and r['candidate_id']==key] for key in ids]
        if len(paths)!=4:raise ValueError('Four structures required per branch.')
        cases={}
        for key,path in zip(ids,paths):
            if [r['frame_index'] for r in path]!=list(range(len(path))):raise ValueError('Missing frames.')
            if [r['B_step'] for r in path[1:]]!=list(range(len(path)-1)):raise ValueError('Missing optimizer steps.')
            cases[key]=dict(optimizer_steps=path[-1]['B_step'],input_error=path[0]['gap_absolute_error'],
                after_symmetry_error=path[1]['gap_absolute_error'],final_error=path[-1]['gap_absolute_error'],
                symmetry_error_jump=path[1]['gap_absolute_error']-path[0]['gap_absolute_error'],
                optimizer_error_changes=monotonicity([r['gap_absolute_error'] for r in path[1:]]))
        maximum=max(path[-1]['B_step'] for path in paths)
        curve=[dict(B_step=k,MAE=fixed_mae(paths,k),denominator=4) for k in range(maximum+1)]
        start=statistics.mean(path[0]['gap_absolute_error'] for path in paths)
        end=curve[-1]['MAE'];threshold=start+.8*(end-start)
        result['arms'][arm]=dict(ids=ids,cases=cases,input_MAE=start,post_symmetry_MAE=curve[0]['MAE'],final_MAE=end,
            aggregate_optimizer_changes=monotonicity([r['MAE'] for r in curve]),
            first_step_reaching_80percent_net_loss=next((r['B_step'] for r in curve if r['MAE']>=threshold),None) if end>start else None,
            checkpoints={str(k):fixed_mae(paths,k) for k in (0,1,5,10,20)},fixed_denominator_curve=curve,
            normalized_progress=[dict(fraction=i/100,MAE=statistics.mean(next(r for r in reversed(path) if r['stage']=='optimizer' and r['B_step']<=int(i*path[-1]['B_step']/100))['gap_absolute_error'] for path in paths)) for i in range(101)])
    (root/'path-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{key:value for key,value in v.items() if key not in ('fixed_denominator_curve','normalized_progress')} for k,v in result['arms'].items()},indent=2))


if __name__=='__main__':main()
