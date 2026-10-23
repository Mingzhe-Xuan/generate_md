"""Audited full128 denominators, frozen predictor rates and one shared SUN pool."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adam_cycles'))
from adam_policy import rate

ARMS={'single':1,'alternating':3,'restart200':1}
LABELS={'single':'A600 (single cosine) -> B','alternating':'(A200 -> B) x3','restart200':'A600 (200-step cosine restarts) -> B'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows,ids):
    result={r['candidate_id']:r for r in rows}
    if len(rows)!=len(ids) or len(result)!=len(ids) or set(result)!=set(ids):raise ValueError('Missing, duplicate or different candidate IDs.')
    return result


def conjunction(values):
    return False if False in values else None if None in values else True


def property_stats(rows,ids):
    selected=indexed(rows,ids);n=len(ids);output={}
    for model in ('crystalformer','alignn'):
        for metric in ('gap','eform','joint'):
            output[model+'_'+metric]=rate([selected[i][model+'_flags'][metric] for i in ids],n)
    output['structure_valid']=rate([selected[i].get('structure_valid') for i in ids],n)
    output['both_models_joint']=rate([conjunction([selected[i][m+'_flags']['joint'] for m in ('crystalformer','alignn')]) for i in ids],n)
    return output


def sun_stats(rows,properties,ids):
    sun=indexed(rows,ids);pred=indexed(properties,ids);n=len(ids)
    result={m:rate([sun[i][m] for i in ids],n) for m in ('stable','unique','novel','sun')}
    for model in ('crystalformer','alignn'):
        result['sun_and_'+model+'_joint']=rate([conjunction([sun[i]['sun'],pred[i][model+'_flags']['joint']]) for i in ids],n)
    return result


def load_arms(root):
    result={};baseline=None;initial=None
    for arm,cycle in ARMS.items():
        run=root/arm;protocol=json.loads((run/'protocol.json').read_text());ids=protocol['ids']
        if len(ids)!=128 or len(set(ids))!=128 or protocol['denominator']!=128:raise ValueError('Require original full128 cohort.')
        if not json.loads((run/'complete.json').read_text())['completed']:raise ValueError('Incomplete optimization.')
        audit=json.loads((run/'gpu-audit.json').read_text())
        if not audit['passed'] or not audit['all_cuda']:raise ValueError('CUDA optimization audit failed.')
        invariant=('ids','baseline_npz_sha256','author_source_sha256','a_device','a_dtype','a_base_rates','temperature',
                   'gaussian_derivative','b_fmax','b_maxstep','b_max_steps','gradient_normalization','microbatch_size')
        if baseline is None:baseline=protocol
        elif any(protocol[k]!=baseline[k] for k in invariant):raise ValueError('Comparison protocols differ.')
        expected=(200,3,'per_phase') if arm=='alternating' else (600,1,'restart200' if arm=='restart200' else 'per_phase')
        if (protocol['steps_per_A'],protocol['cycles'],protocol['lr_cycle_policy'])!=expected:raise ValueError('Wrong arm schedule.')
        hashes={i:digest(run/'initial'/Path(i).stem/'structure.vasp') for i in ids}
        if initial is None:initial=hashes
        elif initial!=hashes:raise ValueError('Original initial geometries differ between arms.')
        endpoint=run/('evaluation-cycle'+str(cycle))
        inventory=json.loads((endpoint/'inventory-249/inventory.json').read_text());indexed(inventory['candidates'],ids)
        predictions=json.loads((endpoint/'cross-prediction/predictions.json').read_text())
        after=[r for r in predictions if r['stage']=='after'];before=[r for r in predictions if r['stage']=='before']
        indexed(after,ids);indexed(before,ids)
        for row in after:
            candidate=next(c for c in inventory['candidates'] if c['candidate_id']==row['candidate_id'])
            if row['input_sha256']!=candidate['sha256']:raise ValueError('Prediction input digest differs.')
            record=json.loads((endpoint/'relaxation'/Path(row['candidate_id']).stem/'result.json').read_text())
            if record['input_sha256']!=candidate['sha256']:raise ValueError('Relaxation input digest differs.')
            if record['status']=='converged' and row.get('geometry_sha256')!=record['relaxed_sha256']:raise ValueError('Predicted B geometry digest differs.')
        result[arm]=dict(protocol=protocol,endpoint=endpoint,inventory=inventory,before=before,after=after)
    return baseline['ids'],initial,result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--prepare-scope',action='store_true');p.add_argument('--selection',type=Path)
    args=p.parse_args();ids,initial,arms=load_arms(args.root)
    scope=[];stats={}
    for arm,value in arms.items():
        endpoint=value['endpoint'];statuses=[]
        for c in value['inventory']['candidates']:
            record=json.loads((endpoint/'relaxation'/Path(c['candidate_id']).stem/'result.json').read_text());statuses.append(record['status'])
            if record['status']=='converged':scope.append(dict(candidate_id=arm+'_'+c['candidate_id'],elements=c['elements']))
        stats[arm]=dict(label=LABELS[arm],denominator=128,relaxation_status_counts=dict(Counter(statuses)),
            before_last_B=property_stats(value['before'],ids),after_B=property_stats(value['after'],ids),
            protocol=value['protocol'],prediction_sha256=digest(endpoint/'cross-prediction/predictions.json'))
        if args.selection:
            rows=json.loads((endpoint/'SUN/candidates.json').read_text())
            stats[arm]['SUN']=sun_stats(rows,value['after'],ids)
    if args.prepare_scope:
        (args.root/'reference-scope.json').write_text(json.dumps(dict(denominator=len(scope),candidates=scope),indent=2)+'\n')
    if not args.prepare_scope and not args.selection:p.error('Use --prepare-scope or --selection.')
    common=[i for i in ids if all(next(r for r in value['after'] if r['candidate_id']==i)['status']=='completed' for value in arms.values())]
    paired={}
    for arm,value in arms.items():
        rows=indexed(value['after'],ids)
        paired[arm]=dict(properties=property_stats([rows[i] for i in common],common) if common else {},gap_errors={})
        for model in ('crystalformer','alignn'):
            errors=[abs(rows[i][model+'_gap']-2.5) for i in common]
            paired[arm]['gap_errors'][model]=dict(n=len(errors),mean_absolute_error=statistics.mean(errors) if errors else None,
                median_absolute_error=statistics.median(errors) if errors else None,by_id=dict(zip(common,errors)))
    output=dict(denominator=128,candidate_ids=ids,initial_structure_sha256=initial,arms=stats,
        shared_reference_selection=str(args.selection) if args.selection else None,
        shared_reference_inventory_sha256=digest(args.selection/'mattersim-inventory.json') if args.selection else None,
        same_completed_ids=common,same_completed_comparison=paired,
        thresholds=dict(gap_eV=[2.3,2.7],eform_eV_atom_less_than=-.5,stable_hull_eV_atom_less_than=.1),
        note='All128 original templates remain in each denominator. Unknown physical outcomes are explicit; failed B is operational SUN failure. '
        'Joint requires gap, formation energy and structural validity. U deduplicates within each method; N uses the same relaxed MP pool. '
        'Before B means before the final B, not the original templates. Fixed budgets do not establish joint convergence. DFT paused.')
    (args.root/'comparison.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    lines=['# Full128 GPU comparison','', '| Metric after final B | '+ ' | '.join(LABELS[a] for a in ARMS)+' |',
           '|---|'+'---:|'*3]
    metrics=list(stats['single']['after_B'])+(['stable','unique','novel','sun','sun_and_crystalformer_joint','sun_and_alignn_joint'] if args.selection else [])
    for metric in metrics:
        cells=[]
        for arm in ARMS:
            value=stats[arm]['after_B'].get(metric) or stats[arm]['SUN'][metric]
            cells.append(f"{value['pass_count']}/128 ({100*value['confirmed_rate']:.2f}%; unknown {value['unknown_count']})")
        lines.append('| '+metric+' | '+' | '.join(cells)+' |')
    lines+=['','Common fully predicted endpoint cohort: '+str(len(common))+'/128.','',output['note'],'']
    (args.root/'comparison.md').write_text('\n'.join(lines));print(json.dumps(dict(denominator=128,common_completed=len(common),SUN_complete=bool(args.selection))),flush=True)


if __name__=='__main__':main()
