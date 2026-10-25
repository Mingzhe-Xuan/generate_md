"""Full-parent, paired Best-of-N table with geometry novelty and held-out ALIGNN."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

from selection import pick


def load(p):
    return json.loads(Path(p).read_text())


def geometry(task):
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    case,baseline=task
    b=case['prediction']['relaxation'];case['matches']={};case['separated']=False
    if b['status']!='converged':return case
    path=Path(b['relaxed_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=b['relaxed_sha256']:raise ValueError('Terminal geometry changed.')
    if case['prediction']['geometry_sha256']!=b['relaxed_sha256']:raise ValueError('Scored geometry differs.')
    s=Structure.from_file(path)
    case['separated']=len(s.get_neighbor_list(.5-1e-9)[0])==0
    original=Structure.from_file(case['original_path'])
    if list(s.atomic_numbers)!=list(original.atomic_numbers):raise ValueError('Fixed ordered species changed.')
    if baseline is not None:
        parent=Structure.from_file(baseline)
        for name,values in [('strict',(.02,.05,1)),('looser',(.1,.2,5))]:
            matcher=StructureMatcher(ltol=values[0],stol=values[1],angle_tol=values[2],primitive_cell=False,scale=False,attempt_supercell=False)
            case['matches'][name]=bool(matcher.fit(s,parent))
    return case


def summarize(selected,baseline,denominator=128):
    rows=[r for r in selected if r is not None]
    answer=dict(denominator=denominator,selected=len(rows),no_selection=denominator-len(rows))
    for model in ('crystalformer','alignn'):
        known=[r for r in rows if r['prediction'].get(model+'_gap') is not None]
        paired=[r for r in known if baseline.get(r['parent_id'],{}).get(model+'_gap') is not None]
        errors=[abs(r['prediction'][model+'_gap']-2.5) for r in known]
        before=[abs(baseline[r['parent_id']][model+'_gap']-2.5) for r in paired]
        after=[abs(r['prediction'][model+'_gap']-2.5) for r in paired]
        flags={f:sum(r['prediction'][model+'_flags'][f] is True for r in rows) for f in ('gap','eform','joint')}
        answer[model]=dict(known=len(known),unknown=denominator-len(known),MAE=sum(errors)/len(errors) if errors else None,
            RMSE=math.sqrt(sum(x*x for x in errors)/len(errors)) if errors else None,pass_counts=flags,
            paired_count=len(paired),baseline_MAE=sum(before)/len(before) if before else None,
            paired_MAE=sum(after)/len(after) if after else None)
    return answer


def tables(groups,parents,baseline):
    result={};selections={}
    for gate in ('legacy','force_eform'):
        for criterion in ('looser','strict'):
            for ordering in ('best_then_classify','best_among_novel'):
                key=f'{gate}/{criterion}/{ordering}';result[key]=[];selections[key]={}
                for name,cases in groups.items():
                    by_parent={p:[c for c in cases if c['parent_id']==p] for p in parents}
                    chosen={p:pick(cs,gate,criterion if ordering=='best_among_novel' else None,zero=name in ('CF20_only','CF200_reference')) for p,cs in by_parent.items()}
                    novel=[c for c in chosen.values() if c is not None and c['matches'].get(criterion) is False]
                    result[key].append(dict(method=name,selected_summary=summarize(list(chosen.values()),baseline),
                        novel_summary=summarize(novel,baseline),
                        selected_novel_count=len(novel),selected_unknown_geometry=sum(c is not None and criterion not in c['matches'] for c in chosen.values())))
                    selections[key][name]={p:None if c is None else dict(candidate_id=c['candidate_id'],branch_index=c['branch_index'],matches=c['matches']) for p,c in chosen.items()}
    return result,selections


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--workers',type=int,default=16);args=p.parse_args()
    root=args.run.resolve();plan=load(root/'plan.json');source=Path(plan['source']);parents=plan['parent_ids']
    if len(parents)!=128 or len(set(parents))!=128:raise ValueError('Full128 parents required.')
    inventory=source/'inventory-249/inventory.json'
    if hashlib.sha256(inventory.read_bytes()).hexdigest()!=plan['source_inventory_sha256']:raise ValueError('Source inventory changed.')
    baseline={r['candidate_id']:r for r in load(source/'cross-prediction/predictions.json') if r['stage']=='after'}
    baseline_paths={}
    for parent in parents:
        b=load(source/'relaxation'/Path(parent).stem/'result.json')
        baseline_paths[parent]=b['relaxed_path'] if b['status']=='converged' else None
    groups={};tasks=[];seen=[];checkpoint=None
    def add(group,row,parent,branch):
        case=dict(group=group,parent_id=parent,candidate_id=row['candidate_id'],branch_index=branch,prediction=row,original_path=str(source/'structures'/parent))
        tasks.append((case,baseline_paths[parent]))
    for shard in sorted((root/'shards').iterdir()):
        completed=load(shard/'complete.json')
        if not completed['passed']:raise ValueError('Native shard gate failed.')
        ids=load(shard/'parents.json')['parent_ids'];seen+=ids
        for amplitude in ('0.10','0.20','0.30'):
            report=load(shard/f'rms-{amplitude}/comparison.json')
            if report['parent_ids']!=ids:raise ValueError('Shard parent coverage changed.')
            for arm in ('no_CF','CF20'):
                current=report['predictor_controls'][arm]['checkpoint_zip_sha256']
                if checkpoint is not None and current!=checkpoint:raise ValueError('Frozen checkpoints differ.')
                checkpoint=current
                rows=list(report['candidates'][arm].values())
                if len(rows)!=88 or any(sorted(r['branch_index'] for r in rows if r['parent_id']==parent)!=list(range(11)) for parent in ids):raise ValueError('Paired attempt coverage changed.')
                for row in rows:
                    if row['branch_index']>0:add(f'{arm}_noise{amplitude}',row,row['parent_id'],row['branch_index'])
                    elif amplitude=='0.10' and arm=='CF20':add('CF20_only',row,row['parent_id'],0)
    if seen!=parents:raise ValueError('Full parent ordering/coverage differs.')
    for cohort in ('neutral','unmasked'):
        folder=Path('results/mattergen/alignn-full-361/cf200')/cohort/'evaluation-cycle1'
        for row in load(folder/'cross-prediction/predictions.json'):
            if row['stage']!='after':continue
            parent=row['candidate_id'];row['relaxation']=load(folder/'relaxation'/Path(parent).stem/'result.json')
            add('CF200_reference',row,parent,0)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records=list(pool.map(geometry,tasks,chunksize=8))
    for case in records:groups.setdefault(case['group'],[]).append(case)
    if len(groups)!=8 or any(len(cs)!=(128 if key in ('CF20_only','CF200_reference') else 1280) for key,cs in groups.items()):raise ValueError('Full method coverage differs.')
    table,selected=tables(groups,parents,baseline)
    result=dict(parent_count=128,N=10,source_commit=plan['source_commit'],tables=table,
        stability='Force convergence <=0.01 eV/A plus CF formation energy <-0.5 eV/atom; legacy also requires author validity, force_eform requires no distances below0.5A. No SUN/DFT/MD.',
        selection='CF absolute gap error to2.5 only. ALIGNN held out. Zero controls excluded from Best10. CF200 is an existing different-budget reference.',
        baseline_failures=[p for p in parents if baseline_paths[p] is None])
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (root/'selection-records.json').write_text(json.dumps(dict(selections=selected,candidates=records),indent=2,allow_nan=False)+'\n')
    lines=['# Full128 parent-level Best-of-10', '',result['stability'],result['selection']]
    def val(x):return 'unknown' if x is None else f'{x:.4f}'
    for key,rows in table.items():
        lines+=['',f'## {key}','','| Method | Selected /128 | Novel /128 | CF MAE baseline -> selected | ALIGNN MAE baseline -> selected | CF gap/Ef/joint | ALIGNN gap/Ef/joint |','|---|---:|---:|---|---|---|---|']
        for row in rows:
            s=row['novel_summary'];cf=s['crystalformer'];al=s['alignn']
            def flags(model):
                n=s['selected']
                return '; '.join(f"{model['pass_counts'][f]}/{n} ({100*model['pass_counts'][f]/n:.1f}%)" if n else '0/0 (undefined)' for f in ('gap','eform','joint'))
            lines.append(f"| {row['method']} | {row['selected_summary']['selected']} | {s['selected']} | {val(cf['baseline_MAE'])} -> {val(cf['paired_MAE'])} | {val(al['baseline_MAE'])} -> {val(al['paired_MAE'])} | {flags(cf)} | {flags(al)} |")
        lines+=['','Pass counts and paired MAE refer to selected novel terminals only; MAE uses known paired predictions. All parent denominators remain128; see JSON for missing coverage.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(parent_count=128,method_count=len(groups),geometry_count=len(records),table_count=len(table))),flush=True)


if __name__=='__main__':main()
