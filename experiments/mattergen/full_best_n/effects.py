"""Audit final Best10 outcomes against same-parent direct B, without reranking."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=128)
def baseline(folder):
    from pymatgen.core import Structure
    record=load(Path(folder)/'result.json');path=Path(folder)/'relaxed.vasp'
    if record['status']!='converged' or sha(path)!=record['relaxed_sha256']:raise ValueError('Baseline geometry changed.')
    return Structure.from_file(path),record


def examine(task):
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from effect_geometry import measure
    case,folder,old=task;row=case['prediction'];b=row['relaxation'];path=Path(b['relaxed_path'])
    if sha(path)!=b['relaxed_sha256'] or row['geometry_sha256']!=b['relaxed_sha256']:raise ValueError('Selected scored pose changed.')
    first,record=baseline(folder)
    if b['checkpoint_sha256']!=record['checkpoint_sha256']:raise ValueError('Energy potentials differ.')
    second=Structure.from_file(path);metrics=measure(first,second)
    matches={}
    for name,ltol,stol,angle,scale,primitive in [('strict',.02,.05,1,False,False),('looser',.1,.2,5,False,False),('scaled_looser',.1,.2,5,True,False),('author',.2,.3,5,True,True)]:
        matcher=StructureMatcher(ltol=ltol,stol=stol,angle_tol=angle,scale=scale,primitive_cell=primitive,attempt_supercell=primitive)
        matches[name]=bool(matcher.fit(second,first))
    if any(matches[k]!=case['matches'][k] for k in ('strict','looser')):raise ValueError('Old endpoint label not reproducible.')
    energy=b['final_energy_eV']/b['num_atoms']-record['final_energy_eV']/record['num_atoms']
    compact=lambda r:{k:v for k,v in r.items() if k in ('crystalformer_gap','crystalformer_eform','alignn_gap','alignn_eform','crystalformer_flags','alignn_flags')}
    return dict(method=case['group'],parent_id=case['parent_id'],candidate_id=case['candidate_id'],baseline_prediction=compact(old),prediction=compact(row),
        baseline_path=str(Path(folder)/'relaxed.vasp'),selected_path=str(path),baseline_sha256=record['relaxed_sha256'],selected_sha256=b['relaxed_sha256'],
        matches=matches,metrics=metrics,energy_delta_eV_atom=energy)


def distribution(values):
    a=np.array(values,dtype=float)
    if not len(a):return None
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),p90=float(np.quantile(a,.9)),max=float(a.max()))


def summarize(rows):
    result=dict(n=len(rows),geometry={k:sum(not c['matches'][k] for c in rows) for k in ('strict','looser','scaled_looser','author')},
        metrics={k:distribution([c['metrics'][k] for c in rows]) for k in ('internal_RMS_A','species_pair_distance_RMS_A','volume_change_percent')},
        energy_delta_eV_atom=distribution([c['energy_delta_eV_atom'] for c in rows]),
        higher_energy_count=sum(c['energy_delta_eV_atom']>1e-3 for c in rows),lower_energy_count=sum(c['energy_delta_eV_atom']< -1e-3 for c in rows))
    for model in ('crystalformer','alignn'):
        known=[c for c in rows if c['prediction'].get(model+'_gap') is not None and c['baseline_prediction'].get(model+'_gap') is not None]
        before=np.array([abs(c['baseline_prediction'][model+'_gap']-2.5) for c in known]);after=np.array([abs(c['prediction'][model+'_gap']-2.5) for c in known]);delta=after-before
        ci=None
        if len(delta):
            rng=np.random.default_rng(42);means=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1);ci=np.quantile(means,[.025,.975]).tolist()
        flag=lambda c,stage:c[stage][model+'_flags']['joint'] is True
        result[model]=dict(n=len(known),MAE_before=float(before.mean()) if len(before) else None,MAE_after=float(after.mean()) if len(after) else None,
            MAE_delta_CI95=ci,MAE_improved_count=int(sum(delta<0)),
            gap_change_ge_0p1_count=sum(abs(c['prediction'][model+'_gap']-c['baseline_prediction'][model+'_gap'])>=.1 for c in known),
            joint_before=sum(flag(c,'baseline_prediction') for c in rows),joint_after=sum(flag(c,'prediction') for c in rows),
            joint_gained=sum(flag(c,'prediction') and not flag(c,'baseline_prediction') for c in rows),joint_lost=sum(flag(c,'baseline_prediction') and not flag(c,'prediction') for c in rows))
    both=lambda c,stage:all(c[stage][m+'_flags']['joint'] is True for m in ('crystalformer','alignn'))
    result['both_joint']=dict(before=sum(both(c,'baseline_prediction') for c in rows),after=sum(both(c,'prediction') for c in rows),
        gained=sum(both(c,'prediction') and not both(c,'baseline_prediction') for c in rows),lost=sum(both(c,'baseline_prediction') and not both(c,'prediction') for c in rows))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=16);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Batch geometry comparisons must use Slurm.')
    source=args.source.resolve();root=args.output.resolve();plan=load(source/'plan.json');records=load(source/'selection-records.json')
    if len(set(plan['parent_ids']))!=128 or plan['N']!=10:raise ValueError('Expected full128 Best10.')
    if not root.is_relative_to((Path.cwd()/'results/mattergen').resolve()):raise ValueError('Use task results.')
    root.mkdir(parents=True,exist_ok=False);original=Path(plan['source'])
    old={r['candidate_id']:r for r in load(original/'cross-prediction/predictions.json') if r['stage']=='after'}
    tasks=[];coverage={}
    selections=records['selections']['force_eform/looser/best_then_classify']
    for method,selected in selections.items():
        by_id={c['candidate_id']:c for c in records['candidates'] if c['group']==method};coverage[method]=dict(selected=0,no_selected=0,baseline_unknown=[])
        if set(selected)!=set(plan['parent_ids']):raise ValueError('Parent coverage differs.')
        for parent in plan['parent_ids']:
            value=selected[parent]
            if value is None:coverage[method]['no_selected']+=1;continue
            coverage[method]['selected']+=1
            case=by_id[value['candidate_id']];folder=original/'relaxation'/Path(parent).stem
            if load(folder/'result.json')['status']!='converged':coverage[method]['baseline_unknown'].append(parent);continue
            tasks.append((case,str(folder),old[parent]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:compared=list(pool.map(examine,tasks,chunksize=2))
    tables={};examples={}
    for method in selections:
        rows=[c for c in compared if c['method']==method]
        if len(rows)!=coverage[method]['selected']-len(coverage[method]['baseline_unknown']):raise ValueError('Paired geometry coverage lost.')
        tables[method]={name:summarize(cs) for name,cs in [('all',rows),('looser_unmatched',[c for c in rows if not c['matches']['looser']]),('author_unmatched',[c for c in rows if not c['matches']['author']])]}
        examples[method]=sorted([c for c in rows if not c['matches']['author']],key=lambda c:c['metrics']['species_pair_distance_RMS_A'],reverse=True)[:5]
    result=dict(source=str(source),source_records_sha256=sha(source/'selection-records.json'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        parent_denominator=128,coverage=coverage,tables=tables,examples=examples,
        interpretation='Frozen final Best10; no novelty-first selection or new optimization. Matcher failure is not proof of another physical basin. Internal fixed-site RMS removes global translation and cell deformation; species-pair distance RMS is permutation invariant but not a complete structural descriptor. Author match allows scaling, primitive cells and supercells. Bootstrap intervals describe selected parents, no multiple-comparison adjustment.')
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (root/'pairs.json').write_text(json.dumps(compared,indent=2,allow_nan=False)+'\n')
    lines=['# Saved-pose structural effects versus direct MatterSim', '',result['interpretation'],'',
        '| Method | Paired | Strict unmatched | Looser unmatched | Scaled-looser unmatched | Author unmatched | Median internal RMS A | Median pair-distance RMS A | CF MAE before -> after | ALIGNN MAE before -> after |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for method,table in tables.items():
        s=table['all'];g=s['geometry'];metrics=s['metrics']
        lines.append(f"| {method} | {s['n']} | {g['strict']} | {g['looser']} | {g['scaled_looser']} | {g['author']} | {metrics['internal_RMS_A']['median']:.4f} | {metrics['species_pair_distance_RMS_A']['median']:.4f} | {s['crystalformer']['MAE_before']:.4f} -> {s['crystalformer']['MAE_after']:.4f} | {s['alignn']['MAE_before']:.4f} -> {s['alignn']['MAE_after']:.4f} |")
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(pairs=len(compared),methods=len(tables),output=str(root))),flush=True)


if __name__=='__main__':main()
