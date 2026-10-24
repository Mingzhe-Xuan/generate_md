"""Parent-level Best-of-N under force/CF-formation-energy gates; independent ALIGNN checks."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from policy import best,eligible


def load(path):
    return json.loads(Path(path).read_text())


def selected_summary(rows,denominator):
    known=[r for r in rows if r is not None]
    result=dict(denominator=denominator,selected=len(known),no_eligible=denominator-len(known))
    for model in ('crystalformer','alignn'):
        errors=[abs(r[model+'_gap']-2.5) for r in known if r.get(model+'_gap') is not None]
        result[model]=dict(MAE=sum(errors)/len(errors) if errors else None,
            RMSE=math.sqrt(sum(v*v for v in errors)/len(errors)) if errors else None,
            known=len(errors),unknown=denominator-len(errors),gap_pass=sum(r[model+'_flags']['gap'] is True for r in known),
            eform_pass=sum(r[model+'_flags']['eform'] is True for r in known),
            joint_pass=sum(r[model+'_flags']['joint'] is True for r in known))
    return result


def paired_errors(first,second,parents):
    common=[p for p in parents if first[p] is not None and second[p] is not None]
    result=dict(parent_ids=common,n=len(common),MAE={},known_parent_ids={})
    for model in ('crystalformer','alignn'):
        known=[p for p in common if first[p].get(model+'_gap') is not None and second[p].get(model+'_gap') is not None]
        result['known_parent_ids'][model]=known
        result['MAE'][model]={name:sum(abs(rows[p][model+'_gap']-2.5) for p in known)/len(known) if known else None for name,rows in [('no_CF',first),('CF20',second)]}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    root=args.run.resolve()
    manifest=load(root/'manifest.json')
    parents=manifest['parent_ids']
    if len(parents)!=4 or manifest['N']!=10 or len(manifest['candidates'])!=44:
        raise ValueError('Expected four raw parents and ten paired perturbations plus four zero controls.')
    init=load(root/'cf20/initialization.json')
    if init['branch_manifest_sha256']!=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Paired manifest changed after optimization.')
    protocol=load(root/'cf20/protocol.json')
    if protocol['steps_per_A']!=20 or protocol['lr_period']!=20 or not load(root/'cf20/gpu-audit.json')['passed']:
        raise ValueError('Missing actual twenty-step CUDA optimization.')
    if not load(root/'cf20/fixed-species-audit.json')['passed']:
        raise ValueError('Fixed species were not verified.')
    paths=dict(no_CF=root/'raw-evaluation',CF20=root/'cf20/evaluation-cycle1')
    expected={r['candidate_id']:r for r in manifest['candidates']}
    candidates={}
    metadata={}
    matcher=StructureMatcher(ltol=.02,stol=.05,angle_tol=1,primitive_cell=False,scale=False,attempt_supercell=False)
    uniqueness={}
    for arm,path in paths.items():
        meta=load(path/'cross-prediction/metadata.json')
        if not meta['crystalformer_control']['control_passed'] or len(meta['controls'])<2 or any(max(c['max_errors'])>1e-5 for c in meta['controls']):
            raise ValueError('Frozen predictor controls failed.')
        metadata[arm]=meta
        rows=[r for r in load(path/'cross-prediction/predictions.json') if r['stage']=='after']
        if {r['candidate_id'] for r in rows}!=set(expected) or len(rows)!=44:
            raise ValueError('Both methods must retain all paired branch IDs.')
        candidates[arm]={}
        uniqueness[arm]={}
        for row in rows:
            key=row['candidate_id']
            case=dict(row,**{k:expected[key][k] for k in ('parent_id','branch_index','amplitude_A')})
            case['relaxation']=load(path/'relaxation'/Path(key).stem/'result.json')
            candidates[arm][key]=case
        for parent in parents:
            representatives=[]
            count=0
            for case in candidates[arm].values():
                if case['parent_id']!=parent or case['branch_index']==0 or not eligible(case):
                    continue
                count+=1
                geometry=Structure.from_file(case['relaxation']['relaxed_path'])
                if not any(matcher.fit(geometry,s) for s in representatives):
                    representatives.append(geometry)
            uniqueness[arm][parent]=dict(eligible_branches=count,distinct_eligible_terminal_structures=len(representatives))
    if metadata['no_CF']['checkpoint_zip_sha256']!=metadata['CF20']['checkpoint_zip_sha256']:
        raise ValueError('Paired evaluators used different checkpoints.')
    selections={}
    curves={}
    for prefix in (1,3,5,10):
        chosen={arm:{parent:best([r for r in candidates[arm].values() if r['parent_id']==parent],prefix) for parent in parents} for arm in paths}
        curves[str(prefix)]=dict(summary={arm:selected_summary(list(chosen[arm].values()),4) for arm in paths},
            paired=paired_errors(chosen['no_CF'],chosen['CF20'],parents))
        if prefix==10:
            selections=chosen
    zero={arm:{} for arm in paths}
    for arm in paths:
        for parent in parents:
            row=next(r for r in candidates[arm].values() if r['parent_id']==parent and r['branch_index']==0)
            zero[arm][parent]=row if eligible(row) else None
    for arm in selections:
        for parent,selected in selections[arm].items():
            baseline=next(r for r in candidates['no_CF'].values() if r['parent_id']==parent and r['branch_index']==0)['relaxation']
            if selected is not None and baseline['status']=='converged':
                selected['MatterSim_energy_delta_to_unperturbed_B_eV_atom']=(selected['relaxation']['final_energy_eV']/selected['relaxation']['num_atoms']-
                    baseline['final_energy_eV']/baseline['num_atoms'])
    status={arm:dict(Counter(r['relaxation']['status'] for r in rows.values())) for arm,rows in candidates.items()}
    result=dict(parent_count=4,N=10,parent_ids=parents,primary_selector='CF only: minimize |gap-2.5| after force+CF_Ef eligibility; ALIGNN never used for selection.',
        eligibility=dict(require_optimizer_converged=True,max_filter_force=.01,max_raw_atomic_force=.01,
            CF_eform_less_than_eV_atom=-.5,require_structure_valid=True),
        stability_scope='Local force convergence plus predicted formation-energy requirement only; no hull/SUN,phonons,MD orDFT.',
        summaries={arm:selected_summary(list(rows.values()),4) for arm,rows in selections.items()},
        paired=paired_errors(selections['no_CF'],selections['CF20'],parents),selected=selections,
        zero_controls=zero,zero_summaries={arm:selected_summary(list(rows.values()),4) for arm,rows in zero.items()},
        curves=curves,uniqueness=uniqueness,relaxation_status=status,candidates=candidates,predictor_controls=metadata,
        note='Same ten coordinate perturbations for each raw parent and both methods. 0.05/0.10/0.20A RMS counts2/6/2, seed42, per-site displacement capped2RMS; no initial cell perturbation. '
            'CF20 uses a full20-update cosine schedule, fresh per-branch Adam states, per-candidate gradient normalization1, fixed ordered elements and bounded geometry proposals. '
            'Zero controls are outside Best-of-10. Ineligible branches consume their attempt; parents with no eligible candidate remain explicitly missing. MAE is across selected parents with coverage shown. '
            'Dedup usesltol.02/stol.05/angle1deg/primitiveFalse/scaleFalse, diagnostic only. Four-parent pilot, not population evidence.')
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Paired random-start Best-of-10 pilot','',result['primary_selector'],result['stability_scope'],'',
        '| Method | Eligible parents | CF selected MAE | CF gap pass | AL selected MAE | AL gap pass | AL joint pass |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,summary in [('Raw zero -> B',result['zero_summaries']['no_CF']),('Zero -> CF20 -> B',result['zero_summaries']['CF20']),
                          ('Random -> B, Best10',result['summaries']['no_CF']),('Random -> CF20 -> B, Best10',result['summaries']['CF20'])]:
        def value(x):
            return f'{x:.6f}' if x is not None else 'unknown'
        lines.append(f"| {label} | {summary['selected']}/4 | {value(summary['crystalformer']['MAE'])} | {summary['crystalformer']['gap_pass']}/4 | {value(summary['alignn']['MAE'])} | {summary['alignn']['gap_pass']}/4 | {summary['alignn']['joint_pass']}/4 |")
    lines+=['',result['note'],'', 'Paired same-parent MAE: '+json.dumps(result['paired'])]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summaries=result['summaries'],paired=result['paired'],relaxation_status=status)),flush=True)


if __name__=='__main__':
    main()
