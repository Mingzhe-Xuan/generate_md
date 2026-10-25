"""Exploratory Best-of-32 across two frozen N16 searches, retaining16 parents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from policy import ACCEPTANCE,METHODS,best,physical
from report import interval


def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def choose_branch(branches,base_energy):
    for branch in branches:
        point=branch['point'];row=branch['evaluation'];pred=row['prediction']
        if row['point_id']!=point['id'] or pred['geometry_sha256']!=point['B']['relaxed_sha256']:raise ValueError('Selected scored geometry differs.')
        if max(abs(pred['crystalformer_gap']-point['gap']),abs(pred['crystalformer_eform']-point['eform']))>1e-5:raise ValueError('Final independent CF score differs.')
    selected=best([b['point'] for b in branches],base_energy)
    return next(b for b in branches if b['point'] is selected)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Batch geometry must use Slurm.')
    if len(args.run)!=2:p.error('Exactly two complete N16 runs are required.')
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from endpoints import pair_files,plot_pairs
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    runs=[dict(root=root,report=load(root/'report.json'),search=load(root/'search-results.json')) for root in args.run]
    parents=runs[0]['report']['plan']['parent_ids'];plan=runs[0]['report']['plan']
    if len(parents)!=16 or len(set(parents))!=16:raise ValueError('Expected16 parents.')
    if {r['report']['search_seed'] for r in runs}!={20261026,20261027}:raise ValueError('Expected frozen discovery and confirmation seeds.')
    for run in runs:
        other=run['report']['plan']
        if other['parent_ids']!=parents or other['source_inventory_sha256']!=plan['source_inventory_sha256'] or other['acceptance']!=ACCEPTANCE:raise ValueError('Frozen cohorts/criteria differ.')
        if other!=load(run['root']/'plan.json') or [c['parent_id'] for c in other['cases']]!=parents:raise ValueError('Frozen plan/case order differs.')
        if [c['B']['relaxed_sha256'] for c in other['cases']]!=[c['B']['relaxed_sha256'] for c in plan['cases']]:raise ValueError('Direct-relaxation baseline differs.')
        if not load(run['root']/'native-controls.json')['passed'] or not load(run['root']/'search.complete.json')['completed']:raise ValueError('Native run incomplete.')
        if run['search']['search_seed']!=run['report']['search_seed']:raise ValueError('Seed provenance changed.')
        for method in METHODS:
            if [r['parent_id'] for r in run['report']['methods'][method]['cases']]!=parents or run['report']['methods'][method]['summary']['cost']['proposal_attempts']!=256:raise ValueError('N16 budget/denominator changed.')
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    reports={};beam_pairs=[]
    for method in METHODS:
        rows=[]
        for index,case in enumerate(plan['cases']):
            parent=case['parent_id'];baseline=case['B'];base_energy=baseline['final_energy_eV']/baseline['num_atoms']
            branches=[dict(point=r['search']['selected'][method][index],evaluation=r['report']['methods'][method]['cases'][index],seed=r['report']['search_seed']) for r in runs]
            if any(b['evaluation']['baseline_prediction']!=branches[0]['evaluation']['baseline_prediction'] for b in branches):raise ValueError('Paired baseline scores differ.')
            branch=choose_branch(branches,base_energy);point=branch['point'];B=point['B'];row=dict(branch['evaluation'])
            if sha(baseline['relaxed_path'])!=baseline['relaxed_sha256'] or sha(point['pose'])!=B['relaxed_sha256']:raise ValueError('Saved endpoint SHA changed.')
            first=Structure.from_file(baseline['relaxed_path']);second=Structure.from_file(point['pose']);metrics=measure(first,second)
            match=bool(matcher.fit(second,first));valid=physical(B,base_energy) and point['eform']<-.5
            changed=not match and metrics['species_pair_distance_RMS_A']>=ACCEPTANCE['min_pair_distance_RMS_A']
            improvement=abs(row['baseline_prediction']['crystalformer_gap']-2.5)-abs(row['prediction']['crystalformer_gap']-2.5)
            row.update(selected_seed=branch['seed'],physical_eligible=valid,author_match=match,geometry=metrics,material_change=changed,
                material_CF_improvement=bool(valid and changed and improvement>=ACCEPTANCE['min_CF_error_improvement_eV']),CF_error_improvement_eV=improvement,
                neutral_mask=case['neutral'],selected_mechanics=B,selected_path=point['pose'])
            rows.append(row)
            if method=='beam':
                poses=pair_files(baseline['relaxed_path'],baseline['relaxed_sha256'],point['pose'],B['relaxed_sha256'],output/'beam'/Path(parent).stem,
                    [dict(stage='direct',parent_id=parent),dict(stage='pooled32',parent_id=parent,selected_seed=branch['seed'])])
                if row['material_CF_improvement']:beam_pairs.append((poses,row))
        summary=dict(denominator=16,attempts_per_parent=32,material_changed=sum(r['material_change'] for r in rows),material_CF_improved=sum(r['material_CF_improvement'] for r in rows),
            physical_selected=sum(r['physical_eligible'] for r in rows),neutral_material_improved=sum(r['material_CF_improvement'] and r['neutral_mask'] for r in rows))
        for model in ('crystalformer','alignn'):
            known=[r for r in rows if r['prediction'].get(model+'_gap') is not None]
            before=np.array([abs(r['baseline_prediction'][model+'_gap']-2.5) for r in known]);after=np.array([abs(r['prediction'][model+'_gap']-2.5) for r in known])
            summary[model]=dict(known=len(known),MAE_before=float(before.mean()) if len(before) else None,MAE_after=float(after.mean()) if len(after) else None,
                delta_CI95=interval(after-before) if len(after) else None,joint_before=sum(r['baseline_prediction'][model+'_flags']['joint'] and r['baseline_physical_eligible'] for r in rows),
                joint_after=sum(r['prediction'][model+'_flags']['joint'] and r['physical_eligible'] for r in rows))
        costs=[r['report']['methods'][method]['summary']['cost'] for r in runs]
        summary['cost']={key:sum(c[key] for c in costs) for key in costs[0]}
        reports[method]=dict(summary=summary,cases=rows)
    plot_pairs(beam_pairs,output/'pooled32-beam-material-comparison.png')
    result=dict(parent_ids=parents,search_seeds=[r['report']['search_seed'] for r in runs],methods=reports,
        interpretation='Exploratory pooling of two complete16-attempt searches. Exactly16 pairedparents,32 attempts perparent, noALIGNNranking. Does not replace two separate endpoint confirmations or constitute a third independent seed. NoDFT/SUN/phonon certification.')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({m:r['summary'] for m,r in reports.items()}),flush=True)


if __name__=='__main__':main()
