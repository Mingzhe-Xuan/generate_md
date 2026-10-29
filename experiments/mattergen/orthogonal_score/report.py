"""Paired structural change and SUN outcomes after identical physical relaxation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())


def compare(a,b,matcher,strict,measure):
    same_composition=a.composition==b.composition
    row=dict(same_composition=same_composition,same_ordered_types=list(a.atomic_numbers)==list(b.atomic_numbers),
             same_formula=a.composition.reduced_composition==b.composition.reduced_composition,
             formula_a=a.composition.formula,formula_b=b.composition.formula,
             author_match=bool(matcher.fit(a,b)),strict_match=bool(strict.fit(a,b)),volume_change_percent=100*(b.volume/a.volume-1))
    if row['same_ordered_types']:
        row.update(measure(a,b))
    return row


def main(args):
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    root=args.run.resolve();physics=load(root/'physics-complete.json');gen=load(root/'generation-complete.json')
    if not physics['passed'] or not gen['passed']:raise ValueError('Incomplete paired physics run.')
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    strict=StructureMatcher(ltol=.02,stol=.05,angle_tol=1,primitive_cell=False,scale=False,attempt_supercell=False)
    arms={};inventories={};sunrows={}
    for arm in ['baseline','orthogonal']:
        inventory=load(root/arm/'inventory-249/inventory.json');inventories[arm]=inventory['candidates']
        summary=load(root/arm/'SUN/summary.json');sr=load(root/arm/'SUN/candidates.json');sunrows[arm]=sr
        if inventory['denominator']!=16 or summary['denominator']!=16 or len(sr)!=16:raise ValueError('Wrong SUN cohort.')
        B=physics['B'][arm]
        arms[arm]=dict(SUN=summary,relax_converged=sum(r['status']=='converged' for r in B),
            B_failures=[dict(candidate_id=r['candidate_id'],status=r['status'],error=r.get('error')) for r in B if r['status']!='converged'],
            raw_force_converged=sum(r.get('max_raw_atomic_force',float('inf'))<=.01 for r in B),
            sampling=gen['sampling'][arm],B_NFE=sum(r['NFE'] for r in B))
    pairs=[]
    for i in range(16):
        a,b=(inventories[k][i] for k in ['baseline','orthogonal'])
        if a['trajectory']!=i or b['trajectory']!=i:raise ValueError('Unpaired trajectories.')
        if sha(a['source'])!=a['sha256'] or sha(b['source'])!=b['sha256']:raise ValueError('Changed raw structure.')
        row=dict(trajectory=i,before=compare(Structure.from_file(a['source']),Structure.from_file(b['source']),matcher,strict,measure),
                 SUN_baseline=sunrows['baseline'][i],SUN_orthogonal=sunrows['orthogonal'][i])
        ba,bb=(physics['B'][k][i] for k in ['baseline','orthogonal'])
        if ba['status']=='converged' and bb['status']=='converged':
            for point in [ba,bb]:
                if sha(point['relaxed_path'])!=point['relaxed_sha256']:raise ValueError('Changed B terminal.')
            row['after']=compare(Structure.from_file(ba['relaxed_path']),Structure.from_file(bb['relaxed_path']),matcher,strict,measure)
            row['after']['energy_delta_eV_atom']=bb['final_energy_eV']/bb['num_atoms']-ba['final_energy_eV']/ba['num_atoms']
        pairs.append(row)
    structures={}
    for stage in ['before','after']:
        rows=[p[stage] for p in pairs if stage in p]
        rms=[r['internal_RMS_A'] for r in rows if 'internal_RMS_A' in r]
        structures[stage]=dict(paired=len(rows),composition_changed=sum(not r['same_composition'] for r in rows),
            author_unmatched=sum(not r['author_match'] for r in rows),strict_unmatched=sum(not r['strict_match'] for r in rows),
            fixed_identity_RMS_known=len(rms),fixed_identity_RMS_median=float(np.median(rms)) if rms else None,
            fixed_identity_RMS_max=float(max(rms)) if rms else None,
            RMS_at_least_0p1A=sum(r>=.1 for r in rms))
    interventions=[]
    with (root/'orthogonal/interventions.jsonl').open() as f:
        for line in f:
            event=json.loads(line)
            for state in event['states']:
                if state['triggered']:interventions.append(dict(state,phase=event['phase']))
    counts=Counter(r['reason'] or 'projected' for r in interventions)
    output=dict(passed=True,policy=physics['policy'],arms=arms,structures=structures,pairs=pairs,
        intervention=dict(projected=len(interventions),negative_cosine=sum(r['cosine_before']<0 for r in interventions),
            cosine_below_0p7=sum(r['cosine_before']<=.7 for r in interventions),
            norm_ratio_median=float(np.median([r['norm_ratio'] for r in interventions])) if interventions else None),
        SUN_gain=arms['orthogonal']['SUN']['sun_pass']-arms['baseline']['SUN']['sun_pass'],
        generation_seconds=gen['seconds'],total_physics_seconds=physics['seconds'],NFE_total=physics['NFE_total'],
        legacy_baseline_replay=gen['legacy_checks'],
        limitations=['16 paired random trajectories, exploratory and not population-level evidence.',
          'Score projection also changes official corrector adaptive step/noise amplitude; not an isolated direction-only intervention.',
          'Per-arm SUN uniqueness and common reference pool; missing/failed references retain unknown stability/novelty.',
          'StructuralMatcher mismatch alone does not prove a distinct energy basin; RMS is fixed-identity and translation-removed.',
          'MS/MP2020 surrogate SUN, no DFT/phonon/MD validation.'])
    (root/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    lines=['# Available-force orthogonalized MatterGen sampling','',
        'No cosine threshold. Same random seed/priors/noise sequence; both arms then identical MatterSim relaxation.',
        '', '| Method | B converged | S /16 | U /16 | N /16 | SUN /16 | S unknown | N unknown |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,detail in arms.items():
        s=detail['SUN'];lines.append(f'| {arm} | {detail["relax_converged"]} | {s["stable_pass"]} | {s["unique_pass"]} | {s["novel_pass"]} | {s["sun_pass"]} | {16-s["stable_known"]} | {s["novel_unknown"]} |')
    lines+=['','| Stage | Paired | Composition changed | Author unmatched | Strict unmatched | RMS median /A | RMS max /A | RMS >=0.1A |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for stage,s in structures.items():
        lines.append(f'| {stage} | {s["paired"]} | {s["composition_changed"]} | {s["author_unmatched"]} | {s["strict_unmatched"]} | {s["fixed_identity_RMS_median"]} | {s["fixed_identity_RMS_max"]} | {s["RMS_at_least_0p1A"]} |')
    lines+=['',*output['limitations']]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(SUN_gain=output['SUN_gain'],structures=structures,intervention=output['intervention'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);main(p.parse_args())
