"""Compare equal-budget fixed-composition RMS amplitudes, retaining parent coverage."""
import argparse
import json
from pathlib import Path

from policy import eligible


AMPLITUDES=(.1,.2,.3)


def summarize(reports):
    if set(reports)!=set(AMPLITUDES):
        raise ValueError('Expected the predeclared 0.10/0.20/0.30 RMS sweep.')
    parents=reports[.1]['parent_ids']
    for amplitude,report in reports.items():
        if report['parent_ids']!=parents or report['N']!=10 or report['amplitudes_A']!=[amplitude]*10:
            raise ValueError('Unequal parent cohort, branch budget or amplitude.')
        if report['max_cf_displacement_A']!=.6 or report['eligibility']!=reports[.1]['eligibility']:
            raise ValueError('Sweep must use identical proposal bounds and eligibility.')
    rows=[]
    common={}
    for arm in ('no_CF','CF20'):
        shared=[p for p in parents if all(reports[a]['selected'][arm][p] is not None for a in AMPLITUDES)]
        common[arm]=dict(parent_ids=shared,MAE={})
        for amplitude in AMPLITUDES:
            report=reports[amplitude]
            candidates=report['candidates'][arm]
            row=dict(amplitude_A=amplitude,arm=arm,summary=report['summaries'][arm],
                eligible_perturbed=sum(eligible(r) for r in candidates.values() if r['branch_index']>0),
                unique_terminal_counts=[report['uniqueness'][arm][p]['distinct_eligible_terminal_structures'] for p in parents],
                zero_summary=report['zero_summaries'][arm])
            rows.append(row)
            common[arm]['MAE'][str(amplitude)]={}
            for model in ('crystalformer','alignn'):
                known=[p for p in shared if all(reports[a]['selected'][arm][p].get(model+'_gap') is not None for a in AMPLITUDES)]
                errors=[abs(report['selected'][arm][p][model+'_gap']-2.5) for p in known]
                common[arm]['MAE'][str(amplitude)][model]=dict(parent_ids=known,MAE=sum(errors)/len(errors) if errors else None)
    return dict(parent_ids=parents,N=10,rows=rows,common_parent_comparisons=common,
        scope='Four-parent pilot; CF selects under force+CF formation-energy gates; ALIGNN held out. No SUN/DFT/MD. '
              'All amplitudes share a0.60A CF parent-displacement bound; previous mixed356 used0.40A and is not the controlled amplitude baseline.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    root=args.run.resolve()
    reports={a:json.loads((root/f'rms-{a:.2f}/comparison.json').read_text()) for a in AMPLITUDES}
    manifests={a:json.loads((root/f'rms-{a:.2f}/manifest.json').read_text()) for a in AMPLITUDES}
    reference=manifests[.1]
    for manifest in manifests.values():
        if manifest['parents']!=reference['parents'] or manifest['source_inventory_sha256']!=reference['source_inventory_sha256']:
            raise ValueError('Sweep raw parents changed.')
        if [(r['parent_id'],r['branch_index']) for r in manifest['candidates']]!=[(r['parent_id'],r['branch_index']) for r in reference['candidates']]:
            raise ValueError('Branch pairing changed.')
    result=summarize(reports)
    result['collision_redraws']={str(a):[{k:r[k] for k in ('candidate_id','draw_seed','draw_attempts')} for r in manifests[a]['candidates'] if r['draw_attempts']>1] for a in AMPLITUDES}
    result['seed_direction_mismatches']=[reference['candidates'][i]['candidate_id'] for i in range(44)
        if len({tuple(manifests[a]['candidates'][i]['draw_seed']) for a in AMPLITUDES})!=1]
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Equal-budget amplitude sweep','',result['scope'],'',
        '| RMS/A | Method | Eligible candidates | Selected parents | CF MAE/eV | ALIGNN MAE/eV | CF gap/joint | ALIGNN gap/joint | Terminal counts |',
        '|---|---|---:|---:|---:|---:|---|---|---|']
    for row in sorted(result['rows'],key=lambda r:(r['amplitude_A'],r['arm'])):
        s=row['summary']
        cf,al=s['crystalformer'],s['alignn']
        def value(x):
            return 'unknown' if x is None else f'{x:.6f}'
        lines.append(f"| {row['amplitude_A']:.2f} | {row['arm']} | {row['eligible_perturbed']}/40 | {s['selected']}/4 | {value(cf['MAE'])} | {value(al['MAE'])} | {cf['gap_pass']}/4, {cf['joint_pass']}/4 | {al['gap_pass']}/4, {al['joint_pass']}/4 | {row['unique_terminal_counts']} |")
    lines+=['','Cross-amplitude seed mismatches after collision resampling: '+json.dumps(result['seed_direction_mismatches']),
        '', 'Compare common-parent metrics in comparison.json when coverage differs. Zero controls are excluded from Best10 and reported separately in each amplitude report.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=result['rows'],seed_direction_mismatches=result['seed_direction_mismatches'])),flush=True)


if __name__=='__main__':
    main()
