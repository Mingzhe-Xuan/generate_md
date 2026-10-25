"""Join frozen property scores with S/U/N, preserving unknown reference coverage."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
    load=lambda p:json.loads(p.read_text())
    manifest=load(root/'manifest.json');summary=load(root/'SUN/summary.json');rows=load(root/'SUN/candidates.json')
    by_id={r['candidate_id']:r for r in rows}
    if len(rows)!=12 or set(by_id)!={c['candidate_id'] for c in manifest['candidates']} or summary['denominator']!=12:raise ValueError('SUN cohort changed.')
    joined=[]
    for candidate in manifest['candidates']:
        row=by_id[candidate['candidate_id']];flags=[row['stable'],row['unique'],row['novel']]
        expected=False if False in flags else True if all(x is True for x in flags) else None
        if row['sun'] is not expected:raise ValueError('Incorrect three-state SUN conjunction.')
        if row['stable'] is True and (not row['reference_coverage_complete'] or row['hull_eV_atom']>=.1):raise ValueError('Uncertified stability.')
        joined.append(dict(parent_id=candidate['parent_id'],branch_index=candidate['branch_index'],**row,
            CF_gap=candidate['prediction']['crystalformer_gap'],CF_Ef=candidate['prediction']['crystalformer_eform'],
            ALIGNN_joint=candidate['prediction']['alignn_flags']['joint']))
    counts={k:dict(passed=sum(r[k] is True for r in rows),failed=sum(r[k] is False for r in rows),unknown=sum(r[k] is None for r in rows)) for k in ('stable','unique','novel','sun')}
    result=dict(conditional_denominator=12,original_parent_denominator=128,counts=counts,
        confirmed_CF_joint_SUN_count=counts['sun']['passed'],confirmed_CF_joint_SUN_yield=counts['sun']['passed']/128,
        candidates=joined,protocol=manifest,assessment_summary=summary,
        scope='SUN conditional on12 preselected CFjoint candidates; U within12; cached MatterSim5M+MP2020 hull<0.1, no DFT or phonon claim. Incomplete reference coverage remains unknown.')
    (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# CF-joint12: noise0.30 + CF20 + MatterSim SUN','',result['scope'],'',
        '| Parent | Branch | CF gap | Hull eV/atom | Reference coverage | S | U | N | SUN |',
        '|---|---:|---:|---:|---|---|---|---|---|']
    value=lambda x:'unknown' if x is None else 'pass' if x is True else 'fail'
    for row in joined:
        hull='unknown' if row['hull_eV_atom'] is None else f"{row['hull_eV_atom']:.6f}"
        lines.append(f"| {row['parent_id']} | {row['branch_index']} | {row['CF_gap']:.6f} | {hull} | {row.get('available_references',0)}/{row.get('expected_references',0)} | "+' | '.join(value(row[k]) for k in ('stable','unique','novel','sun'))+' |')
    lines+=['',json.dumps(counts),'',f"Confirmed CF-joint AND SUN: {counts['sun']['passed']}/128 original parents; conditional {counts['sun']['passed']}/12. Unknowns are not counted as certified passes."]
    (root/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(counts=counts,denominator=12)),flush=True)


if __name__=='__main__':main()
