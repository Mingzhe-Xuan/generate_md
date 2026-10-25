"""Native gate for actual branch budgets, parent coverage and frozen scorers."""
import argparse
import json
from pathlib import Path


def load(p):return json.loads(p.read_text())


def verify(root):
    parents=load(root/'parents.json')['parent_ids']
    if len(parents)!=8:raise ValueError('Eight declared parents required.')
    diagnostics={}
    for amplitude in ('0.10','0.20','0.30'):
        folder=root/f'rms-{amplitude}';m=load(folder/'manifest.json');c=load(folder/'comparison.json')
        if m['parent_ids']!=parents or len(m['candidates'])!=88 or c['parent_ids']!=parents:raise ValueError('Parent/branch coverage changed.')
        protocol=load(folder/'cf20/protocol.json')
        if protocol['steps_per_A']!=20 or protocol['gradient_normalization']!=1 or protocol['a_device']!='cuda':raise ValueError('Wrong optimizer budget/device/denominator.')
        if not load(folder/'cf20/gpu-audit.json')['passed'] or not load(folder/'cf20/fixed-species-audit.json')['passed']:raise ValueError('Native GPU/species gate failed.')
        rows=load(folder/'cf20/results.json')['cycles'][0]['candidates']
        expected_neutral={r['candidate_id']:r['eligible'] for r in load(folder/'cf20/initialization.json')['selected']}
        raw=load(folder/'raw-evaluation/inventory-249/inventory.json')['candidates']
        if {r['candidate_id']:r['original_valid']=='True' for r in raw}!=expected_neutral:raise ValueError('Raw-arm neutrality metadata changed.')
        if len(rows)!=88 or {r['candidate_id'] for r in rows}!=set(expected_neutral):raise ValueError('Actual A coverage differs.')
        if any(r['a_updates']!=20 and r.get('reason',{}).get('status')!='A_failed' for r in rows):raise ValueError('Unexplained short A.')
        for arm in ('no_CF','CF20'):
            if len(c['candidates'][arm])!=88:raise ValueError('Attempt coverage lost.')
            meta=c['predictor_controls'][arm]
            if not meta['crystalformer_control']['control_passed'] or len(meta['controls'])<2 or any(max(x['max_errors'])>1e-5 for x in meta['controls']):raise ValueError('Frozen scorer control failed.')
        diagnostics[amplitude]=dict(parents=8,attempts_per_arm=88,A_update_counts={str(n):sum(r['a_updates']==n for r in rows) for n in {r['a_updates'] for r in rows}},B_status=c['relaxation_status'])
    return dict(passed=True,parent_ids=parents,diagnostics=diagnostics)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    result=verify(args.run.resolve());(args.run/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
