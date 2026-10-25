"""Predeclared independent-seed continuation, never a completion certificate."""
import argparse
import json
from pathlib import Path
from policy import ACCEPTANCE


def qualifying(report):
    if report['plan']['acceptance']!=ACCEPTANCE:raise ValueError('Discovery acceptance changed.')
    if report['plan']['parent_count']!=16 or len(set(report['plan']['parent_ids']))!=16:raise ValueError('Discovery cohort changed.')
    return [name for name in ('beam','soft_md') if report['methods'].get(name,{}).get('summary',{}).get('discovery_gate_passed') is True]


def frozen_parents(plan,ids,inventory_sha):
    parents=plan['parent_ids']
    if plan['source_inventory_sha256']!=inventory_sha or len(parents)!=16 or len(set(parents))!=16:raise ValueError('Frozen cohort provenance changed.')
    if parents!=[p for p in ids if p in set(parents)]:raise ValueError('Frozen parents missing or reordered.')
    if plan['acceptance']!=ACCEPTANCE:raise ValueError('Frozen acceptance criteria changed.')
    return parents


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);args=p.parse_args()
    report=json.loads((args.source/'report.json').read_text());methods=qualifying(report)
    result=dict(qualifying_methods=methods,confirmation_seed=ACCEPTANCE['confirmation_seed'],will_run=bool(methods),
        note='A passing discovery triggers all3 independently seeded search strategies; this does not imply final goal success.')
    (args.source/'confirmation-decision.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    raise SystemExit(0 if methods else 3)


if __name__=='__main__':main()
