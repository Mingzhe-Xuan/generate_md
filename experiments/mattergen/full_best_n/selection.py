"""Full-parent partition and CF-only stable-candidate selection."""
import math


def partition(ids,size=8):
    if len(ids)!=128 or len(set(ids))!=128 or size!=8:
        raise ValueError('This experiment requires128 unique parents and8-parent shards.')
    return [ids[i:i+size] for i in range(0,len(ids),size)]


def eligible(case,gate='legacy'):
    row=case['prediction'];b=row['relaxation']
    values=[b.get('max_filter_force'),b.get('max_raw_atomic_force'),row.get('crystalformer_gap'),row.get('crystalformer_eform')]
    if b.get('status')!='converged' or not b.get('converged') or any(v is None or not math.isfinite(v) for v in values):return False
    if values[0]>.01 or values[1]>.01 or values[3]>=-.5:return False
    if gate=='legacy':return row.get('structure_valid') is True
    if gate=='force_eform':return case.get('separated') is True
    raise ValueError('Unknown selection gate.')


def pick(cases,gate='legacy',criterion=None,prefix=10,zero=False):
    candidates=[r for r in cases if (r['branch_index']==0 if zero else 1<=r['branch_index']<=prefix)
        and eligible(r,gate) and (criterion is None or r['matches'].get(criterion) is False)]
    if not candidates:return None
    return min(candidates,key=lambda c:(abs(c['prediction']['crystalformer_gap']-2.5),
        c['prediction']['relaxation']['final_energy_eV']/c['prediction']['relaxation']['num_atoms'],c['branch_index']))
