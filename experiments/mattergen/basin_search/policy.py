"""Frozen sampling, budgets and success criteria for the16-parent search."""
import random
import math

METHODS=('iid','beam','soft_md')
ROUNDS=4
WIDTH=4


def choose_parents(ids,converged,seed=20261026):
    if len(ids)!=128 or len(set(ids))!=128:raise ValueError('Expected complete128-parent inventory.')
    pool=[p for p in ids if p in converged]
    if len(pool)<16:raise ValueError('Need16 known direct-relaxation comparators.')
    chosen=set(random.Random(seed).sample(pool,16))
    return [p for p in ids if p in chosen]


def physical(record,base_energy):
    return (record.get('status')=='converged' and record.get('converged') is True
        and all(record.get(k) is not None and math.isfinite(record[k]) and record[k]<=.01 for k in ('max_filter_force','max_raw_atomic_force'))
        and record.get('separated',False) and .65<=record.get('volume_ratio',1.)<=1.4
        and record['final_energy_eV']/record['num_atoms']<=base_energy+.05)


def search_loss(point):
    return abs(point['gap']-2.5)+4*max(0.,point['eform']+.5)


def best(points,base_energy):
    feasible=[p for p in points if physical(p['B'],base_energy) and p['eform']<-.5]
    return min(feasible,key=lambda p:(abs(p['gap']-2.5),p['B']['final_energy_eV']/p['B']['num_atoms'],p['id'])) if feasible else points[0]


ACCEPTANCE=dict(min_changed_improved_parents=4,min_CF_error_improvement_eV=.1,min_pair_distance_RMS_A=.1,
    require_author_match_failure=True,min_joint_net_gain=3,require_negative_paired_MAE_CI95_or_joint_gain=True,
    final_fmax=.01,final_raw_fmax=.01,CF_eform_upper=-.5,energy_increase_ceiling_eV_atom=.05,
    volume_ratio_bounds=[.65,1.4],
    confirmation_seed=20261027)
