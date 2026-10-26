"""Preregistered independent-parent validation and sufficient-decrease acceptance."""
from pathlib import Path
import sys
from settings import POLICY
sys.path.append(str(Path(__file__).resolve().parents[1]/'basin_search'))
from policy import best,physical

VALIDATION_POLICY=dict(parent_count=48,pool_count=64,pilot_count=16,search_seed=20261027,
    family='plain_CF_gap_squared_loss',arms=['descent','reverse','random'],scales=[.05,.2,.5,.8],
    primary_selection='retained_acceptance',secondary_selection='ordinary_Best4',minimum_retained_gain_eV=.1,
    acceptance='Primary picks Best4 only when CFerror improves>=.1 beyond BOTH originalB and seedB; otherwise originalB. Same rule for all controls.',
    directional_evidence=dict(min_material_parents=4,min_mean_advantage_vs_each_control_eV=.05,require_negative_parent_paired_CI95_both_controls=True),
    heldout_ALIGNN='Never used to rank or accept; report population and same-pose agreement. Independent physical re-quench required for material examples.',
    baseline_and_physics=POLICY,
    role='Independent48-parent/search-seed validation after exploratory16; pilot outcomes remain separate and unchanged')


def heldout_ids(ids,neutral,converged,pilot):
    if len(ids)!=128 or len(set(ids))!=128:raise ValueError('Require unique128 original inventory.')
    pool=[p for p in ids if p in neutral and p in converged]
    if len(pool)!=64:raise ValueError('Original neutral/converged pool must remain64.')
    if len(pilot)!=16 or len(set(pilot))!=16 or pilot!=[p for p in ids if p in pilot] or not set(pilot).issubset(pool):
        raise ValueError('Original16 pilot changed/not eligible.')
    selected=[p for p in pool if p not in pilot]
    if len(selected)!=48:raise ValueError('Expected all48 remaining parents.')
    return selected


def retained_choice(baseline,seed,proposals,base_energy):
    candidate=best([baseline,seed]+list(proposals),base_energy)
    comparator=min(abs(baseline['gap']-2.5),abs(seed['gap']-2.5))
    retained=comparator-abs(candidate['gap']-2.5)>=.1
    return candidate if retained and physical(candidate['B'],base_energy) and candidate['eform']<-.5 else baseline
