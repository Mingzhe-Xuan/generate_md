"""Adaptive state, cache and physical guard controls for multi-round gradient TTO."""
import copy
import numpy as np
import pytest
from feedback import validate_source,validate_feedback_results,direction,proposal_identifier
from settings import POLICY,FAMILIES,ARMS,SCALES
from probe import code_hashes
from geometry import Geometry


def point(identifier,gap=3.):
    return dict(id=identifier,gap=gap,eform=-1.,B=dict(status='converged',converged=True,max_filter_force=.005,max_raw_atomic_force=.005,
        separated=True,volume_ratio=1.,volume_A3=100.,final_energy_eV=-5.,num_atoms=2))


def source_data():
    parents=[f'mattergen-{i:04d}.vasp' for i in range(16)];base=[point(p) for p in parents]
    cases=[]
    for p,b in zip(parents,base):
        archived={k:v for k,v in b['B'].items() if k not in ('volume_A3','volume_ratio')}
        cases.append(dict(parent_id=p,B=archived))
    plan=dict(parent_ids=parents,policy=POLICY,cases=cases)
    attempts=[dict(parent_id=p,family=f,arm=a,scale=k) for p in parents for f in FAMILIES for a in ARMS for k in SCALES]
    data=dict(completed_parents=16,baseline=base,seeds=[dict(parent_id=p,point=b) for p,b in zip(parents,base)],attempts=attempts,
        selected={f+'-'+a:copy.deepcopy(base) for f in FAMILIES for a in ARMS})
    proof=dict(passed=True,code_sha256=code_hashes(),operators=[dict(parent_id=p,passed=True) for p in parents])
    return plan,data,proof


def test_exact_cached_first_round_and_deep_copy():
    plan,data,proof=source_data();selected=validate_source(plan,data,proof)
    selected['descent'][0]['gap']=0.
    assert data['selected']['plain-descent'][0]['gap']==3.
    for change in ('missing','duplicate','choice','proof','failed_case'):
        altered=copy.deepcopy(data);gate=copy.deepcopy(proof)
        if change=='missing':altered['attempts'].pop()
        elif change=='duplicate':altered['attempts'].append(copy.deepcopy(altered['attempts'][0]))
        elif change=='choice':altered['selected']['plain-descent'][0]['gap']=2.5
        elif change=='proof':gate['code_sha256']['geometry.py']='wrong'
        else:gate['operators'][0]['passed']=False
        with pytest.raises(ValueError):validate_source(plan,altered,gate)


def test_adaptive_gradient_direction_sign_and_random_metric():
    geom=Geometry([14,8],np.eye(3)*5.,[[.1,.2,.3],[.6,.7,.8]]);g=np.arange(geom.size,dtype=float)
    d=direction(geom,g,'descent',20261026,1,2);v=direction(geom,g,'reverse',20261026,1,2);r=direction(geom,g,'random',20261026,1,2)
    assert g@d<0 and g@v>0
    np.testing.assert_allclose(d,-v,rtol=0,atol=0)
    for block in (slice(None,-6),slice(-6,None)):assert np.isclose(np.linalg.norm(d[block]),np.linalg.norm(r[block]))
    np.testing.assert_allclose(r[:-6].reshape(2,3).mean(0),0,atol=1e-15)
    assert np.linalg.norm(direction(geom,np.zeros_like(g),'descent',20261026,1,2))==0
    identifiers={proposal_identifier(p,a,r,k) for p in ('p0','p1') for a in ARMS for r in (2,3) for k in (None,0,1,2,3)}
    assert len(identifiers)==60


def feedback_data():
    plan,source,proof=source_data();initial={a:copy.deepcopy(source['baseline']) for a in ARMS};selected=copy.deepcopy(initial)
    attempts=[];seeds=[];history=[]
    for round_number in (2,3):
        for i,parent in enumerate(plan['parent_ids']):
            starts={a:copy.deepcopy(selected[a][i]) for a in ARMS};choices={}
            for arm in ARMS:
                seeds.append(dict(parent_id=parent,round=round_number,arm=arm,original_volume_A3=100.))
                winner=point(proposal_identifier(parent,arm,round_number,0),2.7 if round_number==2 else 2.6)
                for k,scale in enumerate(SCALES):
                    row=dict(parent_id=parent,round=round_number,arm=arm,scale=scale,original_volume_A3=100.)
                    if k==0:row['point']=winner
                    attempts.append(row)
                selected[arm][i]=winner;choices[arm]=copy.deepcopy(winner)
            history.append(dict(parent_id=parent,round=round_number,starts=starts,selected=choices))
    return plan,dict(baseline=source['baseline'],initial=initial,selected=selected,history=history,attempts=attempts,seeds=seeds)


def test_adaptive_ledger_requires_all_parents_rounds_and_zero_steps():
    plan,data=feedback_data();validate_feedback_results(plan,data)
    for field in ('attempts','seeds','history'):
        altered=copy.deepcopy(data);altered[field][-1]=copy.deepcopy(altered[field][0])
        with pytest.raises(ValueError,match='Missing/duplicate'):validate_feedback_results(plan,altered)


def test_adaptive_ledger_rejects_stale_incumbent_and_unphysical_winner():
    plan,data=feedback_data();changed=copy.deepcopy(data);changed['history'][16]['starts']['descent']=copy.deepcopy(data['initial']['descent'][0])
    with pytest.raises(ValueError,match='incumbent'):validate_feedback_results(plan,changed)
    changed=copy.deepcopy(data);changed['attempts'][0]['point']['B']['max_raw_atomic_force']=.1
    with pytest.raises(ValueError,match='physical/CF'):validate_feedback_results(plan,changed)


def test_adaptive_volume_guard_stays_relative_to_original_structure():
    plan,data=feedback_data();changed=copy.deepcopy(data);changed['attempts'][0]['original_volume_A3']=90.
    with pytest.raises(ValueError,match='Original volume'):validate_feedback_results(plan,changed)
    changed=copy.deepcopy(data);changed['attempts'][0]['point']['B']['volume_A3']=80.
    with pytest.raises(ValueError,match='Rolling volume'):validate_feedback_results(plan,changed)
