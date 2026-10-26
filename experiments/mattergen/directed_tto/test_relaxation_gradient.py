"""Post-quench derivative identity and separation of sign fitting from validation."""
import copy
import numpy as np
import pytest
from relaxation_gradient import oracle_loss,box_gradient,calibrated_sign,axis_case


def point(identifier,gap):
    return dict(id=identifier,gap=gap,eform=-1.,B=dict(status='converged',converged=True,max_filter_force=.005,max_raw_atomic_force=.005,
        separated=True,volume_ratio=1.,volume_A3=100.,final_energy_eV=-5.,num_atoms=2))


def test_box_smoothed_derivative_crosses_basins_when_local_relaxation_is_flat():
    raw=lambda x:1+.1*x-.1*x**3+(x*x-4)**2
    relax=lambda x:-2. if x<-.5 else 2. if x>.5 else 0.
    loss=lambda x:raw(relax(x))**2
    radius=.8
    def integral(s):
        left=s-radius;right=s+radius
        weights=np.array([max(0.,min(right,-.5)-left),max(0.,min(right,.5)-max(left,-.5)),max(0.,right-max(left,.5))])
        return weights@np.array([loss(-2),loss(0),loss(2)])/(2*radius)
    derivative=box_gradient(loss(radius),loss(-radius),radius)
    eps=1e-6;fd=(integral(eps)-integral(-eps))/(2*eps)
    assert np.isclose(derivative,fd,rtol=1e-8,atol=1e-8) and derivative<0
    assert loss(eps)==loss(-eps) and raw(eps)>raw(-eps)
    assert loss(1.2)<loss(-1.2)


def test_oracle_penalizes_force_or_formation_failure_despite_perfect_gap():
    record=dict(point=point('a',2.5));assert oracle_loss(record,-2.5)==0
    bad=copy.deepcopy(record);bad['point']['B']['max_raw_atomic_force']=.1;assert oracle_loss(bad,-2.5)==1
    bad=copy.deepcopy(record);bad['point']['eform']=0.;assert oracle_loss(bad,-2.5)==1
    assert oracle_loss({},-2.5)==1
    with pytest.raises(ValueError):box_gradient(0.,1.,0.)


def rows():
    return [dict(arm=a,scale=s,point=point(a+str(s),gap),status='converged') for a,s,gap in (
        ('descent',.5,3.3),('reverse',.5,2.9),('descent',.8,3.2),('reverse',.8,2.6))]


def test_heldout_scores_cannot_choose_calibration_sign():
    base=point('base',3.5);data=rows();first=axis_case(base,base,data,1)
    assert first['calibrated_sign']==-1 and first['policies']['calibrated']['selected']['id']=='reverse0.8'
    altered=copy.deepcopy(data);altered[-1]['point']['gap']=7.;altered[-2]['point']['gap']=2.5
    second=axis_case(base,base,altered,1)
    assert second['calibrated_sign']==first['calibrated_sign'] and second['smoothed_axis_derivative']==first['smoothed_axis_derivative']
    assert second['policies']['calibrated']['selected']['id']=='reverse0.5'
    assert second['policies']['random_sign']['chosen_sign']==first['policies']['random_sign']['chosen_sign']
    with pytest.raises(ValueError,match='Missing'):axis_case(base,base,data[:-1],1)


def test_stationary_calibration_consumes_attempt_and_keeps_probe_fallback():
    base=point('base',3.5);data=rows();data[0]['point']['gap']=data[1]['point']['gap']=3.
    result=axis_case(base,base,data,1)
    assert result['calibrated_sign']==0
    assert result['policies']['calibrated']['extra_attempt_status']=='stationary_budget_consumed'
    assert result['policies']['calibrated']['selected']==result['query_best']
    assert calibrated_sign(.1,.1)==0 and calibrated_sign(1.,0.)==-1
