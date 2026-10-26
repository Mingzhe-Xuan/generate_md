"""Optimizer behavior, conservative controls and physical-normalization tests."""
import numpy as np
import pytest
from anneal_policy import weights,PEAKS,FRACTIONS
from anneal_math import combine,bias,random_slope,bounded_project
from anneal_fire import minimize


def identity(q):return np.asarray(q).copy()


def test_schedules_release_full_bias_and_keep_fixed_and_zero_controls_explicit():
    assert weights('full',.3)==[.3*f for f in FRACTIONS]
    assert weights('reverse',.3)==weights('random',.3)==weights('full',.3)
    assert weights('full',.3)[-1]==0 and weights('fixed',.3)==[.3]*8 and weights('zero',.3)==[0.]*8
    with pytest.raises(ValueError):weights('full',2.)
    assert PEAKS==(.1,.3,1.)


def test_random_tilt_is_conservative_and_matches_both_initial_gradient_blocks():
    gradient=np.arange(12,dtype=float)-3
    slope=random_slope(gradient,np.random.default_rng(34),identity)
    for block in (slice(None,-6),slice(-6,None)):
        assert np.isclose(np.linalg.norm(slope[block]),np.linalg.norm(gradient[block]))
    q=np.linspace(-.1,.1,12);anchor=np.zeros(12);direction=np.linspace(.2,.5,12);epsilon=1e-6
    plus=bias(q+epsilon*direction,'random',None,anchor,2.,slope)[0]
    minus=bias(q-epsilon*direction,'random',None,anchor,2.,slope)[0]
    assert np.isclose((plus-minus)/(2*epsilon),slope@direction,rtol=1e-8,atol=1e-8)


def test_actual_joint_objective_normalizes_only_physical_energy_and_has_correct_signs():
    q=np.array([.2,-.1]);anchor=np.zeros(2);slope=np.array([.3,.4]);weight=.3
    def objective(v,arm):
        energy=4.*(v@v);g=8.*v
        cf=dict(loss=.5*((v-1)@(v-1)),gradient=v-1)
        return combine(v,arm,weight,energy,g,4,0.,cf,anchor,1.,slope,identity)
    for arm in ('full','reverse','random','zero','fixed'):
        result=objective(q,arm);direction=np.array([.7,-.4]);eps=1e-6
        fd=(objective(q+eps*direction,arm)['energy']-objective(q-eps*direction,arm)['energy'])/(2*eps)
        assert np.isclose(fd,result['gradient']@direction,rtol=1e-8,atol=1e-8)
    assert np.allclose(objective(q,'full')['gradient']-objective(q,'zero')['gradient'],weight*(q-1))


def test_fire_converges_on_a_coupled_quadratic_without_exceeding_step_budget():
    def objective(q):return dict(energy=float(.5*q@q),gradient=q.copy())
    result=minimize(np.array([1.,-.4]),objective,identity,lambda q:True,dict(steps=1000,fmax=1e-7))
    assert result['converged'] and result['updates']<=1000 and np.linalg.norm(result['q'])<1e-6


def release(well):
    q=np.array([-1.02])
    for strength in weights('full',.3):
        def objective(v):
            x=v[0]
            energy=.1*(x*x-1)**2 if well=='double' else .5*(x+1)**2
            derivative=.4*x*(x*x-1) if well=='double' else x+1
            return dict(energy=energy+strength*.5*(x-1)**2,gradient=np.array([derivative+strength*(x-1)]))
        result=minimize(q,objective,identity,lambda p:abs(p[0])<=2,dict(steps=1000,fmax=1e-6))
        assert result['converged']
        q=result['q']
    return q[0]


def test_bias_can_retain_a_new_basin_but_cannot_invent_one_in_a_single_well():
    assert abs(release('double')-1)<1e-5
    assert abs(release('single')+1)<1e-5


def test_boundary_stall_is_not_reported_as_force_convergence():
    def objective(q):return dict(energy=-float(q[0]),gradient=np.array([-1.]))
    result=minimize(np.array([0.]),objective,lambda q:np.clip(q,-.1,.1),lambda q:True)
    assert result['status']=='stalled_boundary' and not result['converged'] and result['final_gradient_inf']==1


def test_invalid_initial_point_never_calls_the_objective():
    called=[]
    with pytest.raises(ValueError):minimize(np.array([0.]),lambda q:called.append(q),identity,lambda q:False)
    assert not called


def test_bound_projection_preserves_translation_gauge_and_limits_cell_and_atoms():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'directed_tto'))
    from geometry import Geometry
    g=Geometry([14,8],np.eye(3)*5,[[.2]*3,[.6]*3]);q=np.arange(g.size,dtype=float)
    result=bounded_project(g,q)
    assert np.linalg.norm(np.sqrt(g.n)*result[:-6].reshape(g.n,3),axis=1).max()<=2.+1e-12
    assert np.allclose(result[:-6].reshape(g.n,3).mean(0),0)
    strain=np.einsum('a,aij->ij',result[-6:]/g.scale,g.basis)
    assert np.linalg.norm(strain,2)<=.15+1e-12


def test_component_cache_reuses_frozen_values_across_bias_weights_without_mutation():
    from anneal_objective import Provider
    class Geometry:
        n=2
        project=staticmethod(identity)
        def energy_gradient(self,q,force,stress):return 2*np.asarray(q)
    class Client:
        calls=0
        def points(self,geom,Q):
            self.calls+=len(Q)
            return [dict(energy_eV=float(q@q),forces_eV_A=np.zeros((2,3)),stress_eV_A3=np.zeros((3,3))) for q in Q]
    queried=[]
    def cf(geom,q):
        queried.append(q.copy())
        return dict(gap=2.5+float(np.sum(q)),eform=-1.,loss=.5*float((q-1)@(q-1)),gradient=q-1)
    q=np.array([.2,.3]);client=Client();provider=Provider(Geometry(),None,client,0.,q,1.,np.ones(2),cf_callback=cf)
    forward=provider.evaluate(q,.3,'full');reverse=provider.evaluate(q,.1,'reverse');provider.evaluate(q,0.,'zero')
    assert client.calls==1 and len(queried)==1
    assert np.allclose(forward['gradient'],q+.3*(q-1))
    assert np.allclose(reverse['gradient'],q-.1*(q-1))
    assert np.allclose(provider.physics[q.tobytes()]['gradient_E_total'],2*q)
    provider.evaluate(q+.1,.3,'random')
    assert client.calls==2 and len(queried)==1 and provider.costs()['unique_physical_points']==2


def test_fresh_native_cli_cannot_start_models_on_a_login_node(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    env=dict(os.environ);env.pop('SLURM_JOB_ID',None);out=tmp_path/'not-created'
    p=subprocess.run([sys.executable,str(Path(__file__).with_name('anneal_native.py')),'--source','unused','--run',str(out)],env=env,capture_output=True,text=True)
    assert p.returncode==2 and 'Use Slurm GPU.' in p.stderr and not out.exists()
