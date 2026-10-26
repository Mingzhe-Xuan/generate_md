"""Coordinate metric, physical pullback and genuine gradient propagation tests."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import sys
import types
from unittest.mock import patch


def module(name):
    spec=importlib.util.spec_from_file_location('directed_'+name,Path(__file__).with_name(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


geometry=module('geometry');directions=module('directions');settings=module('settings')


def reference():
    return geometry.Geometry([14,14,8,8],[[5,0,0],[1.2,5,0],[.3,1.8,6]],[[.1,.2,.3],[.6,.5,.2],[.3,.7,.8],[.8,.2,.7]])


def test_new_neutral_cohort_is_frozen_and_property_blind():
    ids=[str(i) for i in range(128)];neutral=set(ids[::2]);converged=set(ids)-{'4'}
    a=settings.choose(ids,neutral,converged)
    assert len(a)==16 and a==[i for i in ids if i in a] and '4' not in a and all(i in neutral for i in a)
    assert settings.choose(ids,neutral,converged)==a
    with pytest.raises(ValueError):settings.choose(ids[:-1],neutral,converged)


def test_cell_basis_and_translation_projection():
    ref=reference();B=geometry.strain_basis();np.testing.assert_allclose(np.einsum('aij,bij->ab',B,B),np.eye(6),atol=1e-15)
    q=np.arange(ref.size,dtype=float);p=ref.project(q)
    np.testing.assert_allclose(p[:-6].reshape(ref.n,3).mean(0),0,atol=1e-14)
    np.testing.assert_array_equal(p[-6:],q[-6:]);np.testing.assert_allclose(ref.project(p),p)


def test_stress_pullback_at_nonzero_cartesian_offsets():
    ref=reference();rng=np.random.default_rng(2);q=rng.normal(size=ref.size)*.07
    cell,frac=ref.pose(q);cart=frac@cell;k=.7;beta=.2
    force=-k*cart;G_affine=beta*cell+frac.T@(k*cart);stress=cell.T@G_affine/np.linalg.det(cell)
    analytic=ref.energy_gradient(q,force,stress)
    def energy(v):
        h,s=ref.pose(v);return .5*k*np.sum((s@h)**2)+.5*beta*np.sum(h*h)
    eps=1e-5;eye=np.eye(ref.size)*eps
    fd=np.array([(energy(q+v)-energy(q-v))/(2*eps) for v in eye])
    np.testing.assert_allclose(analytic,fd,atol=2e-8,rtol=1e-8)


def test_torch_geometry_matches_numpy_and_chainrule():
    import torch
    ref=reference();q=torch.tensor(np.linspace(-.03,.04,ref.size),dtype=torch.float64,requires_grad=True)
    h,r=ref.torch_pose(q);hn,sn=ref.pose(q.detach().numpy())
    np.testing.assert_allclose(h.detach(),hn,atol=1e-12);np.testing.assert_allclose(r.detach(),sn@hn,atol=1e-12)
    loss=.35*torch.sum(r*r)+.1*torch.sum(h*h);g=torch.autograd.grad(loss,q)[0].numpy()
    F=-.7*(sn@hn);G_aff=.2*hn+sn.T@(-F);stress=hn.T@G_aff/np.linalg.det(hn)
    np.testing.assert_allclose(g,ref.energy_gradient(q.detach().numpy(),F,stress),atol=1e-11)


def test_soft_gradient_and_matched_random_block_norms():
    ref=reference();g=ref.project(np.random.default_rng(4).normal(size=ref.size));H=np.diag(np.linspace(-.1,10,ref.size))
    d,_=directions.soft_descent(g,H,ref.project);assert np.dot(g,d)<0
    r=directions.matched_random(d,np.random.default_rng(5),ref.project)
    for block in (slice(None,-6),slice(-6,None)):assert abs(np.linalg.norm(d[block])-np.linalg.norm(r[block]))<1e-12
    np.testing.assert_allclose(r[:-6].reshape(ref.n,3).mean(0),0,atol=1e-15)


def test_unrolled_hvp_gradient_matches_exact_quadratic_and_fd():
    H=np.array([[2.,.3,0],[.3,.5,.1],[0,.1,1.1]]);C=np.array([[1.,2.,-.5]]);target=np.array([.7]);q=np.array([.2,-.1,.4]);eta=.1
    grad=lambda Q:np.asarray(Q)@H.T
    property_gradient=lambda x:C.T@(C@x-target)
    identity=lambda x:np.asarray(x)
    actual,states=directions.unrolled(q,grad,property_gradient,eta,8,identity)
    M=np.linalg.matrix_power(np.eye(3)-eta*H,8);expected=M.T@C.T@(C@(M@q)-target)
    np.testing.assert_allclose(actual,expected,atol=1e-12);np.testing.assert_allclose(states[-1],M@q,atol=1e-12)
    recovered,asym=directions.hessian(grad,q);np.testing.assert_allclose(recovered,H,atol=1e-12);assert asym<1e-12


def test_gradient_importance_requires_sign_controls_and_valid_retention():
    b=np.full(16,2.);g=np.full(16,1.2);r=np.full(16,1.5);reverse=np.full(16,1.7);valid=np.ones(16,dtype=bool)
    assert settings.assess_gradient(g,r,reverse,b,b,valid)['discovery_gradient_advantage']
    assert not settings.assess_gradient(g,np.full(16,1.1),reverse,b,b,valid)['discovery_gradient_advantage']
    assert not settings.assess_gradient(g,r,np.full(16,1.1),b,b,valid)['discovery_gradient_advantage']
    assert not settings.assess_gradient(g,r,reverse,b,np.full(16,1.21),valid)['discovery_gradient_advantage']
    assert not settings.assess_gradient(g,r,reverse,b,b,~valid)['discovery_gradient_advantage']


def test_real_attention_adapter_is_scoped_to_backward_entry():
    alias='directed_test.indexed_lattice_multi_head_attention';attention=types.ModuleType(alias);legacy=object();attention.AverageGaussiansBySoftmaxXYFunc_=legacy
    layer=type('TestAttention',(),{'__module__':alias})()
    class FakeScorer:
        def __init__(self):self.models=[types.SimpleNamespace(modules=lambda:[layer])];self.metadata={}
        def predict(self,path):
            assert attention.AverageGaussiansBySoftmaxXYFunc_ is legacy
            return (2.5,-1.)
    with patch.dict(sys.modules,{'scorer':types.SimpleNamespace(Scorer=FakeScorer),alias:attention}):adapter=module('cf_gradient');scorer=adapter.GradientScorer()
    assert scorer.predict(None)==(2.5,-1.)
    with scorer.operators(True):assert attention.AverageGaussiansBySoftmaxXYFunc_ is scorer.gaussian
    assert attention.AverageGaussiansBySoftmaxXYFunc_ is legacy
    with pytest.raises(RuntimeError):
        with scorer.operators(True):raise RuntimeError('test cleanup')
    assert attention.AverageGaussiansBySoftmaxXYFunc_ is legacy


def test_existing_gaussian_input_derivative_matches_finite_difference():
    import torch
    path=Path(__file__).resolve().parents[2]/'smoacs/block_alternating/gaussian.py'
    spec=importlib.util.spec_from_file_location('directed_gaussian_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    x=torch.tensor([[[[.4],[1.2],[2.1]]]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[[[-.3]],[[-.7]]]],dtype=torch.float64,requires_grad=True)
    function=lambda a,b:m.ExactGaussianAverage.apply(3.,4,a,b,-2)
    assert torch.autograd.gradcheck(function,(x,y),eps=1e-6,atol=1e-6,rtol=1e-4)
