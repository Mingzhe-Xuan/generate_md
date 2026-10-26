"""Independent estimator identities, causal controls, complete budgets and CLI tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import guided_common
from guided_common import Geometry,utility,common_seed,source_plan,validate_plan
from guided_math import perturbations,estimate,unit_or_zero
from guided_settings import PARENTS,FAMILIES,ARMS,SCALES,SEED,OUTPUTS
from guided_validate import validate_coverage,validate_study
from guided_report import assessment


def geom():return Geometry([11,17],np.eye(3)*5,[[0,0,0],[.5,.5,.5]])


def test_guided_sampler_has_trace_one_covariance_and_no_translation():
    g=geom();size=g.size;projector=np.column_stack([g.project(v) for v in np.eye(size)]);guide=unit_or_zero(g.project(np.arange(size)))
    normal=np.vstack([np.eye(size),np.zeros(size)]);scalar=np.r_[np.zeros(size),1.];sigma=.8
    noise,meta=perturbations(guide,g.project,normal,scalar,sigma,.5)
    covariance=.5/(size-3)*projector+.5*np.outer(guide,guide)
    np.testing.assert_allclose(noise.T@noise/sigma**2,covariance,atol=1e-14)
    assert np.isclose(np.trace(covariance),1.) and meta['effective_dimension']==size-3
    assert all(np.linalg.norm(v[:-6].reshape(g.n,3).mean(0))<1e-14 for v in noise)
    stationary,meta=perturbations(np.zeros(size),g.project,normal,scalar,sigma,.5)
    isotropic,_=perturbations(guide,g.project,normal,scalar,sigma,1.)
    np.testing.assert_allclose(stationary,isotropic);assert meta['stationary_guide'] and meta['alpha']==1.


def test_antithetic_quadratic_estimator_matches_covariance_times_true_gradient():
    g=geom();size=g.size;P=np.column_stack([g.project(v) for v in np.eye(size)]);u=unit_or_zero(g.project(np.arange(size)));C=.5/(size-3)*P+.5*np.outer(u,u)
    eigen,vectors=np.linalg.eigh(C);keep=eigen>1e-12;m=int(sum(keep));sigma=.1;noise=(sigma*np.sqrt(m)*vectors[:,keep]*np.sqrt(eigen[keep])).T
    x=g.project(np.linspace(-.02,.03,size));H=np.diag(np.linspace(.3,1.2,size));value=lambda q:.2+.05*(q@H@q)
    plus=[value(x+d) for d in noise];minus=[value(x-d) for d in noise];actual=estimate(noise,plus,minus,sigma,g.project);true=.1*H@x
    np.testing.assert_allclose(actual,C@true,rtol=1e-11,atol=1e-13);assert actual@true>0
    with pytest.raises(ValueError):estimate(noise,plus[:-1],minus,sigma,g.project)
    with pytest.raises(ValueError):estimate(noise,np.full(m,2.),minus,sigma,g.project)


def candidate(identifier='p'):
    return dict(id=identifier,gap=3.5,eform=-1.,B=dict(status='converged',converged=True,max_filter_force=.005,max_raw_atomic_force=.005,separated=True,
        volume_ratio=1.,volume_A3=125.,final_energy_eV=-5.,num_atoms=2,relaxed_sha256=identifier))


def test_post_quench_utility_respects_all_physical_and_formation_guards():
    q=candidate();assert utility(q,-2.5)==.5 and utility(None,-2.5)==1.
    for key,value in [('max_raw_atomic_force',.1),('volume_ratio',1.5),('final_energy_eV',-4.)]:
        bad=copy.deepcopy(q);bad['B'][key]=value;assert utility(bad,-2.5)==1.
    assert utility(dict(q,eform=0.),-2.5)==1.
    with pytest.raises(ValueError):utility(dict(q,eform=float('nan')),-2.5)


def ledger_fixture():
    shape=geom();size=shape.size;zeros=np.zeros(size).tolist();bases=[candidate(p) for p in PARENTS];operators=[];cases=[];seeds=[];attempts=[];estimates={f:[] for f in FAMILIES};selected={k:[] for k in OUTPUTS}
    for parent,base in zip(PARENTS,bases):
        index=int(parent.split('-')[1].split('.')[0]);qseed,_=common_seed(shape,index,SEED);rng=np.random.default_rng([SEED,index,330]);normal=np.stack([shape.project(v) for v in rng.normal(size=(4,size))]);scalar=rng.normal(size=4)
        checks=[dict(block=b,epsilon=e,passed=True) for b in ('atomic','strain','both') for e in (1e-5,1e-6)]
        operators.append(dict(parent_id=parent,passed=True,geometry=dict(numbers=shape.numbers.tolist(),cell=shape.cell.tolist(),frac=shape.frac.tolist()),
            qseed=qseed.tolist(),normal=normal.tolist(),scalar=scalar.tolist(),cf_guide=zeros,random_guide=zeros,derivative_checks=checks))
        cases.append(dict(parent_id=parent,B=copy.deepcopy(base['B'])))
        seeds.append(dict(id=parent+'-seed',parent_id=parent,kind='seed',q=qseed.tolist(),utility=1.,status='failed'))
        for f in FAMILIES:
            noise=.8/np.sqrt(size-3)*normal
            for k,d in enumerate(noise):
                for side,sign in [('plus',1),('minus',-1)]:attempts.append(dict(id=f'{parent}-{f}-p{k}-{side}',parent_id=parent,family=f,kind='probe',pair=k,side=side,q=(qseed+sign*d).tolist(),utility=1.,status='failed'))
            estimates[f].append(dict(parent_id=parent,noise=noise.tolist(),gradient=zeros,plus=[1.]*4,minus=[1.]*4,
                sampling=dict(alpha=1.,effective_dimension=size-3,guide=zeros,stationary_guide=True),directions={a:zeros for a in ARMS},probe_control=base))
            for a in ARMS:
                for scale in SCALES:attempts.append(dict(id=f'{parent}-{f}-{a}-{scale}',parent_id=parent,family=f,kind='ray',arm=a,scale=scale,q=qseed.tolist(),utility=1.,status='stationary_or_geometric_rejection'))
                selected[f+'-'+a].append(base)
            selected[f+'-probes'].append(base)
        selected['baseline'].append(base)
    return dict(parent_ids=list(PARENTS),cases=cases),dict(passed=True,operators=operators,baseline=bases,NFE=16),dict(completed_parents=16,baseline=bases,seeds=seeds,attempts=attempts,estimates=estimates,selected=selected,NFE_native=16,NFE_search=0,NFE_total=16)


def test_all_attempts_and_failures_keep_1152_budget_and_common_probe_fallback():
    plan,proof,data=ledger_fixture();validate_coverage(plan['parent_ids'],data['attempts']);validate_study(plan,proof,data)
    altered=copy.deepcopy(data);altered['attempts'][-1]=copy.deepcopy(altered['attempts'][0])
    with pytest.raises(ValueError,match='coverage'):validate_study(plan,proof,altered)
    altered=copy.deepcopy(data);altered['estimates']['cf_guided'][0]['gradient'][0]=1.
    with pytest.raises(ValueError,match='gradient'):validate_study(plan,proof,altered)
    altered=copy.deepcopy(data);altered['NFE_search']=1
    with pytest.raises(ValueError,match='cost'):validate_study(plan,proof,altered)
    altered=copy.deepcopy(data);altered['selected']['cf_guided-descent'][0]=candidate('unqueried')
    with pytest.raises(ValueError,match='selection'):validate_study(plan,proof,altered)
    bad=copy.deepcopy(proof);bad['operators'][0]['normal'][0][0]+=1.
    with pytest.raises(ValueError,match='random draws'):validate_study(plan,bad,data)


def error_fixture():
    return {k:np.full(16,1.2) for k in OUTPUTS}


def test_probe_only_sampling_gains_do_not_count_as_gradient_importance():
    errors=error_fixture()
    for f,value in [('cf_guided',.4),('random_guided',.8),('isotropic',.8)]:
        for a in ARMS+('probes',):errors[f+'-'+a]=np.full(16,value)
    result=assessment(errors,np.ones(16,dtype=bool))
    assert result['CF_prior_ablations']['isotropic']['final']['mean_delta']<0
    assert result['CF_prior_ablations']['isotropic']['incremental']['mean_delta']==0
    assert not result['development_CF_gradient_importance'] and not result['families']['cf_guided']['development_directional_evidence']


def test_prior_importance_needs_both_direction_and_both_incremental_controls():
    errors=error_fixture()
    for f in FAMILIES:
        errors[f+'-probes']=np.ones(16)
        for a in ARMS:errors[f+'-'+a]=np.full(16,.9)
    errors['cf_guided-descent']=np.full(16,.4);errors['isotropic-descent']=np.full(16,.8)
    result=assessment(errors,np.ones(16,dtype=bool));assert result['development_CF_gradient_importance'] and not result['goal_completion_proven']
    assert not assessment(errors,np.zeros(16,dtype=bool))['development_CF_gradient_importance']
    errors['isotropic-descent']=np.full(16,.3);assert not assessment(errors,np.ones(16,dtype=bool))['development_CF_gradient_importance']


def test_each_gradient_family_uses_its_own_physical_mask_and_keeps_prior_separate():
    errors=error_fixture()
    for f in FAMILIES:
        errors[f+'-probes']=np.ones(16)
        for a in ARMS:errors[f+'-'+a]=np.full(16,.95)
        errors[f+'-descent']=np.full(16,.4)
    masks={f:np.ones(16,dtype=bool) for f in FAMILIES};masks['cf_guided'][:]=False
    result=assessment(errors,masks)
    assert result['families']['isotropic']['material_parents_beyond_probe_pool']==16
    assert result['families']['isotropic']['development_directional_evidence']
    assert not result['families']['cf_guided']['development_directional_evidence']
    assert not result['development_CF_gradient_importance']


def test_source_plan_is_all_original16_and_immutable(tmp_path,monkeypatch):
    monkeypatch.setattr(guided_common,'code_hashes',lambda:dict(math='frozen'))
    old=dict(parent_ids=list(PARENTS),cases=[dict(parent_id=p,neutral=True,B=dict(status='converged')) for p in PARENTS])
    for filename,obj in [('plan.json',old),('native-probe.json',dict(passed=True,code_sha256=dict(math='frozen'))),('direction-results.json',dict(completed_parents=16)),('report.json',{})]:
        (tmp_path/filename).write_text(json.dumps(obj))
    p=source_plan(tmp_path);validate_plan(p)
    bad=copy.deepcopy(p);bad['policy']=dict(p['policy'],search_seed=99)
    with pytest.raises(ValueError,match='plan changed'):validate_plan(bad)
    (tmp_path/'report.json').write_text('changed')
    with pytest.raises(ValueError,match='evidence changed'):validate_plan(p)


def test_clean_cli_imports_do_not_depend_on_other_model_tests(tmp_path):
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    for name in ['guided_run.py','guided_report.py']:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name(name)),'--help'],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
        assert '--run' in result.stdout
