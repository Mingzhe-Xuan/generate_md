"""Causal comparisons require complete budgets and a truly shared seed pool."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from mapping import LatentGeometry
from denoised_math import controlled_directions,validate_coverage,feasible_pose
from denoised_policy import PARENTS,TIMES,SCALES,ARMS,OUTPUTS
from denoised_search import select_parent
from denoised_report import assessment


def test_reverse_and_random_preserve_full_direction_norm_and_atomic_cell_balance():
    g=LatentGeometry([14,8],np.eye(3)*5,[[.2]*3,[.6]*3])
    source=np.arange(g.size,dtype=float)-3
    directions=controlled_directions(source,np.roll(source,3),np.roll(source,1),g.project,np.random.default_rng(32))
    assert set(directions)==set(ARMS)
    assert np.array_equal(directions['full'],-directions['reverse'])
    for section in (slice(None,-6),slice(-6,None)):
        assert np.isclose(np.linalg.norm(directions['full'][section]),np.linalg.norm(directions['random'][section]))
    assert np.isclose(np.linalg.norm(directions['random']),1)
    assert np.allclose(directions['random'][:-6].reshape(2,3).mean(0),0)


def ledger():
    rows=[]
    for p in PARENTS:
        for t in TIMES:
            rows.append(dict(id=f'{p}-{t}-seed',parent_id=p,time=t,kind='seed'))
            rows.extend(dict(id=f'{p}-{t}-{a}-{s}',parent_id=p,time=t,kind='ray',arm=a,scale=s) for a in ARMS for s in SCALES)
    return rows


def test_every_rejected_query_still_consumes_its_fixed_budget():
    rows=ledger()
    assert len(rows)==672
    for row in rows:row['status']='geometric_rejection'
    validate_coverage(rows)
    for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
        with pytest.raises(ValueError):validate_coverage(bad)


def candidate(identifier,error,force=.001):
    return dict(id=identifier,gap=2.5+error,eform=-1.,B=dict(status='converged',converged=True,max_filter_force=force,max_raw_atomic_force=force,
        separated=True,volume_ratio=1.,final_energy_eV=-10.,num_atoms=2))


def test_both_seeds_are_shared_and_other_arm_rays_cannot_leak_into_selection():
    base=candidate('base',1.)
    rows=[dict(kind='seed',point=candidate('seed1',.8)),dict(kind='seed',point=candidate('seed2',.5)),
          dict(kind='ray',arm='reverse',point=candidate('reverse-only',.1)),
          dict(kind='ray',arm='full',point=candidate('invalid-force',0.,force=.1))]
    selected=select_parent(base,rows)
    assert selected['reverse']['id']=='reverse-only'
    assert selected['baseline']['id']=='base'
    assert all(selected[m]['id']=='seed2' for m in ('seeds','full','random','stop','direct'))


def test_denoising_only_improvement_cannot_establish_gradient_direction_value():
    errors={m:np.ones(16) for m in OUTPUTS}
    errors['seeds']=errors['full']=np.full(16,.5)
    result=assessment(errors,np.ones(16,dtype=bool))
    assert result['material_parents_beyond_shared_seed_pool']==0
    assert not result['development_directional_evidence'] and not result['goal_completion_proven']
    positive=copy.deepcopy(errors)
    positive['full']=np.full(16,.3)
    result=assessment(positive,np.ones(16,dtype=bool))
    assert result['development_directional_evidence'] and result['development_decoder_Jacobian_value']
    assert result['unseen_parent_confirmation_required'] and not result['goal_completion_proven']


def test_common_physical_guard_ignores_rigid_translation_rotation_and_rejects_collision():
    cell=np.eye(3)*5;frac=np.array([[.2,.3,.4],[.6,.7,.8]])
    assert feasible_pose([14,8],cell,frac,cell,frac)
    assert feasible_pose([14,8],cell,(frac+.49)%1,cell,frac)
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    assert feasible_pose([14,8],cell@rotation,frac,cell,frac)
    assert not feasible_pose([14,8],cell,np.array([frac[0],frac[0]+.001]),cell,frac)
    assert not feasible_pose([14,8],cell*1.2,frac,cell,frac)


def test_fresh_search_cli_requires_slurm_before_models_or_output(tmp_path):
    env=dict(os.environ);env.pop('SLURM_JOB_ID',None)
    script=Path(__file__).with_name('denoised_search.py')
    target=tmp_path/'not-created'
    result=subprocess.run([sys.executable,str(script),'--source','unused','--native','unused','--run',str(target)],env=env,capture_output=True,text=True)
    assert result.returncode==2 and 'Use Slurm GPU.' in result.stderr and not target.exists()


def test_legacy_policy_adapter_restores_an_existing_unrelated_module():
    script=Path(__file__).with_name('denoised_shared.py').parent.resolve()
    code="""import sys,types
sys.path.insert(0,sys.argv[1])
old=types.ModuleType('policy');old.sentinel=17
sys.modules['policy']=old
before=list(sys.path)
import denoised_shared as s
assert sys.modules['policy'] is old and sys.path==before
assert s.basin_policy.METHODS==('iid','beam','soft_md')
assert s.best is s.basin_policy.best
"""
    result=subprocess.run([sys.executable,'-c',code,str(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_full_validator_keeps_all_parents_when_every_new_attempt_is_rejected(tmp_path,monkeypatch):
    import denoised_search as search
    import denoised_validate as validator
    from denoised_policy import POLICY
    monkeypatch.setattr(search,'core_hashes',lambda:{'core':'fixed'})
    monkeypatch.setattr(search,'validate_native',lambda *args:None)
    monkeypatch.setattr(validator,'sha',lambda p:'fixed')
    bases=[];cases=[]
    for p in PARENTS:
        q=candidate(p,1.);q['pose']='saved/'+p;q['B']['relaxed_sha256']='fixed';bases.append(q)
        cases.append(dict(parent_id=p,B=copy.deepcopy(q['B'])))
    plan=dict(policy=POLICY,parent_ids=list(PARENTS),cases=cases,core_sha256={'core':'fixed'},native_root='native',native_sha256='fixed',source='source')
    monkeypatch.setattr(search,'load',lambda p:{'cases':cases})
    native=dict(passed=True,core_sha256={'core':'fixed'},plan_sha256='fixed',NFE=16,baseline=bases,
        controls=[dict(parent_id=p,CF_error=0.,physical=dict(passed=True)) for p in PARENTS],
        operators=[dict(parent_id=p,time=t,status='invalid_noisy_cell') for p in PARENTS for t in TIMES])
    rows=ledger()
    for row in rows:row['status']='invalid_noisy_cell'
    data=dict(completed_parents=16,attempts=rows,selected={m:copy.deepcopy(bases) for m in OUTPUTS},NFE_native=16,NFE_search=0,NFE_total=16)
    validator.validate(tmp_path,plan,native,data)
    bad=copy.deepcopy(data);bad['selected']['full'].pop()
    with pytest.raises(ValueError):validator.validate(tmp_path,plan,native,bad)
    bad=copy.deepcopy(data);bad['NFE_total']=17
    with pytest.raises(ValueError):validator.validate(tmp_path,plan,native,bad)
