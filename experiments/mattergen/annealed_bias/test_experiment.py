"""Complete query budgets and zero-CF fallback protect directed TTO attribution."""
import copy
import numpy as np
from anneal_experiment import specifications,choose,seed_components,OUTPUTS
from anneal_report import assessment
from anneal_policy import PARENTS,ARMS


def candidate(id,error,force=.001):
    return dict(id=id,gap=2.5+error,eform=-1.,B=dict(status='converged',converged=True,max_filter_force=force,max_raw_atomic_force=force,separated=True,volume_ratio=1.,final_energy_eV=-10.,num_atoms=2))


def test_integration_and_full_use_identical_control_paths_and_exact_query_counts():
    one=specifications(PARENTS[:1]);allrows=specifications(PARENTS)
    assert len(one)==46 and len(allrows)==736 and len({r['id'] for r in allrows})==736
    for arm in ARMS:assert sum(r.get('arm')==arm for r in one)==9
    assert one==[r for r in allrows if r['parent_id']==PARENTS[0]]


def test_zero_CF_optimizer_improvements_are_shared_with_every_gradient_arm():
    base=candidate('base',1.)
    rows=[dict(kind='seed',point=candidate('seed',.9)),dict(kind='fork',arm='zero',stage=7,point=candidate('zero',.4)),
        dict(kind='fork',arm='full',stage=3,point=candidate('full',.2)),dict(kind='fork',arm='reverse',stage=3,point=candidate('invalid',0.,force=.1))]
    selected=choose(base,rows)
    assert selected['full']['id']==selected['full_stage3']['id']=='full'
    assert selected['full_stage7']['id']==selected['reverse']['id']==selected['random']['id']==selected['fixed']['id']==selected['zero']['id']=='zero'
    assert selected['baseline']['id']=='base' and selected['seed']['id']=='seed'


def test_native_JSON_seed_cache_is_converted_without_modifying_source():
    original=dict(physics_seed=dict(gradient_E_total=[1.,2.],record={}),CF_seed=dict(gradient=[3.,4.],loss=1.))
    before=copy.deepcopy(original);physical,cf=seed_components(original)
    assert original==before and isinstance(physical['gradient_E_total'],np.ndarray) and isinstance(cf['gradient'],np.ndarray)


def test_pilot_cannot_claim_success_and_full_needs_gain_beyond_zero_CF_pool():
    pilot={m:np.zeros(1) for m in OUTPUTS};assert not assessment(pilot,[True],1)['development_directional_evidence']
    values={m:np.ones(16) for m in OUTPUTS};values['zero']=values['full']=np.full(16,.5)
    assert not assessment(values,[True]*16,16)['development_directional_evidence']
    values['full']=np.full(16,.3)
    result=assessment(values,[True]*16,16)
    assert result['development_directional_evidence'] and not result['goal_completion_proven']


def test_full_path_integrates_real_FIRE_bounds_probe_slots_and_telemetry(tmp_path,monkeypatch):
    import anneal_experiment as experiment
    from anneal_shared import Geometry
    from anneal_policy import weights
    class Client:NFE=0
    client=Client()
    class ToyProvider:
        def __init__(self,geom,*args,**kwargs):self.geom=geom;self.calls=0
        def evaluate(self,q,w,arm):
            self.calls+=1;client.NFE+=1
            target=self.geom.project(np.linspace(-.03,.03,len(q)));sign=-1. if arm=='reverse' else 0. if arm=='zero' else 1.
            g=self.geom.project(q+w*sign*(q-target))
            return dict(energy=float(.5*q@q+w*sign*.5*((q-target)@(q-target))),gradient=g,
                raw_physical_fmax=float(np.linalg.norm(q)),CF_gap=2.5,CF_loss=0.,physical_gradient_norm_per_atom=float(np.linalg.norm(q)),
                weighted_bias_gradient_norm=float(w*np.linalg.norm(q-target)),cosine_physical_vs_bias=None)
        def costs(self):return dict(component_requests=self.calls)
    monkeypatch.setattr(experiment,'Provider',ToyProvider)
    geom=Geometry([14,8],np.eye(3)*5,[[.2]*3,[.6]*3]);q=geom.project(np.linspace(-.01,.01,geom.size))
    case=dict(parent_id=PARENTS[0],geometry=dict(numbers=geom.numbers.tolist(),cell=geom.cell.tolist(),frac=geom.frac.tolist()),qseed=q.tolist(),
        physics_seed=dict(gradient_E_total=(2*q).tolist()),CF_seed=dict(gradient=q.tolist(),loss=1.),random_slope=q.tolist())
    result=experiment.trajectory(tmp_path,case,candidate('base',1.),'full',.3,1,None,client)
    assert result['status']=='completed' and len(result['stages'])==8 and set(result['probes'])=={'3','4','7'}
    assert [s['weight'] for s in result['stages']]==weights('full',.3)
    assert all(np.allclose(result['probes'][str(s)],result['stages'][s]['q']) for s in (3,4,7))
    assert result['NFE_steering']==client.NFE==sum(len(s['trace']) for s in result['stages'])
