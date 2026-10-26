"""Complete conditional cohorts, seven physical controls and paired unknowns."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from ase import Atoms
import anneal_retention as runner
from anneal_retention_policy import PARENTS,STAGES,POLICY,material_cases,endpoints,validate_trials,eligible,summarize


def source_fixture():
    data=dict(selected={});arms={}
    for method in ('full','zero','seed','reverse','random','fixed'):
        data['selected'][method]=[dict(id=f'{p}-{method}',gap=3.5,eform=-1.) for p in PARENTS]
        arms[method]=dict(cases=[dict(parent_id=p,point_id=q['id'],physical_eligible=True,selection_eligible=True,
            prediction=dict(crystalformer_gap=3.5,alignn_gap=20.)) for p,q in zip(PARENTS,data['selected'][method])])
    for i in (0,3):
        data['selected']['full'][i]['gap']=3.;arms['full']['cases'][i]['prediction']['crystalformer_gap']=3.
    return dict(count=16,parent_ids=list(PARENTS)),data,dict(arms=arms,assessment=dict(material_parents_beyond_zero_bias_pool=2))


def records_fixture():
    case=dict(parent_id=PARENTS[0],index=0,source_gain_eV=.5);records=[];pred={}
    for stage in STAGES:
        for trial in range(4):
            id=f'{stage}-{trial}';gap=3. if stage=='selected' else 3.6
            records.append(dict(candidate_id=id,parent_id=PARENTS[0],stage=stage,trial=trial,status='converged',converged=True,separated=True,
                max_filter_force=.0005,max_raw_atomic_force=.0005,final_energy_eV=-5.,num_atoms=2,original_energy_eV_atom=-2.5,original_volume_ratio=1.))
            pred[id]=dict(crystalformer_gap=gap,crystalformer_eform=-1.,alignn_gap=gap)
    return case,records,pred


def test_all_material_cases_kept_without_ALIGNN_or_global_flag_filter():
    plan,data,report=source_fixture();report['assessment']['development_directional_evidence']=False
    cases=material_cases(plan,data,report);assert [c['index'] for c in cases]==[0,3]
    points=endpoints(data,dict(baseline=[dict(id=p) for p in PARENTS]),cases[0]);assert set(points)==set(STAGES)
    bad=copy.deepcopy(report);bad['assessment']['material_parents_beyond_zero_bias_pool']=1
    with pytest.raises(ValueError,match='count'):material_cases(plan,data,bad)
    bad=copy.deepcopy(plan);bad['count']=1
    with pytest.raises(ValueError,match='pilot'):material_cases(bad,data,report)


def test_seed_only_improvement_does_not_count_beyond_shared_zero_pool():
    plan,data,report=source_fixture();data['selected']['zero'][0]['gap']=2.8;report['arms']['zero']['cases'][0]['prediction']['crystalformer_gap']=2.8
    report['assessment']['material_parents_beyond_zero_bias_pool']=1
    assert [c['index'] for c in material_cases(plan,data,report)]==[3]
    report['arms']['zero']['cases'].reverse()
    with pytest.raises(ValueError,match='coverage'):material_cases(plan,data,report)


def test_every_case_has_all28_unique_trials():
    case,records,_=records_fixture();validate_trials([case],records)
    with pytest.raises(ValueError,match='seven-stage'):validate_trials([case],records[:-1])
    bad=copy.deepcopy(records);bad[-1]['candidate_id']=bad[0]['candidate_id']
    with pytest.raises(ValueError):validate_trials([case],bad)
    with pytest.raises(ValueError):validate_trials([case,case],records)


def test_paired_summary_preserves_failed_unknown_and_original_energy_guards():
    case,records,pred=records_fixture();result=summarize([case],records,pred)[0]
    assert all(result['models']['crystalformer'][s]['all_four_gain_ge_0p1'] for s in STAGES[:-1])
    records[0]['max_raw_atomic_force']=.01;pred['selected-1']['alignn_gap']=None
    result=summarize([case],records,pred)[0]
    assert result['models']['crystalformer']['original']['eligible_known_pairs']==3
    assert not result['models']['crystalformer']['original']['all_four_gain_ge_0p1']
    assert result['models']['alignn']['zero']['known_pairs']==3
    assert not result['models']['alignn']['zero']['all_four_gain_ge_0p1']
    invalid=copy.deepcopy(records[-1]);invalid['original_volume_ratio']=1.5;assert not eligible(invalid)
    invalid=copy.deepcopy(records[-1]);invalid['final_energy_eV']=0.;assert not eligible(invalid)
    pred['extra']={}
    with pytest.raises(ValueError,match='coverage'):summarize([case],records,pred)


def test_zero_case_does_not_initialize_models_and_is_explicit_negative(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    for name in runner.SOURCE_FILES:(source/name).write_text('{}')
    monkeypatch.setattr(runner,'source_evidence',lambda p:({}, {}, {}, dict(assessment={}), []))
    root=tmp_path/'retention';runner.physics(source,root);spec,_,_,_,_,work=runner.specification(root)
    assert work['NFE']==0 and work['records']==[]
    runner.report(root,spec,work);report=json.loads((root/'report.json').read_text())
    assert not report['goal_completion_proven'] and report['cases']==[]
    for model in ('crystalformer','alignn'):assert report['counts'][model]['original_parent_denominator']==16 and report['counts'][model]['material_cases']==0
    (source/'report.json').write_text('changed')
    with pytest.raises(ValueError,match='artifacts'):runner.specification(root)


def test_real_ASE_perturbations_are_paired_and_trial_zero_is_unchanged():
    a=Atoms('Si2',cell=np.eye(3)*5,scaled_positions=[[.2]*3,[.6]*3],pbc=True)
    b=a.copy();b.positions+=np.array([[.01,0,0],[-.01,0,0]])
    assert np.array_equal(runner.perturb(a,0,0).positions,a.positions)
    x=runner.perturb(a,0,1);y=runner.perturb(b,0,1)
    transform=np.linalg.solve(a.cell.array,x.cell.array)
    assert np.allclose(np.linalg.solve(b.cell.array,y.cell.array),transform,atol=1e-12)
    dx=x.positions-a.get_scaled_positions()@x.cell.array;dy=y.positions-b.get_scaled_positions()@y.cell.array
    assert np.allclose(dx,dy,atol=1e-12) and np.isclose(np.sqrt(np.mean(np.sum(dx*dx,axis=1))),.02)


def test_trial_export_preserves_failed_records_and_actual_pose_hashes(tmp_path):
    case,records,_=records_fixture();records=records[:2];root=tmp_path/'root';root.mkdir();pose=tmp_path/'input.vasp';pose.write_text('pose')
    for row in records:row.update(input=str(pose),input_sha256=runner.sha(pose),status='failed')
    class Scorer:
        def predict(self,path):return 3.,-1.
    plan=dict(cases=[dict(parent_id=case['parent_id'],neutral=True,inventory={})])
    runner.export_trials(root,plan,records,Scorer());inventory=runner.load(root/'evaluation-retention/inventory-249/inventory.json')
    assert inventory['denominator']==2 and len(inventory['candidates'])==2
    assert runner.load(root/'evaluation-retention/relaxation/original-0/result.json')['status']=='failed'


def test_fresh_cli_and_Slurm_guard_do_not_need_previous_imports(tmp_path):
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('SLURM_JOB_ID',None);file=str(Path(runner.__file__).resolve())
    result=subprocess.run([sys.executable,file,'--help'],cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    assert '--mode' in result.stdout
    result=subprocess.run([sys.executable,file,'--mode','physics','--run',str(tmp_path/'never')],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==2 and 'Use Slurm' in result.stderr and not (tmp_path/'never').exists()


def test_declared_policy_has_all_physical_and_pairing_controls():
    assert POLICY['fmax']==.001 and POLICY['preserve_symmetry'] is False and POLICY['trials']==4
    assert POLICY['stages']==list(STAGES) and len(STAGES)==7
