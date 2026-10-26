"""Baseline changes cannot be disguised as gradient benefit or cache validity."""
import copy
import json
from pathlib import Path
import pytest
from baseline_amendment import DIAGNOSTIC_SHA,validate_diagnostic,cached_replays,fixed_pose_control
from validation_run import PLAN_SHA,OUTPUTS,EXTRA_BASELINE,validate_results
from validation_report import fresh_sensitivity
from probe import sha
from test_validation_runtime import fixture


def diagnostic_fixture():
    parents=[f'p{i}' for i in range(48)]
    cases=[dict(parent_id=p,B=dict(relaxed_sha256='pose-'+p)) for p in parents]
    rows=[dict(parent_id=p,source_sha256='pose-'+p,raw_replay_status='converged',raw_path_reproduced=i!=6,
        fixed_pose_control=dict(passed=True,energy_error_eV_atom=1e-7,raw_fmax_error_eV_A=1e-6)) for i,p in enumerate(parents)]
    return dict(parent_ids=parents,cases=cases),dict(plan_sha256=PLAN_SHA,parents=48,cases=rows,all_fixed_pose_controls_passed=True)


def test_diagnostic_requires_complete_fixed_pose_controls_not_raw_path_identity():
    plan,report=diagnostic_fixture();validate_diagnostic(plan,report,DIAGNOSTIC_SHA)
    for change in ('digest','coverage','force','source'):
        altered=copy.deepcopy(report);digest=DIAGNOSTIC_SHA
        if change=='digest':digest='unpinned'
        if change=='coverage':altered['cases'].pop()
        if change=='force':altered['cases'][6]['fixed_pose_control']['raw_fmax_error_eV_A']=.001
        if change=='source':altered['cases'][6]['source_sha256']='changed'
        with pytest.raises(ValueError):validate_diagnostic(plan,altered,digest)
    case=dict(B=dict(checkpoint_sha256='w',final_energy_eV=-4.,num_atoms=2,max_raw_atomic_force=.005))
    assert fixed_pose_control(case,dict(checkpoint_sha256='w',energy_eV=-4.,forces_eV_A=[[.005,0,0],[0,0,0]]))['passed']
    with pytest.raises(ValueError):fixed_pose_control(case,dict(checkpoint_sha256='w',energy_eV=-4.,forces_eV_A=[[.006,0,0],[0,0,0]]))


def test_cached_rawB_rejects_tampered_tasks_model_profile_and_pose(tmp_path):
    raw=tmp_path/'raw.vasp';pose=tmp_path/'pose.vasp';raw.write_text('raw');pose.write_text('relaxed')
    folder=tmp_path/'baseline-replay'/'p';folder.mkdir(parents=True)
    case=dict(parent_id='p.vasp',raw_path=str(raw),B=dict(input_sha256=sha(raw),checkpoint_sha256='model',num_atoms=2))
    r=dict(status='converged',converged=True,input_sha256=sha(raw),checkpoint_sha256='model',num_atoms=2,maxstep=.05,step_limit=1600,preserve_symmetry=True,
        final_energy_eV=-4.,max_raw_atomic_force=.005,max_filter_force=.005,NFE=7,relaxed_path=str(pose),relaxed_sha256=sha(pose))
    path=folder/'result.json';path.write_text(json.dumps(r));task=dict(id='p.vasp',action='relax',source=str(raw),output=str(folder));(folder/'task.json').write_text(json.dumps(task))
    plan=dict(cases=[case]);d=dict(cases=[dict(raw_replay_record_sha256=sha(path))])
    assert cached_replays(tmp_path,plan,d)==[r]
    for field,value in (('maxstep',.1),('checkpoint_sha256','wrong'),('input_sha256','wrong'),('preserve_symmetry',False)):
        bad=dict(r,**{field:value});path.write_text(json.dumps(bad));proof=dict(cases=[dict(raw_replay_record_sha256=sha(path))])
        with pytest.raises(ValueError,match='profile'):cached_replays(tmp_path,plan,proof)
    path.write_text(json.dumps(r));(folder/'task.json').write_text(json.dumps(dict(task,source='other')))
    with pytest.raises(ValueError,match='task'):cached_replays(tmp_path,plan,d)
    (folder/'task.json').write_text(json.dumps(task));pose.write_text('changed')
    with pytest.raises(ValueError,match='pose'):cached_replays(tmp_path,plan,d)


def test_fresh_rawB_ledger_keeps_all48_and_cached_costs_separate():
    plan,data=fixture()
    for c,b in zip(plan['cases'],data['baseline']):
        c['B'].update(input_sha256='raw-'+c['parent_id'],checkpoint_sha256='w');b['B'].update(c['B'])
    fresh=copy.deepcopy(data['baseline'])
    data.update(baseline_amendment_sha256='amended',raw_replay_baseline=fresh,NFE_cached_replay=100,NFE_native=148,NFE_search=200,NFE_total=348)
    validate_results(plan,data)
    for kind in ('coverage','provenance','cost'):
        bad=copy.deepcopy(data)
        if kind=='coverage':bad['raw_replay_baseline'].pop()
        if kind=='provenance':bad['raw_replay_baseline'][0]['B']['checkpoint_sha256']='wrong'
        if kind=='cost':bad['NFE_total']=248
        with pytest.raises(ValueError):validate_results(plan,bad)


def test_fresh_baseline_sensitivity_retains_pairing_unknowns_and_no_fake_gains():
    ids=[f'p{i}' for i in range(48)]
    def rows(gap):
        return [dict(parent_id=p,prediction=dict(crystalformer_gap=gap,alignn_gap=gap),structure_eligible=True,material_CF_gain=True) for p in ids]
    reports={m:dict(cases=rows(3.)) for m in OUTPUTS};reports[EXTRA_BASELINE]=dict(cases=rows(2.8))
    reports['retained-descent']['cases'][0]['prediction']['alignn_gap']=None
    result=fresh_sensitivity(reports,ids)['retained-descent']
    assert result['crystalformer']['MAE_vs_fresh_rawB_delta']>0 and result['crystalformer']['material_gains_beyond_fresh_rawB']==0
    assert result['alignn']['denominator']==48 and result['alignn']['paired_known']==47
    reports[EXTRA_BASELINE]['cases'][0],reports[EXTRA_BASELINE]['cases'][1]=reports[EXTRA_BASELINE]['cases'][1],reports[EXTRA_BASELINE]['cases'][0]
    with pytest.raises(ValueError,match='all48'):fresh_sensitivity(reports,ids)


def test_real_diagnostic_hash_and_coverage_when_local_evidence_present():
    root=Path(__file__).resolve().parents[3]/'results/mattergen/evidence/directed-validation48-20261026-seed27'
    diagnostic=root/'baseline-diagnostic-421.json'
    if not diagnostic.exists():pytest.skip('Native evidence is not shipped with source.')
    plan=json.loads((root/'validation-plan.json').read_text());report=json.loads(diagnostic.read_text())
    validate_diagnostic(plan,report,sha(diagnostic))
    assert report['raw_path_mismatches']==3 and report['NFE']==48
