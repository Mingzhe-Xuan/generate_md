"""Matched original/fresh/selected baselines and complete physical-retention trials."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import validation_retention
from validation_retention import material_indices,validate_trials,paired_summary,export_trials,report,STAGES
from test_validation_runtime import fixture
from probe import sha


def source_fixture():
    plan,data=fixture();data['raw_replay_baseline']=copy.deepcopy(data['baseline']);rows=[]
    for i,parent in enumerate(plan['parent_ids']):
        q=data['selected']['retained-descent'][i];zero=data['seeds'][i]['control'];base=data['baseline'][i]
        gain=min(abs(base['gap']-2.5),abs(zero['gap']-2.5))-abs(q['gap']-2.5)
        rows.append(dict(parent_id=parent,point_id=q['id'],physical_eligible=True,material_CF_gain=True,retained_CF_gain_eV=gain))
    return plan,data,dict(arms={'retained-descent':dict(cases=rows)})


def test_retention_selects_all_material48_without_fresh_or_alignn_filtering():
    plan,data,evidence=source_fixture();rows=evidence['arms']['retained-descent']['cases']
    assert material_indices(plan,data,evidence)==list(range(48))
    data['raw_replay_baseline'][0]['gap']=2.5
    rows[0]['prediction']=dict(alignn_gap=20.)
    data['selected']['retained-descent'][1]['eform']=0.;rows[1]['physical_eligible']=False;rows[1]['material_CF_gain']=False
    data['seeds'][2]['control']['gap']=2.7;rows[2]['retained_CF_gain_eV']=.2-.4;rows[2]['material_CF_gain']=False
    assert material_indices(plan,data,evidence)==[0]+list(range(3,48))
    bad=copy.deepcopy(evidence);bad['arms']['retained-descent']['cases'][0]['material_CF_gain']=False
    with pytest.raises(ValueError,match='linkage'):material_indices(plan,data,bad)
    bad=copy.deepcopy(evidence);bad['arms']['retained-descent']['cases'].reverse()
    with pytest.raises(ValueError,match='coverage'):material_indices(plan,data,bad)


def trials_fixture():
    rows=[];predictions={}
    for stage in STAGES:
        for trial in range(4):
            identifier=f'p-{stage}-t{trial}'
            rows.append(dict(candidate_id=identifier,parent_id='p',stage=stage,trial=trial,status='converged',converged=True,separated=True,max_raw_atomic_force=.0005,max_filter_force=.0005))
            gap=dict(original=4.,fresh=3.7,selected=3.)[stage]
            predictions[identifier]=dict(crystalformer_gap=gap,crystalformer_eform=-1.,alignn_gap=gap)
    return rows,predictions


def test_three_stage_trials_require_all12_and_unique_identifiers():
    rows,_=trials_fixture();validate_trials(['p'],rows)
    for bad in (rows[:-1],rows[:-1]+[rows[0]],copy.deepcopy(rows)):
        if len(bad)==12 and bad[-1] is not rows[0]:bad[-1]['candidate_id']=bad[0]['candidate_id']
        with pytest.raises(ValueError,match='paired trials'):validate_trials(['p'],bad)
    with pytest.raises(ValueError):validate_trials(['p','p'],rows)


def test_retention_reports_matched_original_and_fresh_with_unknowns_and_failures():
    rows,predictions=trials_fixture();parents=paired_summary(['p'],rows,predictions)
    assert parents[0]['summary']['crystalformer']['original']['all_four_gains_ge_0p1']
    assert parents[0]['summary']['crystalformer']['fresh']['all_four_gains_ge_0p1']
    predictions['p-selected-t0']['alignn_gap']=None
    rows[4]['max_raw_atomic_force']=.01
    parent=paired_summary(['p'],rows,predictions)[0]
    assert parent['summary']['crystalformer']['original']['all_four_gains_ge_0p1']
    assert not parent['summary']['crystalformer']['fresh']['all_four_gains_ge_0p1']
    assert parent['summary']['crystalformer']['fresh']['eligible_known_pairs']==3
    assert parent['summary']['alignn']['original']['known_pairs']==3
    assert not parent['summary']['alignn']['original']['all_four_gains_ge_0p1']
    assert len(parent['trials'])==4
    predictions['p-extra']=dict(crystalformer_gap=2.5)
    with pytest.raises(ValueError,match='coverage'):paired_summary(['p'],rows,predictions)


def test_retention_export_keeps_failed_trial_in_actual12_denominator(tmp_path):
    raw=tmp_path/'input.vasp';pose=tmp_path/'terminal.vasp';raw.write_text('input');pose.write_text('terminal')
    rows,_=trials_fixture()
    for r in rows:r.update(input=str(raw),input_sha256=sha(raw),relaxed_path=str(pose),relaxed_sha256=sha(pose))
    rows[-1]['status']='failed';rows[-1].pop('relaxed_path');rows[-1].pop('relaxed_sha256')
    class Scorer:
        def predict(self,path):return (2.5,-1.)
    plan=dict(cases=[dict(parent_id='p',inventory={},neutral=True)])
    export_trials(tmp_path,plan,rows,Scorer());folder=tmp_path/'evaluation-retention'
    inventory=json.loads((folder/'inventory-249/inventory.json').read_text())
    assert inventory['denominator']==12 and len(inventory['candidates'])==12
    with (folder/'result.csv').open() as h:assert len(list(csv.DictReader(h)))==12
    failed=json.loads((folder/'relaxation'/rows[-1]['candidate_id']/'result.json').read_text())
    assert failed['status']=='failed' and 'relaxed_path' not in failed


def test_empty_material_cohort_is_explicit_negative_not_success(tmp_path):
    spec=dict(parent_ids=[],source_directional_evidence=False,source_fresh_baseline_robustness=False)
    report(tmp_path,spec,dict(records=[],controls=[],NFE=0))
    result=json.loads((tmp_path/'retention-report.json').read_text())
    assert result['original_population']==48 and result['selected_material_parents']==0
    assert result['robust_parents_against_both_matched_baselines']==dict(crystalformer=0,alignn=0)
    assert not result['goal_completion_proven']


def test_retention_cli_imports_without_shared_test_state(tmp_path):
    script=Path(__file__).with_name('validation_retention.py');env=dict(os.environ);env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(script),'--help'],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
    assert '--source' in result.stdout and '--mode' in result.stdout


def test_retention_specification_pins_helpers_source_and_complete_physics(tmp_path,monkeypatch):
    root=tmp_path/'retention';root.mkdir();source=tmp_path/'source';source.mkdir();file=source/'source.json';file.write_text('source bytes')
    monkeypatch.setattr(validation_retention,'source_evidence',lambda path:(dict(parent_ids=[]),{}, {}, []))
    spec=dict(policy=validation_retention.RETENTION_POLICY,source=str(source),parent_ids=[],indices=[],source_hashes={'source.json':sha(file)},
        implementation_sha256=sha(validation_retention.__file__),helper_sha256=validation_retention.helper_hashes())
    def save(name,data):(root/name).write_text(json.dumps(data))
    save('retention-plan.json',spec);save('retention-physics.json',dict(records=[],controls=[]));save('retention-physics.complete.json',dict(completed=True,physics_sha256=sha(root/'retention-physics.json')))
    validation_retention.specification(root)
    bad=copy.deepcopy(spec);bad['helper_sha256']['shared_proposals']='changed';save('retention-plan.json',bad)
    with pytest.raises(ValueError,match='implementation'):validation_retention.specification(root)
    save('retention-plan.json',spec);file.write_text('changed source bytes')
    with pytest.raises(ValueError,match='Source48'):validation_retention.specification(root)
    file.write_text('source bytes');save('retention-physics.json',dict(records=[],controls=[],tampered=True))
    with pytest.raises(ValueError,match='complete evidence'):validation_retention.specification(root)
