"""Frozen48 coverage, native derivative gates, conservative controls and export."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import validation_run
from validation_run import six_checks_pass,validate_results,OUTPUTS,validate_manifest,PLAN_SHA
from validation_report import assess
from validation_policy import VALIDATION_POLICY
from settings import SCALES,POLICY
from policy import best
from evaluation_export import export_evaluation
from probe import sha


def point(identifier,gap=3.5):
    return dict(id=identifier,gap=gap,eform=-1.,B=dict(status='converged',converged=True,max_filter_force=.005,max_raw_atomic_force=.005,
        separated=True,volume_ratio=1.,volume_A3=100.,final_energy_eV=-5.,num_atoms=2,relaxed_sha256=identifier))


def fixture():
    ids=[f'p{i}' for i in range(48)];bases=[point(p) for p in ids]
    plan=dict(parent_ids=ids,cases=[dict(parent_id=p,B=copy.deepcopy(b['B'])) for p,b in zip(ids,bases)])
    selected={name:[] for name in OUTPUTS};seeds=[];attempts=[]
    for i,parent in enumerate(ids):
        base=bases[i];seed=point(parent+'-seed',3.4)
        if i==1:seed['gap']=2.5;seed['eform']=0.
        control=best([base,seed],-2.5);seeds.append(dict(parent_id=parent,point=seed,control=control))
        for arm,gap in (('descent',2.9),('reverse',3.1),('random',3.2)):
            winner=point(parent+'-'+arm,gap)
            for scale in SCALES:
                row=dict(parent_id=parent,arm=arm,scale=scale,eligible=scale==.8,status='converged' if scale==.8 else 'geometric_rejection')
                if scale==.8:row['point']=winner
                attempts.append(row)
            selected['ordinary-'+arm].append(winner);selected['retained-'+arm].append(winner)
        selected['baseline'].append(base)
    return plan,dict(completed_parents=48,baseline=bases,seeds=seeds,attempts=attempts,selected=selected)


def test_native_cf_gate_requires_exact_all_six_independent_checks():
    rows=[dict(block=b,epsilon=e,passed=True) for b in ('atomic','strain','both') for e in POLICY['CF_FD_steps_A']]
    assert six_checks_pass(rows)
    assert not six_checks_pass(rows[:-1]+[rows[0]]) and not six_checks_pass(rows[:-1])
    rows[0]['passed']=False;assert not six_checks_pass(rows)


def test_validation_manifest_rejects_changed_seed_policy_or_overlap(monkeypatch):
    monkeypatch.setattr(validation_run,'code_hashes',lambda:dict(math='frozen'))
    ids=[f'p{i}' for i in range(48)]
    plan=dict(parent_ids=ids,cases=[dict(parent_id=p) for p in ids],pilot_ids=[f'old{i}' for i in range(16)],
        search_seed=20261027,policy=copy.deepcopy(VALIDATION_POLICY),native_math_sha256=dict(math='frozen'))
    validate_manifest(plan,PLAN_SHA)
    with pytest.raises(ValueError,match='plan changed'):validate_manifest(plan,'wrong_sha')
    altered=copy.deepcopy(plan);altered['search_seed']=20261026
    with pytest.raises(ValueError,match='plan changed'):validate_manifest(altered,PLAN_SHA)
    altered=copy.deepcopy(plan);altered['pilot_ids'][0]=ids[0]
    with pytest.raises(ValueError,match='cohort changed'):validate_manifest(altered,PLAN_SHA)


def test_validation_ledger_keeps_all48_and_feasible_seed_control():
    plan,data=fixture();validate_results(plan,data)
    assert data['seeds'][1]['control']==data['baseline'][1]
    changed=copy.deepcopy(data);changed['seeds'][1]['control']=changed['seeds'][1]['point']
    with pytest.raises(ValueError,match='comparator'):validate_results(plan,changed)
    changed=copy.deepcopy(data);changed['attempts'][-1]=copy.deepcopy(changed['attempts'][0])
    with pytest.raises(ValueError,match='Missing/duplicate'):validate_results(plan,changed)
    changed=copy.deepcopy(data);changed['completed_parents']=47
    with pytest.raises(ValueError,match='denominator'):validate_results(plan,changed)


def test_validation_ledger_rejects_stale_selection_and_rolling_volume():
    plan,data=fixture();changed=copy.deepcopy(data);changed['selected']['retained-descent'][0]=changed['baseline'][0]
    with pytest.raises(ValueError,match='Retained selection'):validate_results(plan,changed)
    changed=copy.deepcopy(data);changed['attempts'][3]['point']['B']['volume_A3']=80.
    with pytest.raises(ValueError,match='volume'):validate_results(plan,changed)
    changed=copy.deepcopy(data);changed['baseline'][0]['B']['final_energy_eV']=-6.
    with pytest.raises(ValueError,match='Immutable'):validate_results(plan,changed)


def test_validation_evidence_is_parent_paired_and_needs_both_controls():
    g=np.full(48,.4);base=np.ones(48);valid=np.ones(48,dtype=bool)
    good=assess(dict(descent=g,reverse=np.full(48,.7),random=np.full(48,.8)),base,base,valid)
    assert good['independent_directional_evidence'] and good['material_parents']==48 and not good['goal_completion_proven']
    assert not assess(dict(descent=g,reverse=np.full(48,.3),random=np.full(48,.8)),base,base,valid)['independent_directional_evidence']
    assert not assess(dict(descent=g,reverse=np.full(48,.7),random=np.full(48,.8)),base,base,np.zeros(48,dtype=bool))['independent_directional_evidence']
    with pytest.raises(ValueError,match='all48'):assess(dict(descent=g[:-1],reverse=g,random=g),base,base,valid)


def test_variable_export_checks_manifest_source_and_idempotence(tmp_path):
    class Scorer:
        def predict(self,path):return (2.5,-1.)
    raw=tmp_path/'raw.vasp';pose=tmp_path/'pose.vasp';raw.write_text('raw bytes');pose.write_text('pose bytes')
    ids=['a.vasp','b.vasp','c.vasp'];plan=dict(parent_ids=ids,cases=[dict(parent_id=p,inventory={},neutral=True) for p in ids])
    q=dict(id='winner',input=str(raw),pose=str(pose),B=dict(input_sha256=sha(raw),relaxed_sha256=sha(pose),status='converged'))
    points=[copy.deepcopy(q) for _ in ids];export_evaluation(tmp_path,'test',points,plan,Scorer())
    destination=tmp_path/'evaluation-test';inventory=json.loads((destination/'inventory-249/inventory.json').read_text())
    assert inventory['denominator']==3 and [r['candidate_id'] for r in inventory['candidates']]==ids
    with (destination/'result.csv').open() as f:assert len(list(csv.DictReader(f)))==3
    export_evaluation(tmp_path,'test',points,plan,Scorer())
    changed=copy.deepcopy(points);changed[0]['id']='stale'
    with pytest.raises(ValueError,match='Stale'):export_evaluation(tmp_path,'test',changed,plan,Scorer())
    pose.write_text('changed bytes')
    with pytest.raises(ValueError,match='pose changed'):export_evaluation(tmp_path,'test',points,plan,Scorer())


def test_validation_command_imports_are_independent_of_test_order(tmp_path):
    folder=Path(__file__).resolve().parent;environment=dict(os.environ);environment.pop('PYTHONPATH',None)
    for script in ('validation_run.py','validation_report.py'):
        result=subprocess.run([sys.executable,str(folder/script),'--help'],cwd=tmp_path,env=environment,check=True,capture_output=True,text=True)
        assert '--run' in result.stdout
