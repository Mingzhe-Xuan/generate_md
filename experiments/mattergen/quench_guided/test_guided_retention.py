"""All-family selection, matched eight-stage controls and correlated-trial accounting."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import guided_retention
from guided_retention import material_cases,endpoints,stages,validate_trials,summarize,report
from guided_settings import FAMILIES,PARENTS,ARMS


def source_fixture():
    data=dict(baseline=[dict(id=p,gap=3.5) for p in PARENTS],selected={});arms={};assessment={}
    for f in FAMILIES:
        for a in ARMS+('probes',):
            points=[dict(id=f'{f}-{a}-{p}',gap=3.5) for p in PARENTS];data['selected'][f+'-'+a]=points
            arms[f+'-'+a]=dict(cases=[dict(parent_id=p,point_id=q['id'],physical_eligible=True,prediction=dict(crystalformer_gap=3.5,alignn_gap=99.)) for p,q in zip(PARENTS,points)])
        index=1 if f!='random_guided' else 3
        data['selected'][f+'-descent'][index]['gap']=3.;arms[f+'-descent']['cases'][index]['prediction']['crystalformer_gap']=3.
        assessment[f]=dict(material_parents_beyond_probe_pool=1)
    return dict(parent_ids=list(PARENTS)),data,dict(arms=arms,assessment=dict(families=assessment))


def records_fixture(family='cf_guided'):
    case=dict(family=family,parent_id='p',index=0,source_gain_eV=.5);records=[];pred={}
    for stage in stages(family):
        for trial in range(4):
            identifier=f'{family}-{stage}-{trial}'
            records.append(dict(candidate_id=identifier,family=family,parent_id='p',stage=stage,trial=trial,status='converged',converged=True,separated=True,
                max_raw_atomic_force=.0005,max_filter_force=.0005,final_energy_eV=-5.,num_atoms=2,original_energy_eV_atom=-2.5,original_volume_ratio=1.))
            gap=3. if stage=='selected' else 3.6
            pred[identifier]=dict(crystalformer_gap=gap,crystalformer_eform=-1.,alignn_gap=gap)
    return case,records,pred


def test_all_material_families_selected_without_alignn_filter_or_parent_pooling():
    plan,data,evidence=source_fixture();cases=material_cases(plan,data,evidence)
    assert [c['family'] for c in cases]==list(FAMILIES) and len({c['parent_id'] for c in cases})==2
    points=endpoints(data,cases[0]);assert len(points)==8 and set(points)==set(stages('cf_guided'))
    assert points['other_isotropic']==data['selected']['isotropic-descent'][1]
    altered=copy.deepcopy(evidence);altered['assessment']['families']['isotropic']['material_parents_beyond_probe_pool']=0
    with pytest.raises(ValueError,match='count'):material_cases(plan,data,altered)
    altered=copy.deepcopy(evidence);altered['arms']['cf_guided-probes']['cases'].reverse()
    with pytest.raises(ValueError,match='coverage'):material_cases(plan,data,altered)


def test_every_material_case_keeps_all32_matched_trials_and_unique_ids():
    case,records,_=records_fixture();validate_trials([case],records)
    with pytest.raises(ValueError,match='eight-stage'):validate_trials([case],records[:-1])
    altered=copy.deepcopy(records);altered[-1]['candidate_id']=altered[0]['candidate_id']
    with pytest.raises(ValueError):validate_trials([case],altered)
    with pytest.raises(ValueError):validate_trials([case,case],records)


def test_matched_probes_direction_and_other_prior_controls_preserve_unknowns():
    case,records,pred=records_fixture();result=summarize([case],records,pred)[0]
    assert result['models']['crystalformer']['probes']['all_four_gain_ge_0p1']
    assert result['models']['crystalformer']['other_isotropic']['all_four_gain_gt_1meV']
    for r in records:
        if r['stage']=='random' and r['trial']==0:r['max_raw_atomic_force']=.01
    pred['cf_guided-selected-0']['alignn_gap']=None
    result=summarize([case],records,pred)[0]
    assert result['models']['crystalformer']['random']['eligible_known_pairs']==3
    assert not result['models']['crystalformer']['random']['all_four_gain_gt_1meV']
    assert result['models']['alignn']['probes']['known_pairs']==3
    assert not result['models']['alignn']['probes']['all_four_gain_ge_0p1']
    pred['extra']={}
    with pytest.raises(ValueError,match='coverage'):summarize([case],records,pred)


def test_empty_material_report_is_explicit_negative_for_every_family(tmp_path):
    report(tmp_path,dict(cases=[],source_assessment={}),dict(records=[],controls=[],NFE=0))
    result=json.loads((tmp_path/'report.json').read_text())
    assert not result['goal_completion_proven'] and result['cases']==[]
    for f in FAMILIES:
        assert result['counts'][f]['crystalformer']['original_parent_denominator']==16
        assert result['counts'][f]['crystalformer']['retained_vs_original_and_probes']==0


def test_retention_specification_pins_source_helpers_and_complete_data(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();root=tmp_path/'retention';root.mkdir();artifact=source/'report.json';artifact.write_text('source')
    sha=guided_retention.sha;monkeypatch.setattr(guided_retention,'source_evidence',lambda p:({}, {}, {}, []))
    spec=dict(policy=guided_retention.POLICY,source=str(source),cases=[],source_hashes={'report.json':sha(artifact)},implementation_sha256=sha(guided_retention.__file__),helper_sha256=guided_retention.helpers())
    (root/'retention-plan.json').write_text(json.dumps(spec));(root/'physics.json').write_text(json.dumps(dict(records=[],controls=[])))
    (root/'physics.complete.json').write_text(json.dumps(dict(completed=True,physics_sha256=sha(root/'physics.json'))))
    guided_retention.specification(root)
    artifact.write_text('changed')
    with pytest.raises(ValueError,match='artifact'):guided_retention.specification(root)
    artifact.write_text('source');bad=copy.deepcopy(spec);bad['helper_sha256']['controls']='changed';(root/'retention-plan.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='implementation'):guided_retention.specification(root)


def test_guided_retention_cli_imports_without_previous_model_imports(tmp_path):
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(Path(__file__).with_name('guided_retention.py')),'--help'],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
    assert '--mode' in result.stdout and '--source' in result.stdout
