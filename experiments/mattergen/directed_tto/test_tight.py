"""Material-case selection, stricter force gates and paired diagnostic export tests."""
import copy
import csv
import json
import numpy as np
import pytest
from ase import Atoms
from tight_relaxation import material_indices,perturb,tight_eligible,validate_trials,validate_control
import tight_evaluation
from probe import sha


def test_material_selector_uses_all_frozen_cases_and_both_comparators():
    ids=[str(i) for i in range(16)]
    plan=dict(parent_ids=ids,cases=[dict(predictions=dict(after=dict(crystalformer_gap=3.5))) for _ in ids])
    data=dict(selected={'seed':[dict(gap=3.5) for _ in ids],'plain-descent':[dict(gap=3.5) for _ in ids]})
    report=dict(arms={'plain-descent':dict(cases=[dict(parent_id=p,physical_eligible=True) for p in ids])})
    for i in (1,2,3):data['selected']['plain-descent'][i]['gap']=2.8
    report['arms']['plain-descent']['cases'][2]['physical_eligible']=False;data['selected']['seed'][3]['gap']=2.7
    assert material_indices(plan,data,report)==[1]
    bad=copy.deepcopy(report);bad['arms']['plain-descent']['cases'][0]['parent_id']='wrong'
    with pytest.raises(ValueError,match='coverage'):material_indices(plan,data,bad)


def test_shared_perturbations_preserve_species_source_and_physical_amplitudes():
    reference=Atoms(numbers=[14,8],cell=np.eye(3)*5,scaled_positions=[[.1,.2,.3],[.6,.7,.8]],pbc=True)
    other=reference.copy();other.set_cell(reference.cell.array*1.04,scale_atoms=True)
    before=reference.positions.copy();zero=perturb(reference,15,0)
    np.testing.assert_array_equal(zero.positions,before)
    for trial in (1,2,3):
        a=perturb(reference,15,trial);b=perturb(other,15,trial)
        noise=a.positions-reference.get_scaled_positions()@a.cell.array
        other_noise=b.positions-other.get_scaled_positions()@b.cell.array
        np.testing.assert_allclose(noise,other_noise,atol=1e-14)
        assert np.isclose(np.sqrt(np.mean(np.sum(noise**2,axis=1))),.02)
        np.testing.assert_array_equal(a.numbers,reference.numbers)
        assert np.linalg.norm(np.linalg.solve(reference.cell.array,a.cell.array)-np.eye(3),2)<=.00300000001
    np.testing.assert_array_equal(reference.positions,before)


def test_tighter_force_gate_and_complete_correlated_trial_coverage():
    record=dict(status='converged',converged=True,separated=True,max_raw_atomic_force=.0005,max_filter_force=.0008)
    assert tight_eligible(record)
    for key,value in (('max_raw_atomic_force',.005),('max_filter_force',float('nan')),('converged',False)):
        bad=dict(record);bad[key]=value;assert not tight_eligible(bad)
    rows=[dict(parent_id=p,stage=s,trial=t) for p in ('a','b') for s in ('direct','selected') for t in range(4)]
    validate_trials(['a','b'],rows)
    with pytest.raises(ValueError,match='Missing/duplicate'):validate_trials(['a','b'],rows[:-1]+[rows[0]])
    saved=dict(checkpoint_sha256='pinned',final_energy_eV=-4.,num_atoms=2,max_raw_atomic_force=.005)
    assert validate_control(-4.0001,.00505,saved,'pinned')['passed']
    with pytest.raises(ValueError):validate_control(-4.1,.005,saved,'pinned')
    with pytest.raises(ValueError):validate_control(-4.,.005,saved,'changed')


def test_diagnostic_export_uses_actual_trial_count_not_parent_count(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();root=tmp_path/'diagnostic';root.mkdir()
    pose=source/'parent.vasp';pose.write_text('immutable source')
    case=dict(parent_id='mattergen-0000.vasp',B=dict(relaxed_path=str(pose),relaxed_sha256=sha(pose)),
        predictions=dict(after=dict(crystalformer_gap=2.5,crystalformer_eform=-1.)),inventory={},neutral=True)
    selected=dict(pose=str(pose),gap=2.5,eform=-1.,B=dict(relaxed_sha256=sha(pose)))
    (source/'plan.json').write_text(json.dumps(dict(cases=[case])))
    (source/'direction-results.json').write_text(json.dumps(dict(selected={'plain-descent':[selected]})))
    class FakeScorer:
        def __init__(self):self.metadata={}
        def predict(self,path):return (2.5,-1.)
    monkeypatch.setattr(tight_evaluation,'GradientScorer',FakeScorer)
    records=[]
    for stage in ('direct','selected'):
        for trial in range(4):
            records.append(dict(parent_id=case['parent_id'],candidate_id=f'parent-{stage}-t{trial}.vasp',stage=stage,trial=trial,
                input=str(pose),input_sha256=sha(pose),relaxed_path=str(pose),relaxed_sha256=sha(pose),status='converged'))
    tight_evaluation.prepare(root,dict(indices=[0]),source,dict(records=records))
    inventory=json.loads((root/'evaluation-tight/inventory-249/inventory.json').read_text())
    assert inventory['denominator']==8 and len({r['candidate_id'] for r in inventory['candidates']})==8
    with (root/'evaluation-tight/result.csv').open() as f:assert len(list(csv.DictReader(f)))==8
    assert len(list((root/'evaluation-tight/relaxation').glob('*/result.json')))==8
