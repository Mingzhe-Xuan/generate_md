"""Disjoint population validation and conservative CF-only acceptance controls."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import pytest
from validation_policy import heldout_ids,retained_choice


def point(identifier,gap):
    return dict(id=identifier,gap=gap,eform=-1.,B=dict(status='converged',converged=True,max_raw_atomic_force=.005,max_filter_force=.005,
        separated=True,volume_ratio=1.,final_energy_eV=-5.,num_atoms=2))


def test_all_remaining48_never_use_property_or_outcome_screening():
    ids=[str(i) for i in range(128)];neutral=set(ids[:65]);converged=set(ids)-{'64'};pilot=ids[:16]
    assert heldout_ids(ids,neutral,converged,pilot)==ids[16:64]
    assert set(heldout_ids(ids,neutral,converged,pilot)).isdisjoint(pilot)
    for altered_ids,altered_neutral,altered_pilot in ((ids[:-1]+[ids[0]],neutral,pilot),(ids,neutral-{'20'},pilot),(ids,neutral,pilot[:-1]+['90']),(ids,neutral,list(reversed(pilot)))):
        with pytest.raises(ValueError):heldout_ids(altered_ids,altered_neutral,converged,altered_pilot)


def test_retained_acceptance_requires_material_gain_beyond_both_controls():
    base=point('base',3.5);seed=point('seed',3.4);small=point('small',3.35);good=point('good',3.1)
    assert retained_choice(base,seed,[small],-2.5) is base
    assert retained_choice(base,seed,[small,good],-2.5) is good
    before=copy.deepcopy([base,seed,good]);retained_choice(base,seed,[good],-2.5)
    assert [base,seed,good]==before
    strong_seed=point('strong-seed',3.05)
    assert retained_choice(base,strong_seed,[good],-2.5) is base


def test_retained_acceptance_never_bypasses_physical_or_formation_guard():
    base=point('base',3.5);seed=point('seed',3.4);perfect=point('perfect',2.5)
    for key,value in (('max_raw_atomic_force',.1),('volume_ratio',1.5),('final_energy_eV',-4.)):
        bad=copy.deepcopy(perfect);bad['B'][key]=value
        assert retained_choice(base,seed,[bad],-2.5) is base
    bad=copy.deepcopy(perfect);bad['eform']=0.
    assert retained_choice(base,seed,[bad],-2.5) is base


def test_preregistration_imports_in_clean_process_without_test_order_dependency(tmp_path):
    folder=str(Path(__file__).resolve().parent);environment=dict(os.environ);environment.pop('PYTHONPATH',None)
    code='import sys;sys.path.insert(0,'+repr(folder)+');import validation_policy,validation_plan;print(validation_policy.VALIDATION_POLICY["parent_count"])'
    result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=environment,check=True,capture_output=True,text=True)
    assert result.stdout.strip()=='48'
