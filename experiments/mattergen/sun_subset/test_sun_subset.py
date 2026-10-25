"""No changes to the property-selected cohort or chemical competitors."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('sun_subset_policy',Path(__file__).with_name('policy.py'))
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


def fixture():
    parents=[str(i) for i in range(128)];cases=[];selected={p:None for p in parents}
    for parent in parents[:12]:
        identifier=parent+'-b01';selected[parent]=dict(candidate_id=identifier)
        cases.append(dict(group=policy.GROUP,candidate_id=identifier,parent_id=parent,branch_index=1,
            prediction=dict(crystalformer_flags=dict(joint=True))))
    records=dict(candidates=cases,selections={g+'/looser/best_then_classify':{policy.GROUP:copy.deepcopy(selected)} for g in ('legacy','force_eform')})
    return records,parents


def test_exact_cohort():
    records,parents=fixture();assert len(policy.selected_cases(records,parents))==12


@pytest.mark.parametrize('kind',['missing','parent','zero','gate','count'])
def test_reject_changed_cohort(kind):
    records,parents=fixture()
    if kind=='missing':parents=parents[:-1]
    if kind=='parent':records['candidates'][0]['parent_id']='other'
    if kind=='zero':records['candidates'][0]['branch_index']=0
    if kind=='gate':records['selections']['legacy/looser/best_then_classify'][policy.GROUP]['0']=None
    if kind=='count':records['candidates'][0]['prediction']['crystalformer_flags']['joint']=False
    with pytest.raises(ValueError):policy.selected_cases(records,parents)


def test_subsystems():
    systems=[['Li','O'],['Na','Cl']]
    assert policy.needed(['Li'],systems) and policy.needed(['Li','O'],systems)
    assert not policy.needed(['Li','Cl'],systems) and not policy.needed(['F'],systems)
