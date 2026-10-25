"""Selection semantics independent of model inference."""
import copy
import importlib.util
from pathlib import Path
import pytest
from unittest.mock import patch
import hashlib


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parent/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


policy=module('full_policy','selection.py')
templates=module('full_template_policy','../cf_templates/template_policy.py')


def case(branch,gap=2.5,valid=True,match=False):
    return dict(branch_index=branch,separated=True,matches=dict(looser=match),prediction=dict(structure_valid=valid,crystalformer_gap=gap,crystalformer_eform=-1,alignn_gap=9,
        relaxation=dict(status='converged',converged=True,max_filter_force=.001,max_raw_atomic_force=.001,final_energy_eV=-2,num_atoms=2)))


def test_partition():
    ids=[str(i) for i in range(128)];parts=policy.partition(ids)
    assert len(parts)==16 and all(len(p)==8 for p in parts) and sum(parts,[])==ids
    with pytest.raises(ValueError):policy.partition(ids[:-1])
    with pytest.raises(ValueError):policy.partition(ids,16)


def test_explicit_nonneutral_and_order():
    rows=[dict(candidate_id=str(i),eligible=i%2==0) for i in range(4)]
    assert templates.select_explicit(rows,['1','3'])==[rows[1],rows[3]]
    for ids in (['3','1'],['1','1'],['9'],[]):
        with pytest.raises(ValueError):templates.select_explicit(rows,ids)


def test_gate_and_failure():
    c=case(1,valid=False)
    assert not policy.eligible(c) and policy.eligible(c,'force_eform')
    c['separated']=False;assert not policy.eligible(c,'force_eform')
    for field,value in [('max_filter_force',.011),('max_raw_atomic_force',.011),('converged',False),('status','failed')]:
        c=case(1);c['prediction']['relaxation'][field]=value;assert not policy.eligible(c)
    for value in (-.5,float('nan'),None):
        c=case(1);c['prediction']['crystalformer_eform']=value;assert not policy.eligible(c)


def test_no_alignn_leak_and_zero_exclusion():
    a=case(1,2.6);b=case(2,2.7);b['prediction']['alignn_gap']=2.5
    assert policy.pick([case(0),a,b]) is a
    assert policy.pick([case(0),a,b],zero=True)['branch_index']==0
    assert policy.pick([a,b],prefix=1) is a
    assert policy.pick([]) is None


def test_orderings_and_ties():
    matched=case(1,match=True);novel=case(2,gap=2.6)
    assert policy.pick([matched,novel]) is matched
    assert policy.pick([matched,novel],criterion='looser') is novel
    unknown=copy.deepcopy(novel);unknown['matches']={}
    assert policy.pick([unknown],criterion='looser') is None
    a=case(2);b=case(1);assert policy.pick([a,b]) is b
    a['prediction']['relaxation']['final_energy_eV']=-3;assert policy.pick([a,b]) is a


def test_report_missing_and_orderings():
    with patch.dict('sys.modules',{'selection':policy}):report=module('full_report','report.py')
    empty=report.summarize([],{});assert empty['denominator']==128 and empty['no_selection']==128
    assert empty['alignn']['MAE'] is None and empty['alignn']['unknown']==128
    parents=[str(i) for i in range(128)]
    a=case(1,match=True);b=case(2,gap=2.6)
    for r in (a,b):
        r.update(parent_id='0',candidate_id=str(r['branch_index']))
        r['matches']['strict']=r['matches']['looser']
        for model in ('crystalformer','alignn'):r['prediction'][model+'_flags']=dict(gap=True,eform=True,joint=True)
    tables,_=report.tables({'no_CF_noise0.10':[a,b],'CF20_noise0.10':[a,b]},parents,{})
    assert tables['legacy/looser/best_then_classify'][0]['selected_novel_count']==0
    row=tables['legacy/looser/best_among_novel'][0]
    assert row['selected_novel_count']==1 and row['novel_summary']['denominator']==128
    assert row['novel_summary']['crystalformer']['paired_count']==0
    assert tables['legacy/looser/best_among_novel'][1]['selected_novel_count']==1


def test_saved_geometry_hash_species_and_periodic_separation(tmp_path):
    from pymatgen.core import Structure,Lattice
    with patch.dict('sys.modules',{'selection':policy}):report=module('full_geometry_report','report.py')
    path=tmp_path/'pose.vasp';s=Structure(Lattice.cubic(3),['Si'],[[0,0,0]])
    s.to(filename=str(path),fmt='poscar');digest=hashlib.sha256(path.read_bytes()).hexdigest()
    c=case(1);c['original_path']=str(path);c['prediction']['geometry_sha256']=digest
    c['prediction']['relaxation'].update(relaxed_path=str(path),relaxed_sha256=digest)
    checked=report.geometry((copy.deepcopy(c),str(path)))
    assert checked['separated'] and checked['matches']==dict(strict=True,looser=True)
    checked=report.geometry((copy.deepcopy(c),None));assert checked['matches']=={}
    bad=copy.deepcopy(c);bad['prediction']['geometry_sha256']='wrong'
    with pytest.raises(ValueError):report.geometry((bad,str(path)))
    collision=Structure(Lattice.cubic(3),['Si','Si'],[[0,0,0],[.01,0,0]])
    collision.to(filename=str(path),fmt='poscar');digest=hashlib.sha256(path.read_bytes()).hexdigest()
    c['prediction']['geometry_sha256']=digest;c['prediction']['relaxation']['relaxed_sha256']=digest
    assert not report.geometry((c,None))['separated']
