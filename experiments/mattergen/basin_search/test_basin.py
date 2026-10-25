"""Cohort, physical guards and new proposal geometry without model inference."""
import importlib.util
import copy
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from unittest.mock import patch


def module(name):
    spec=importlib.util.spec_from_file_location('basin_'+name,Path(__file__).with_name(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


policy=module('policy');proposals=module('proposals')


def test_cohort():
    ids=[str(i) for i in range(128)];converged=set(ids)-{'5'};a=policy.choose_parents(ids,converged)
    assert len(a)==len(set(a))==16 and '5' not in a and a==[x for x in ids if x in a]
    assert policy.choose_parents(ids,converged)==a
    assert policy.choose_parents(ids,converged,42)!=a
    with pytest.raises(ValueError):policy.choose_parents(ids[:-1],converged)


def test_physical_selection_no_holdout_leak():
    b=dict(status='converged',converged=True,max_filter_force=.001,max_raw_atomic_force=.002,separated=True,final_energy_eV=-20,num_atoms=10,volume_ratio=1.)
    base=dict(id='base',gap=1,eform=-1,B=b)
    a=dict(id='a',gap=2.4,eform=-1,B=dict(b),alignn_gap=7)
    c=dict(id='c',gap=2.1,eform=-1,B=dict(b),alignn_gap=2.5)
    assert policy.best([base,a,c],-2) is a
    a['B']['volume_ratio']=2.;assert policy.best([base,a,c],-2) is c
    c['B']['final_energy_eV']=-18;assert policy.best([base,a,c],-2) is base
    for key,value in [('max_filter_force',.011),('max_raw_atomic_force',.011),('separated',False),('converged',False)]:
        record=dict(b);record[key]=value;assert not policy.physical(record,-2)


def test_soft_directions_translation_free():
    masses=np.array([1.,2.,3.]);H=np.kron(np.eye(3)-np.ones((3,3))/3,np.eye(3));values,modes=proposals.soft_directions(H,masses)
    assert len(values)==6 and modes.shape==(6,3,3) and np.isfinite(modes).all()
    assert np.max(np.abs(np.einsum('mij,i->mj',modes,masses)))<1e-10


def test_bounded_collective_proposals():
    atoms=Atoms('Si4',scaled_positions=[[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5]],cell=[6,6,6],pbc=True)
    for collective in (False,True):
        changed=proposals.mutate(atoms,atoms,np.random.default_rng(42),.4,collective=collective,strain=.04)
        assert np.array_equal(changed.numbers,atoms.numbers)
        assert .65<=changed.get_volume()/atoms.get_volume()<=1.4
        distances=changed.get_all_distances(mic=True);np.fill_diagonal(distances,np.inf);assert distances.min()>=.7
        assert not np.allclose(changed.positions,atoms.positions)


def test_attempt_ledger_keeps_rejections():
    with patch.dict('sys.modules',{'policy':policy}):report=module('report')
    parents=['p0','p1'];history=[]
    for method in policy.METHODS:
        for parent in parents:
            for generation in range(policy.ROUNDS):
                for child in range(policy.WIDTH):
                    identifier=f'{parent}-{method}-g{generation:02}-c{child}'
                    history.append(dict(method=method,parent=parent,id=identifier,status='proposal_rejected'))
    ledger=report.audit_attempts(history,parents);assert len(ledger)==96
    with pytest.raises(ValueError):report.audit_attempts(history[:-1],parents)


def test_physical_nan_and_cell_guard():
    b=dict(status='converged',converged=True,max_filter_force=.001,max_raw_atomic_force=.002,separated=True,final_energy_eV=-20,num_atoms=10,volume_ratio=1.)
    for value in (float('nan'),.6,1.5):
        b['volume_ratio']=value;assert not policy.physical(b,-2)


def test_ase_md_velocity_cleanup():
    from ase.calculators.lj import LennardJones
    from ase.md.verlet import VelocityVerlet
    from ase import units
    atoms=Atoms('Ar2',positions=[[0,0,0],[3.8,0,0]],cell=[12,12,12],pbc=True)
    atoms.calc=LennardJones();atoms.set_velocities(np.ones((2,3))*.01)
    VelocityVerlet(atoms,.5*units.fs,logfile=None).run(2)
    before=atoms.positions.copy();atoms.set_momenta(None)
    np.testing.assert_allclose(atoms.positions,before)
    assert not atoms.has('momenta')


def test_confirmation_trigger_and_frozen_cohort():
    with patch.dict('sys.modules',{'policy':policy}):confirmation=module('confirmation')
    ids=[str(i) for i in range(128)];parents=policy.choose_parents(ids,set(ids)-{'5'})
    plan=dict(parent_count=16,parent_ids=parents,source_inventory_sha256='fixed',acceptance=copy.deepcopy(policy.ACCEPTANCE))
    report=dict(plan=plan,methods=dict(iid=dict(summary=dict(discovery_gate_passed=True))))
    assert confirmation.qualifying(report)==[]
    report['methods']['beam']=dict(summary=dict(discovery_gate_passed=True))
    assert confirmation.qualifying(report)==['beam']
    assert confirmation.frozen_parents(plan,ids,'fixed')==parents
    with pytest.raises(ValueError):confirmation.frozen_parents(plan,ids,'changed')
    bad=copy.deepcopy(plan);bad['acceptance']['min_changed_improved_parents']=1
    with pytest.raises(ValueError):confirmation.frozen_parents(bad,ids,'fixed')
    bad=copy.deepcopy(plan);bad['parent_ids']=parents[::-1]
    with pytest.raises(ValueError):confirmation.frozen_parents(bad,ids,'fixed')


def test_attempt_prefix_counts_rejected_proposals():
    with patch.dict('sys.modules',{'policy':policy}):report=module('report')
    with patch.dict('sys.modules',{'policy':policy,'report':report}):scaling=module('scaling')
    base=dict(id='mattergen-0001.vasp')
    # First attempt failed, so the first saved successful point has attempt index2.
    candidate=dict(id='mattergen-0001-beam-g00-c1')
    points=[base,candidate]
    assert scaling.prefix_pool(points,base['id'],'beam',1)==[base]
    assert scaling.prefix_pool(points,base['id'],'beam',2)==points
    assert scaling.attempt_number('mattergen-0001-beam-g03-c3',base['id'],'beam')==16
    with pytest.raises(ValueError):scaling.prefix_pool([base,dict(id='mattergen-0002-beam-g00-c0')],base['id'],'beam',16)
    with pytest.raises(ValueError):scaling.attempt_number('mattergen-0001-beam-g04-c0',base['id'],'beam')


def test_endpoint_archive_actual_ase_roundtrip(tmp_path):
    from ase.io import write,read
    with patch.dict('sys.modules',{'policy':policy}):endpoints=module('endpoints')
    a=Atoms('Si2',scaled_positions=[[0,0,0],[.3,.4,.5]],cell=[[4,0,0],[.1,4,0],[.2,.1,5]],pbc=True)
    b=a.copy();b.positions[1,0]+=.1
    first=tmp_path/'first.vasp';second=tmp_path/'second.vasp'
    write(first,a,format='vasp');write(second,b,format='vasp')
    labels=[dict(stage='direct'),dict(stage='selected')]
    endpoints.pair_files(first,endpoints.sha(first),second,endpoints.sha(second),tmp_path/'out',labels)
    frames=read(tmp_path/'out/comparison.traj',index=':')
    np.testing.assert_allclose(frames[0].positions,a.positions,atol=1e-12)
    np.testing.assert_allclose(frames[1].cell.array,b.cell.array,atol=1e-12)
    assert [f.info['stage'] for f in frames]==['direct','selected']
    with pytest.raises(ValueError):endpoints.pair_files(first,'tampered',second,endpoints.sha(second),tmp_path/'invalid',labels)


def test_display_alignment_preserves_skewcell_distances():
    with patch.dict('sys.modules',{'policy':policy}):endpoints=module('endpoints')
    a=Atoms('NaClSiO',scaled_positions=[[.1,.2,.3],[.6,.2,.5],[.3,.8,.1],[.9,.7,.9]],cell=[[5,0,0],[1.1,4,0],[-.6,2.2,6]],pbc=True)
    b=a.copy();b.set_cell([[4.7,0,0],[1.8,4.5,0],[-1.5,1.2,5.5]],scale_atoms=True);b.positions[1]+=[.2,-.1,.3]
    b.rotate(27,'z',rotate_cell=True)
    original=[x.copy() for x in (a,b)]
    display=endpoints.display_pair((a,b))
    for x,y,z in zip((a,b),display,original):
        np.testing.assert_allclose(x.positions,z.positions,atol=1e-12)
        np.testing.assert_allclose(x.get_all_distances(mic=True),y.get_all_distances(mic=True),atol=1e-8)
        np.testing.assert_allclose(x.cell.array@x.cell.array.T,y.cell.array@y.cell.array.T,atol=1e-8)
