"""Cohort, physical guards and new proposal geometry without model inference."""
import importlib.util
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
