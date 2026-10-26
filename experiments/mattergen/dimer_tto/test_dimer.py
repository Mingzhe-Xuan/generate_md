"""Actual ASE activation, centroid gauge and counted conservative force checks."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes
from ase.optimize import BFGS
from dimer_policy import POLICY,PARENTS,ARMS,directions,ledger
from dimer_engine import Meter,Rejected,BudgetExceeded,FixCentroid,geometry_guard,point,centroid_basis,certify,activate
import dimer_native


class DoubleWell(Calculator):
    implemented_properties=['energy','forces','stress']
    def calculate(self,atoms=None,properties=None,system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        x,y,z=atoms.positions[1]-atoms.positions[0]-[3.,0.,0.]
        gradient=np.array([4*x*(x*x-1),4*y,8*z])
        self.results=dict(energy=(x*x-1)**2+2*y*y+4*z*z,forces=np.array([gradient,-gradient]),stress=np.zeros(6))


def fixture():
    ref=Atoms('Si2',positions=[[-1.,0,0],[1.,0,0]],cell=np.eye(3)*10,pbc=True)
    meter=Meter(DoubleWell());meter.bind(ref,0.);return ref,meter


def test_cached_CF_generalized_gradient_converts_to_fixed_cell_Cartesian_direction():
    g=np.r_[np.array([[2.,0,0],[-2.,0,0]]).ravel()*np.sqrt(2),np.ones(6)]
    result=directions(g,2,0)
    assert np.allclose(result['descent'],np.array([[-1,0,0],[1,0,0]])/np.sqrt(2))
    assert np.allclose(result['reverse'],-result['descent'])
    for d in result.values():assert np.isclose(np.linalg.norm(d),1) and np.allclose(d.mean(0),0)
    assert directions(np.r_[np.zeros(6),np.ones(6)],2,0) is None
    with pytest.raises(ValueError):directions(np.zeros(3),2,0)


def test_centroid_gauge_and_basis_remove_only_global_translation():
    ref,_=fixture();positions=ref.positions.copy()+[.1,.2,.3];positions+=np.array([[.03,0,0],[-.03,0,0]])
    constraint=FixCentroid();constraint.adjust_positions(ref,positions)
    assert np.allclose(positions.mean(0),ref.positions.mean(0))
    assert np.allclose(positions[1]-positions[0],ref.positions[1]-ref.positions[0]+[-.06,0,0])
    forces=np.array([[2.,3,4],[-1.,1,2]]);relative=forces[1]-forces[0];constraint.adjust_forces(ref,forces)
    assert np.allclose(forces.mean(0),0) and np.allclose(forces[1]-forces[0],relative)
    basis=centroid_basis(4);assert basis.shape==(12,9) and np.allclose(basis.T@basis,np.eye(9))
    assert np.allclose(basis.reshape(4,3,9).sum(0),0)


def test_geometry_and_actual_calculator_budget_fail_without_hidden_evaluations():
    ref,meter=fixture();bad=ref.copy();bad.positions[0,0]+=2.1
    with pytest.raises(Rejected):geometry_guard(bad,ref)
    bad=ref.copy();bad.set_cell(np.eye(3)*11)
    with pytest.raises(Rejected):geometry_guard(bad,ref)
    meter.bind(ref,0.,cap=0)
    with pytest.raises(BudgetExceeded):point(ref,meter.calc)
    assert meter.calls==0


def test_two_scale_Hessian_certifies_known_index_one_saddle():
    ref,meter=fixture();saddle=ref.copy();saddle.positions[0,0]= -1.5;saddle.positions[1,0]=1.5
    proof,mode=certify(saddle,meter.calc)
    assert proof['passed'] and all(c['resolved_negative']==1 for c in proof['checks'])
    assert np.isclose(proof['checks'][0]['eigenvalues'][0],-8.,atol=.001)
    assert np.allclose(mode.mean(0),0)
    minimum,_=certify(ref,meter.calc);assert not minimum['passed']


def test_actual_dimer_crosses_doublewell_and_both_exit_quenches_remain_counted():
    ref,meter=fixture();mode=np.array([[-1.,0,0],[1.,0,0]])/np.sqrt(2)
    result,exits=activate(ref,meter,mode,0.)
    assert result['status']=='certified_candidate' and result['certificate']['passed'] and len(exits)==2
    assert result['NFE']<=POLICY['NFE_cap_per_activation'] and all(t['raw_fmax']>=0 for t in result['trace'])
    outcomes=[]
    for _,atoms in exits:
        atoms.calc=meter.calc
        with BFGS(atoms,logfile=None,maxstep=.05) as opt:assert opt.run(fmax=.001,steps=200)
        outcomes.append(atoms.positions[1,0]-atoms.positions[0,0]-3)
    assert min(outcomes)<-.99 and max(outcomes)>.99
    result,exits=activate(ref,meter,-mode,0.)
    assert result['status']=='rejected' and not exits


def test_all_twelve_attempts_keep_twenty_four_future_quench_slots(tmp_path):
    slots=[dict(r,status='activation_failed') for r in ledger(PARENTS)]
    attempts=[dict(parent_id=p,arm=a,result=dict(status='rejected',NFE=1)) for p in PARENTS for a in ARMS]
    data=dict(slots=slots,attempts=attempts,preflight_NFE=4,NFE=16);dimer_native.validate_inventory(data)
    bad=copy.deepcopy(data);bad['slots'].pop()
    with pytest.raises(ValueError,match='24'):dimer_native.validate_inventory(bad)
    bad=copy.deepcopy(data);bad['attempts'].pop()
    with pytest.raises(ValueError,match='12'):dimer_native.validate_inventory(bad)
    bad=copy.deepcopy(data);bad['NFE']=15
    with pytest.raises(ValueError,match='ledger'):dimer_native.validate_inventory(bad)
    one=copy.deepcopy(data);path=tmp_path/'exit.vasp';path.write_text('native schema fixture')
    one['attempts'][0]['result'].update(status='certified_candidate',certificate=dict(passed=True),exit_signs_ready=[-1])
    one['slots'][0].update(status='candidate_ready',input=str(path),input_sha256=dimer_native.sha(path))
    dimer_native.validate_inventory(one)
    one['attempts'][0]['result']['exit_signs_ready']=[1]
    with pytest.raises(ValueError,match='status'):dimer_native.validate_inventory(one)


def test_native_CLI_imports_fresh_and_refuses_non_Slurm_model_work(tmp_path):
    file=str(Path(dimer_native.__file__).resolve());env=dict(os.environ);env.pop('SLURM_JOB_ID',None);env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,file,'--help'],cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    assert '--cache' in result.stdout
    result=subprocess.run([sys.executable,file,'--source','missing','--cache','missing','--run',str(tmp_path/'never')],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==2 and 'Use Slurm CPU' in result.stderr and not (tmp_path/'never').exists()
