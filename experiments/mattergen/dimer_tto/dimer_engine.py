"""Bounded counted ASE dimer activation and two-scale projected Hessian checks."""
import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.mep import DimerControl,MinModeAtoms,MinModeTranslate
from ase.neighborlist import neighbor_list
from dimer_policy import POLICY


class Rejected(ValueError):pass
class BudgetExceeded(Rejected):pass


class FixCentroid(FixConstraint):
    def get_removed_dof(self,atoms):return 3
    def adjust_positions(self,atoms,newpositions):newpositions-=(newpositions-atoms.positions).mean(0)
    def adjust_forces(self,atoms,forces):forces-=forces.mean(0)


def geometry_guard(atoms,reference):
    if (not np.array_equal(atoms.numbers,reference.numbers) or not np.allclose(atoms.cell,reference.cell,atol=1e-12,rtol=0)
        or not np.isfinite(atoms.positions).all() or np.linalg.norm(atoms.positions-reference.positions,axis=1).max()>POLICY['max_offset_A']
        or len(neighbor_list('d',atoms,POLICY['minimum_distance_A']-1e-9))):raise Rejected('Activation geometry guard.')


class Meter:
    def __init__(self,calculator):
        self.calc=calculator;self.calls=0;self.reference=None;self.energy=None;self.cap=None;self.start=0;self.raw=calculator.calculate
        calculator.calculate=self.calculate
    def bind(self,reference,energy,cap=None):
        self.reference=reference.copy();self.energy=float(energy);self.cap=cap;self.start=self.calls
    def calculate(self,atoms=None,*args,**kwargs):
        target=atoms if atoms is not None else self.calc.atoms
        if self.reference is not None:geometry_guard(target,self.reference)
        if self.cap is not None and self.calls-self.start>=self.cap:raise BudgetExceeded('Activation calculator budget exhausted.')
        self.calls+=1;result=self.raw(atoms,*args,**kwargs)
        energy=float(self.calc.results['energy']);forces=self.calc.results['forces']
        if not np.isfinite(energy) or not np.isfinite(forces).all():raise Rejected('Nonfinite physical result.')
        if self.energy is not None and (energy-self.energy)/len(target)>POLICY['intermediate_energy_ceiling_eV_atom']:raise Rejected('Activation energy ceiling.')
        return result


def point(reference,calc,offset=None):
    a=reference.copy();a.set_constraint(None)
    if offset is not None:a.positions+=np.asarray(offset)
    a.calc=calc;force=a.get_forces(apply_constraint=False)
    return float(a.get_potential_energy()),np.asarray(force)


def directional_curvature(reference,calc,mode):
    values=[]
    for eps in POLICY['Hessian_RMS_steps_A']:
        length=eps*np.sqrt(len(reference));_,plus=point(reference,calc,length*mode);_,minus=point(reference,calc,-length*mode)
        values.append(float(-np.sum((plus-minus)*mode)/(2*length)))
    error=abs(values[0]-values[1]);passed=error<=POLICY['source_curvature_absolute']+POLICY['source_curvature_relative']*max(abs(x) for x in values)
    return dict(values=values,error=error,passed=bool(passed))


def centroid_basis(natoms):
    translations=np.zeros((3*natoms,3))
    for axis in range(3):translations[axis::3,axis]=1/np.sqrt(natoms)
    return np.linalg.svd(translations.T,full_matrices=True)[2][3:].T


def certify(reference,calc):
    basis=centroid_basis(len(reference));matrices=[];records=[]
    for eps in POLICY['Hessian_RMS_steps_A']:
        length=eps*np.sqrt(len(reference));columns=[]
        for v in basis.T:
            _,plus=point(reference,calc,length*v.reshape(-1,3));_,minus=point(reference,calc,-length*v.reshape(-1,3))
            columns.append(-basis.T@(plus-minus).ravel()/(2*length))
        raw=np.asarray(columns).T;asym=float(np.linalg.norm(raw-raw.T)/max(np.linalg.norm(raw),1e-12));H=(raw+raw.T)/2
        eigenvalues,vectors=np.linalg.eigh(H);matrices.append(H)
        records.append(dict(eps_RMS_A=eps,matrix=H.tolist(),asymmetry=asym,eigenvalues=eigenvalues.tolist(),resolved_negative=int(sum(eigenvalues<POLICY['negative_curvature_threshold_eV_A2']))))
    error=float(np.linalg.norm(matrices[0]-matrices[1]));allowed=POLICY['Hessian_agreement_absolute']+POLICY['Hessian_agreement_relative']*max(np.linalg.norm(h) for h in matrices)
    passed=all(r['asymmetry']<=POLICY['Hessian_asymmetry_max'] and r['resolved_negative']==1 for r in records) and error<=allowed
    _,vectors=np.linalg.eigh(matrices[0]);mode=(basis@vectors[:,0]).reshape(-1,3)
    return dict(passed=bool(passed),checks=records,two_scale_error=error,two_scale_allowed=float(allowed)),mode


def activate(reference,meter,mode,reference_energy):
    meter.bind(reference,reference_energy,POLICY['NFE_cap_per_activation'])
    atoms=reference.copy();atoms.calc=meter.calc;atoms.set_constraint(FixCentroid());trace=[];status='step_limit';exits=[];certificate=None;alignment=None;exit_errors={}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            with DimerControl(initial_eigenmode_method='displacement',displacement_method='vector',logfile=None,eigenmode_logfile=None,
                dimer_separation=POLICY['dimer_separation_RMS_A']*np.sqrt(len(atoms)),maximum_translation=POLICY['max_translation_A'],
                max_num_rot=POLICY['max_num_rot'],f_rot_min=POLICY['rotation_fmin'],f_rot_max=POLICY['rotation_fmax'],mask=[True]*len(atoms)) as control:
                dimer=MinModeAtoms(atoms,control,random_seed=POLICY['seed']);dimer.displace(displacement_vector=POLICY['initial_RMS_A']*np.sqrt(len(atoms))*mode)
                with MinModeTranslate(dimer,logfile=None) as optimizer:
                    for _ in optimizer.irun(fmax=POLICY['saddle_raw_fmax'],steps=POLICY['steps']):
                        current=np.asarray(dimer.get_eigenmode());raw=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max())
                        if current.shape!=mode.shape or not np.isfinite(current).all() or abs(np.linalg.norm(current)-1)>1e-6 or np.abs(current.mean(0)).max()>1e-6:raise Rejected('Dimer mode normalization/centroid gauge lost.')
                        trace.append(dict(positions=atoms.positions.tolist(),energy_eV=float(atoms.get_potential_energy()),raw_fmax=raw,
                            curvature=float(dimer.get_curvature()),mode_alignment=float(abs(np.sum(mode*current))),NFE=meter.calls-meter.start))
                steps=optimizer.nsteps
                if trace[-1]['raw_fmax']<=POLICY['saddle_raw_fmax'] and trace[-1]['curvature']<POLICY['negative_curvature_threshold_eV_A2']:
                    certificate,escape=certify(atoms,meter.calc)
                    if certificate['passed']:
                        status='certified_candidate';alignment=float(abs(np.sum(mode*escape)))
                        for sign in (-1,1):
                            a=atoms.copy();a.set_constraint(None);a.positions+=sign*POLICY['exit_RMS_A']*np.sqrt(len(a))*escape
                            try:geometry_guard(a,reference)
                            except Rejected as problem:exit_errors[str(sign)]=repr(problem)
                            else:exits.append((sign,a))
                    else:status='unresolved_saddle'
        error=None
    except (Rejected,RuntimeWarning,FloatingPointError) as problem:
        status='budget_exhausted' if isinstance(problem,BudgetExceeded) else 'rejected';error=repr(problem);steps=max(0,len(trace)-1);exits=[]
    return dict(status=status,error=error,steps=steps,trace=trace,certificate=certificate,certified_mode_alignment=alignment,
        exit_signs_ready=[sign for sign,_ in exits],exit_rejections=exit_errors,NFE=meter.calls-meter.start,last_center=atoms.positions.tolist()),exits
